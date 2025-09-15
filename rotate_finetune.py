import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import comet_ml
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# -------------------------------
#  Label-aware rotation transform
# -------------------------------
class RandomRotateWithLabel:
    """
    Rotates a PIL image by a random angle in [-max_deg, max_deg] with probability p.
    Returns (rotated_image, applied_angle_deg).
    If no rotation is applied, angle = 0.
    Uses white background fill (255) which is suitable for book pages.
    """

    def __init__(self, max_deg: float, p: float = 1.0):
        self.max_deg = float(max_deg)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, float]:
        if np.random.rand() > self.p:
            return img, 0.0
        angle = np.random.uniform(-self.max_deg, self.max_deg)
        # PIL>=8 supports 'fillcolor' arg; torchvision will pass it through
        rotated = img.rotate(angle, expand=False, fillcolor=0)
        return rotated, float(angle)


# -------------------------------
#  Dataset
# -------------------------------
class PageAngleDataset(Dataset):
    """
    CSV columns required:
      - For label-mode 'csv': filename, angle_deg
      - For label-mode 'augmented': filename (angle column ignored if present)

    Targets are degrees in [-angle_max, +angle_max].
    """

    def __init__(
        self,
        images_dir: str,
        image_size: int = 640,
        is_train: bool = True,
        angle_max: float = 10.0,
        aug_rotate_prob: float = 1.0,
    ):
        self.is_train = is_train
        self.images_dir = images_dir
        self.angle_max = float(angle_max)
        self.aug_rotate_prob = float(aug_rotate_prob)
        self.df = pd.DataFrame(
            {
                "filename": sorted(
                    [
                        f
                        for f in os.listdir(images_dir)
                        if f.endswith((".png", ".jpg", ".jpeg", ".tif"))
                    ]
                )
            }
        )

        # We'll apply rotation on the PIL image FIRST (label-aware), then resize/crop/normalize.
        self.rotate_tf = RandomRotateWithLabel(
            max_deg=self.angle_max,  # no rotation in val
            p=self.aug_rotate_prob,
        )

        if self.is_train:
            self.after_tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.6, 0.8), ratio=(1, 1)
                    ),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                ]
            )
        else:
            self.after_tf = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 1) Label by adding rotation
        img_rot, target_deg = self.rotate_tf(img_pil)

        # 3) Augment
        img_t = self.after_tf(img_rot)

        # Return tensor target shape (1,)
        return img_t, torch.tensor([target_deg], dtype=torch.float32), target_deg


# -------------------------------
#  Model
# -------------------------------
class DegreeHead(nn.Module):
    """Features -> degrees in [-angle_max, angle_max] via tanh bound."""

    def __init__(self, in_features: int, angle_max: float = 10.0):
        super().__init__()
        self.angle_max = float(angle_max)
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        raw = self.net(x)
        return torch.tanh(raw) * self.angle_max


class AngleDegModel(nn.Module):
    def __init__(self, angle_max: float = 10.0):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        in_feats = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.head = DegreeHead(in_feats, angle_max)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # (B,1)

    def predict_image(self, img_bgr: np.ndarray) -> float:
        """Predict angle for a single image (numpy BGR)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tf = transforms.Compose(
            [
                transforms.Resize(640),
                transforms.CenterCrop(640),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            ]
        )
        img_t = tf(img_pil).unsqueeze(0)  # (1,3,H,W)
        self.eval()
        with torch.no_grad():
            output = self.forward(img_t)
        angle = output.detach().cpu().numpy().reshape(-1)[0]
        return float(angle)


# -------------------------------
#  Training utils
# -------------------------------
@dataclass
class TrainConfig:
    images_train: str
    images_val: str
    out_dir: str = "checkpoints_deg"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    resume: str = ""
    angle_max: float = 10.0
    aug_rotate_prob: float = 1.0


def mae_deg(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    total_loss = 0.0
    for imgs, targets, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # (B,1) degrees

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)  # (B,1) in [-angle_max, angle_max]
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for imgs, targets, true_deg in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * imgs.size(0)

        preds.append(outputs.detach().cpu().numpy().reshape(-1))
        trues.append(np.asarray(true_deg, dtype=np.float32))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), mae_deg(preds, trues)


def save_checkpoint(path, model, optimizer, epoch, best_mae):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_mae,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_mae", float("inf"))


def main():
    comet_ml.login(
        project_name="autorotate_finetune", api_key=os.getenv("COMET_ML_API_KEY")
    )
    experiment = comet_ml.Experiment(project_name="autorotate_finetune")

    ap = argparse.ArgumentParser()
    ap.add_argument("--images-train", default="datasets/yolo2/images/train", type=str)
    ap.add_argument("--images-val", default="datasets/yolo2/images/val", type=str)
    ap.add_argument("--out-dir", default="checkpoints_deg", type=str)
    ap.add_argument("--image-size", default=640, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--weight-decay", default=1e-4, type=float)
    ap.add_argument("--num-workers", default=4, type=int)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default="", type=str)
    ap.add_argument("--angle-max", default=10.0, type=float)
    ap.add_argument(
        "--aug-rotate-prob",
        default=1.0,
        type=float,
        help="probability to apply rotation on train",
    )
    ap.add_argument(
        "--aug-rotate-max",
        default=None,
        type=float,
        help="max |deg| for augmentation rotation; default = angle_max",
    )
    args = ap.parse_args()

    cfg = TrainConfig(
        images_train=args.images_train,
        images_val=args.images_val,
        out_dir=args.out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        resume=args.resume,
        angle_max=args.angle_max,
        aug_rotate_prob=args.aug_rotate_prob,
    )

    device = torch.device("mps")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = PageAngleDataset(
        cfg.images_train,
        cfg.image_size,
        is_train=True,
        angle_max=cfg.angle_max,
        aug_rotate_prob=cfg.aug_rotate_prob,
    )
    val_ds = PageAngleDataset(
        cfg.images_val,
        cfg.image_size,
        is_train=False,
        angle_max=cfg.angle_max,
        aug_rotate_prob=cfg.aug_rotate_prob,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model, optimizer, loss
    model = AngleDegModel(cfg.angle_max).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.HuberLoss(delta=1.0)

    scaler = torch.amp.GradScaler(device=device)

    start_epoch = 0
    best_mae = float("inf")
    if cfg.resume:
        print(f"Resuming from {cfg.resume}")
        start_epoch, best_mae = load_checkpoint(cfg.resume, model, optimizer, device)
        print(f"Resumed epoch={start_epoch}, best_mae={best_mae:.3f}")

    ckpt_best = os.path.join(cfg.out_dir, "best.pth")
    ckpt_last = os.path.join(cfg.out_dir, "last.pth")

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion
        )
        val_loss, val_mae = evaluate(model, val_loader, device, criterion)
        print(
            f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | Val MAE: {val_mae:.3f}°"
        )

        # Log to Comet.ml
        experiment.log_metric("train/loss", train_loss, step=epoch)
        experiment.log_metric("val/loss", val_loss, step=epoch)
        experiment.log_metric("val/mae", val_mae, step=epoch)

        # Save last
        save_checkpoint(ckpt_last, model, optimizer, epoch + 1, best_mae)

        # Save best
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(ckpt_best, model, optimizer, epoch + 1, best_mae)
            print(f"New best MAE: {best_mae:.3f}° -> saved {ckpt_best}")


if __name__ == "__main__":
    # main()

    # load and test the model
    model = AngleDegModel(angle_max=10.0)
    ckpt_path = "checkpoints_deg/best.pth"
    load_checkpoint(ckpt_path, model, map_location="cpu")
    images_path = "datasets/yolo2/images/val"
    dataset = PageAngleDataset(
        images_path, image_size=640, is_train=True, angle_max=10.0, aug_rotate_prob=1.0
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # process and display with streamlit
    import streamlit as st

    st.title("Rotation Model Inference")
    model.eval()
    for imgs, _, true_deg in tqdm(loader, desc="Inference"):
        with torch.no_grad():
            outputs = model(imgs)
        pred_deg = outputs.detach().cpu().numpy().reshape(-1)[0]
        st.write(f"Predicted angle: {pred_deg:.2f}°, True angle: {true_deg[0]:.2f}°")
        image = imgs[0].permute(1, 2, 0).numpy()
        image = (image * 0.25) + 0.5  # unnormalize
        image = (image * 255).astype(np.uint8)
        st.image(image, width=400)
