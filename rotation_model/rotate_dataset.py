import random
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

import cv2
import torch


class PageAngleDataset(Dataset):
    """ """

    def __init__(
        self,
        image_paths: list[str],
        image_bboxes: list[tuple[float, float, float, float]],
        image_size: int = 640,
        is_train: bool = True,
        angle_max: float = 10.0,
        aug_rotate_prob: float = 0.9,
    ):
        self.image_paths = image_paths
        self.image_bboxes = image_bboxes
        self.image_size = image_size
        self.is_train = is_train
        self.angle_max = angle_max
        self.aug_rotate_prob = aug_rotate_prob

        self.normalize_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _denormalize_bbox(
        self, bbox: tuple[float, float, float, float], img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        return (
            int(bbox[0] * img_w),
            int(bbox[1] * img_h),
            int(bbox[2] * img_w),
            int(bbox[3] * img_h),
        )

    def _cxywh_to_xyxy(
        self, xc: int, yc: int, w: int, h: int
    ) -> tuple[int, int, int, int]:
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)
        return x1, y1, x2, y2

    def _rotate_around_center(
        self, img: np.ndarray, angle: float, xc: int, yc: int
    ) -> np.ndarray:
        # Compute rotation matrix
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1.0)
        # Perform affine warp (rotation)
        rotated = cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return rotated

    def _add_jitter(self, x1: int, y1: int, x2: int, y2: int, jitter_scale: int = 20) -> tuple[int, int, int, int]:
        """Add random jitter to the bounding box coordinates."""
        x1 += random.randint(-jitter_scale, jitter_scale)
        y1 += random.randint(-jitter_scale, jitter_scale)
        x2 += random.randint(-jitter_scale, jitter_scale)
        y2 += random.randint(-jitter_scale, jitter_scale)
        return x1, y1, x2, y2

    def _resize_letterbox_pad(self, img: np.ndarray, size: int) -> np.ndarray:
        # Resize image with letterbox padding to keep aspect ratio
        # +---------------------------+
        # |         gray pad          |
        # |+-------------------------+|
        # ||        resized          ||
        # ||        640×480          ||
        # |+-------------------------+|
        # |         gray pad          |
        # +---------------------------+
        h, w = img.shape[:2]
        scale = size / max(h, w)
        #print("Resizing with scale:", scale, "for image size:", (h, w))
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))
        vertical_pad = (size - nh) // 2
        horizontal_pad = (size - nw) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized,
            vertical_pad,
            vertical_pad,
            horizontal_pad,
            horizontal_pad,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return cv2.resize(img_padded, (size, size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        # Binarize images
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # small opening to remove isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        img = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

        angle = 0.0
        # Load crop coordinates
        img_h, img_w, _ = img.shape
        xc, yc, w, h = self._denormalize_bbox(self.image_bboxes[idx], img_w, img_h)

        # Rotate image
        if self.is_train and random.random() < self.aug_rotate_prob:
            angle = random.uniform(-self.angle_max, self.angle_max)
            img = self._rotate_around_center(img, angle, xc, yc)

        # Crop
        x1, y1, x2, y2 = self._cxywh_to_xyxy(xc, yc, w, h)
        if self.is_train:
            x1, y1, x2, y2 = self._add_jitter(x1, y1, x2, y2, jitter_scale=int(0.03 * img_w))

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x1 != x2 and y1 != y2:
            img = img[y1:y2, x1:x2]

        # Resize with padding
        img = self._resize_letterbox_pad(img, self.image_size)

        # Normalize and convert to tensor
        img_pil = Image.fromarray(img)
        img_t = self.normalize_tf(img_pil)
        return img_t, torch.tensor([angle], dtype=torch.float32), angle


if __name__ == "__main__":
    import streamlit as st

    # Example usage
    dataset = PageAngleDataset(
        image_paths=[
            "datasets/yolo-split_per_book/images/train/2619025743-2_0003_p1.jpg",
            "datasets/yolo-split_per_book/images/train/2610065249-1_0008_p1.jpg",
            "datasets/yolo-split_per_book/images/train/2610065249-1_0009_p1.jpg",
            "datasets/yolo-split_per_book/images/train/2610065249-1_0010_p1.jpg",
            "datasets/yolo-split_per_book/images/train/2610065249-1_0012_p1.jpg",
            "datasets/yolo-split_per_book/images/train/2610065249-1_0012_p1.jpg",
        ],
        image_bboxes=[(0.5, 0.5, 0.9, 0.9),
                      (0.5, 0.5, 0.6, 0.6),
                      (0.5, 0.5, 0.6, 0.6),
                      (0.5, 0.5, 0.6, 0.6),
                      (0.5, 0.5, 0.6, 0.6),
                      (0.5, 0.5, 0.9, 0.9)],
        image_size=640,
        is_train=True,
        angle_max=10.0,
        aug_rotate_prob=0.9,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch_idx, (imgs, tensors, angles) in enumerate(dataloader):
        # imgs: (B, C, H, W), angles: (B, 1)
        imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()  # → (B, H, W, C)
        imgs_np = np.clip(
            (imgs_np * 0.25) + 0.5, 0, 1
        )  # inverse normalize: std=0.25, mean=0.5

        st.subheader(f"Batch {batch_idx + 1}")

        # Display all images in the batch side by side
        cols = st.columns(len(imgs_np))
        for i, (img, angle) in enumerate(zip(imgs_np, angles)):
            img_disp = (img * 255).astype(np.uint8)
            cols[i].image(
                img_disp,
                caption=f"Angle: {angle.item():.2f}°",
                use_container_width=True,
            )
