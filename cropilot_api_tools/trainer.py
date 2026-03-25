import argparse
import io
from PIL import Image, ImageOps
from urllib.parse import urljoin
import comet_ml
import os
import random
import requests
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


comet_ml.login(
    project_name="crop-finetune-domain-specific", api_key=os.getenv("COMET_ML_API_KEY")
)

class CropilotTrainer:
    """Automates the process of fine-tuning a YOLO model for Cropilot.
    """

    def __init__(self, api_url: str, api_key: str, base_model: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.base_model = base_model
        self.model_name = model_name
        
        self.directory = f"finetune_dataset_{model_name}"

        self.authenticate()

    def authenticate(self) -> str:
        """Authenticates with the API and returns an access token.

        Args:
            api_url (str): Base URL of the API.
            username (str): API username.
            password (str): API password.

        Returns:
            str: Access token.
        """
        try:
            response = requests.get(
                url=urljoin(self.api_url, "/groups"),
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            group = response.json()[0]
            self.group_id = group["_id"]
        except Exception as e:
            raise Exception("Failed to authenticate. Please check your API key.") from e

        logger.info(f"Successfully authenticated to group: {group['name']}")

    def train_job(self, title_ids: list[str]):
        """
        Main training job orchestrator.
        """
        os.mkdir(self.directory)
        os.mkdir(f"{self.directory}/images")
        os.mkdir(f"{self.directory}/labels")
        os.mkdir(f"{self.directory}/images/train")
        os.mkdir(f"{self.directory}/images/val")
        os.mkdir(f"{self.directory}/labels/train")
        os.mkdir(f"{self.directory}/labels/val")

        try:
            for title_id in title_ids:
                self.download_training_data(title_id)
                self.split_train_val()
            self.create_dataset_yaml()
            self.create_finetuned_model()
            self.upload_trained_model()
        except Exception as e:
            logger.info(f"Error occurred during training job: {e}")
            self.cleanup()
        
        logger.info("Training job completed successfully.")
        
        self.cleanup()
        
    def upload_trained_model(self):
        """Uploads the trained model to the API."""
        model_path = [f for f in os.listdir("crop-finetune-domain-specific") if f.startswith(self.model_name)]
        model_path = sorted(model_path)[-1]
        model_path = os.path.join("crop-finetune-domain-specific", model_path, "weights", "best.pt")
        os.rename(model_path, f"{self.model_name}.pt")

        with open(f"{self.model_name}.pt", "rb") as f:
            response = requests.post(
                url=urljoin(self.api_url, "/models"),
                headers={"X-API-Key": self.api_key},
                files={"file": f}
            )
        response.raise_for_status()
        logger.info(f"Uploaded trained model: {self.model_name}")

    def download_training_data(self, title_id: str):
        """Downloads the training data (images and labels) for a given title ID.

        Args:
            title_id (str): The ID of the title to download data for.
        """
        response = requests.get(
            url=urljoin(self.api_url, f"{title_id}/scans"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()

        scans = response.json()["scans"]
        logger.info(f"Downloaded metadata for title {title_id}, found {len(scans)} scans.")
        for scan in scans:
            self.write_label(scan)
            self.save_scan_image(title_id, scan)

    def create_dataset_yaml(self):
        """Creates the dataset.yaml file required by YOLO."""
        with open(f"{self.directory}/dataset.yaml", "w") as f:
            f.write(f"path: {self.directory}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            f.write("    0: page\n")

    def write_label(self, scan):
        """Writes the label (bbox coordinates) file for a single scan.
        Args:
            scan (dict): The scan metadata containing page coordinates.
        """
        with open(f"{self.directory}/labels/{scan['_id']}.txt", "w") as f:
            if "no_prediction" in scan["flags"]:
                logger.debug(f"Scan {scan['_id']} is flagged as no_prediction, writing empty label file.")
                return
            for page in scan["pages"]:
                f.write(f"0 {page['xc']} {page['yc']} {page['width']} {page['height']}\n")

    def save_scan_image(self, title_id: str, scan: dict):
        """Downloads the image and saves it to disk."""
        response = requests.get(
            url=urljoin(self.api_url, f"{title_id}/files?scan_id={scan['_id']}"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()
        
        # if single page, rotate by page's angle
        if len(scan["pages"]) == 1 and abs(scan["pages"][0]["angle"]) > 1:
            logger.debug(f"Fixing rotation of {scan['_id']}, autorotating by {scan['pages'][0]['angle']} degrees")
            angle = scan["pages"][0]["angle"]
            image = Image.open(io.BytesIO(response.content))
            image = ImageOps.exif_transpose(image)

            # rotate with center of xc, yc
            xc, yc = scan["pages"][0]["xc"], scan["pages"][0]["yc"]
            image = image.rotate(angle, center=(xc, yc))
            image.save(f"{self.directory}/images/{scan['_id']}.jpg")
        else:
            with open(f"{self.directory}/images/{scan['_id']}.jpg", "wb") as f:
                f.write(response.content)

    def split_train_val(self):
        """Splits the dataset into training and validation sets (90% train, 10% val)
        Moves files into corresponding directories.
        """
        all_ids = [f.split(".")[0] for f in os.listdir(f"{self.directory}/images") if f.endswith(".jpg")]
        random.shuffle(all_ids)
        train_ids, val_ids = all_ids[: int(0.9 * len(all_ids))], all_ids[int(0.9 * len(all_ids)) :]

        logger.info(f"Split {len(all_ids)} samples into {len(train_ids)} train and {len(val_ids)} val samples.")
        
        for f in train_ids:
            os.rename(f"{self.directory}/images/{f}.jpg", f"{self.directory}/images/train/{f}.jpg")
            os.rename(f"{self.directory}/labels/{f}.txt", f"{self.directory}/labels/train/{f}.txt")
        for f in val_ids:
            os.rename(f"{self.directory}/images/{f}.jpg", f"{self.directory}/images/val/{f}.jpg")
            os.rename(f"{self.directory}/labels/{f}.txt", f"{self.directory}/labels/val/{f}.txt")
        
        logger.info("Finished splitting data into train and val sets.")

    def create_finetuned_model(self):
        """Creates a new training job for the specified model and dataset.

        Args:
            model (str): The name of the model to train (e.g., "yolo26s").
            name (str): A unique name for this training job.
        """
        model = YOLO(self.base_model)
        model.train(
            data=f"{self.directory}/dataset.yaml",
            project="crop-finetune-domain-specific",
            name=self.model_name,
            epochs=1000,
            imgsz=640,
            batch=16,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            close_mosaic=50,
            degrees=4,
            shear=1.0,
            save_json=True,
            max_det=2,
            single_cls=True,
        )
    
    def cleanup(self):
        """Cleans up resources used during the training job."""
        if os.path.exists(self.directory):
            for root, dirs, files in os.walk(self.directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                    logger.info(f"Removed directory {os.path.join(root, name)}")
            os.rmdir(self.directory)
            logger.info(f"Removed directory {self.directory}")
        logger.info("Cleanup completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="trainer.py")

    parser.add_argument(
        "--api_url",
        type=str,
        default="https://api.ai-orezy.trinera.cloud/",
        help="Base URL of the API (default: https://api.ai-orezy.trinera.cloud/)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="base_models/default.pt",
        help="Path to the base YOLO model to fine-tune (default: base_models/default.pt)",
    )
    parser.add_argument(
        "--api_key", type=str, help="Group API key for authentication"
    )
    parser.add_argument(
        "--model_name", type=str, help="New name of the fine-tuned model"
    )
    parser.add_argument(
        "--title_ids", nargs="+", default=[], help="List of title IDs to train on"
    )

    args = parser.parse_args()

    trainer = CropilotTrainer(
        api_url=args.api_url,
        api_key=args.api_key,
        base_model=args.base_model,
        model_name=args.model_name,
    )
    trainer.train_job(title_ids=args.title_ids)