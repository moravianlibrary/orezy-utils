import argparse
import json
import time
from io import BytesIO
import os
from PIL import Image
from urllib.parse import urljoin
import cv2
import requests


Image.MAX_IMAGE_PIXELS = 933120000


class PageTracer:
    """Automatically crops document pages from images using the Page Trace API.
    Image thumbnails are sent via API for prediction. The returned predictions are then
    used to crop the original high-resolution images.
    """

    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.token = token
        self.web_url = "https://orezy.test.trinera.cloud"
        self.id = None
        self.coordinates = None

    def upload_and_compress(self, input_folder: str, crop_type: str):
        """Uploads and compresses images to the Page Trace API.

        Args:
            input_folder (str): Path to the folder containing input images.
            crop_type (str): Type of crop to perform ("inner" or "outer").
        """
        response = requests.post(
            url=urljoin(self.api_url, "create"),
            headers={"Authorization": f"Bearer {self.token}"},
            json={"crop_type": crop_type},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to create job: {response.text}")
        self.id = response.json()["id"]

        images = sorted(os.listdir(input_folder))
        print(f"Uploading {len(images)} images...")
        for img in images:
            with open(os.path.join(input_folder, img), "rb") as f:
                im = Image.open(f)
                im.thumbnail((1024, 1024))
                buf = BytesIO()
                fmt = im.format or "JPEG"
                im.save(buf, format=fmt)
                buf.seek(0)
                response = requests.post(
                    url=urljoin(self.api_url, f"{self.id}/upload-scan"),
                    headers={"Authorization": f"Bearer {self.token}"},
                    files={"scan_data": (img, buf, "image/jpeg")},
                )

                if response.status_code != 200:
                    raise Exception(f"Failed to upload image {img}: {response.text}")

    def process(self):
        """Queues the uploaded images for ML processing and waits for completion.
        Returns URL to view results.
        """
        print("Predicting page crop coordinates...")
        response = requests.post(
            url=urljoin(self.api_url, f"{self.id}/process"),
            headers={"Authorization": f"Bearer {self.token}"},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to start processing: {response.text}")

        state = requests.get(
            url=urljoin(self.api_url, f"{self.id}/status"),
            headers={"Authorization": f"Bearer {self.token}"},
        )
        while state.json() != "ready":
            time.sleep(10)
            state = requests.get(
                url=urljoin(self.api_url, f"{self.id}/status"),
                headers={"Authorization": f"Bearer {self.token}"},
            )
            print(f"Current status: {state.text}")
        if state.status_code != 200:
            raise Exception(f"Failed to check status: {state.text}")

        print("Results are available at", urljoin(self.web_url, f"book/{self.id}"))
        input("Press Enter to confirm and download...")

    def download_results(self, output_folder: str):
        """Downloads the predicted crop coordinates from the API, saves them to disk.

        Args:
            output_folder (str): Path to the folder to save the coordinates JSON.
        """
        response = requests.get(
            url=urljoin(self.api_url, f"{self.id}/scans"),
            headers={"Authorization": f"Bearer {self.token}"},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to download results: {response.text}")

        self.coordinates = response.json()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, "coordinates.json"), "w") as f:
            f.write(json.dumps(self.coordinates, indent=4))

    def crop_documents(self, input_folder: str, output_folder: str):
        """Crops the original high-resolution images using the downloaded coordinates.
        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder to save cropped images.
        """
        images = sorted(os.listdir(input_folder))
        for img_name, coordinate in zip(images, self.coordinates):
            im = cv2.imread(os.path.join(input_folder, img_name))
            h, w = im.shape[0], im.shape[1]
            for i, page in enumerate(coordinate["pages"]):
                output_image = im.copy()

                xc = page["xc"] * w
                yc = page["yc"] * h
                ww = page["width"] * w
                hh = page["height"] * h
                # rotate image around center
                M = cv2.getRotationMatrix2D((xc, yc), page["angle"], 1.0)
                output_image = cv2.warpAffine(
                    output_image, M, (w, h), flags=cv2.INTER_CUBIC
                )
                # crop image
                output_image = output_image[
                    int(yc - hh / 2) : int(yc + hh / 2),
                    int(xc - ww / 2) : int(xc + ww / 2),
                ]
                # save to disk
                output_path = os.path.join(
                    output_folder, f"{os.path.splitext(img_name)[0]}_page{i + 1}.jpg"
                )
                cv2.imwrite(output_path, output_image)

        print(f"Success! Cropped images saved to {output_folder}")

    def run(self, input_folder: str, output_folder: str, crop_type: str):
        """Runs the full Page Trace process: upload, process, download, and crop.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder to save cropped images.
            crop_type (str): Type of crop to perform ("inner" or "outer").
        """
        self.upload_and_compress(input_folder, crop_type)
        self.process()
        self.download_results(output_folder)
        self.crop_documents(input_folder, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Page Trace on a folder of images and convert them to cropped pages."
    )
    parser.add_argument(
        "--token",
        type=str,
        required=False,
        help="API token",
        default=os.environ.get("BASIC_AUTH_TOKEN", ""),
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=False,
        help="Input folder path",
        default=os.environ.get("SCAN_DATA_PATH", ""),
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=False,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--crop-type", type=str, required=False, help="inner / outer", default="inner"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=False,
        help="API URL",
        default="https://api.ai-orezy.trinera.cloud",
    )
    args = parser.parse_args()

    print(f"Starting Page Tracer for folder {args.input_folder}...")

    tracer = PageTracer(api_url=args.api_url, token=args.token)
    tracer.run(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        crop_type=args.crop_type,
    )
