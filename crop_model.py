import logging
import os
import cv2
from ultralytics import YOLO
import streamlit as st

from utils import add_margin, bbox_from_image_contours
import numpy as np

logger = logging.getLogger(__name__)

CROP_MODEL = YOLO("crop_finetune_model/train2/weights/best.pt", task="detect")


def crop_images_outer(input_folder):
    """Crops images in the input folder by finding the largest contour."""
    files = sorted(os.listdir(input_folder))
    files = [os.path.join(input_folder, f) for f in files if f.endswith((".tif"))]

    results = []
    for file in files:
        image = cv2.imread(file)
        w, h = image.shape[1], image.shape[0]
        outer_box = bbox_from_image_contours(image)
        outer_box = add_margin(outer_box, margin=(w * 0.01, h * 0.01))  # Add 1% margin
        # Cap to image size
        outer_box = [
            max(0, int(outer_box[0])),
            max(0, int(outer_box[1])),
            min(w, int(outer_box[2])),
            min(h, int(outer_box[3])),
        ]

        result = {
            "image_path": file,
            "x_center": (outer_box[0] + outer_box[2]) / 2 / w,
            "y_center": (outer_box[1] + outer_box[3]) / 2 / h,
            "width": (outer_box[2] - outer_box[0]) / w,
            "height": (outer_box[3] - outer_box[1]) / h,
            "confidence": 1.0,
        }
        results.append(result)

        logger.debug(f"Cropped image {file} to box: {outer_box}")
    return results


def crop_images_inner(input_folder, batch_size=16):
    """Crops images in the input folder using the trained YOLO model."""
    files = sorted(os.listdir(input_folder))
    files = [os.path.join(input_folder, f) for f in files if f.lower().endswith((".tif", ".tiff", ".jpg"))]

    logger.info(f"Found {len(files)} images in {input_folder} to crop.")

    results = []
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        batch_result = CROP_MODEL.predict(
            batch, conf=0.1, iou=0, max_det=2, agnostic_nms=True
        )

        for yolo_result in batch_result:
            for box in reversed(yolo_result.boxes):
                w_orig, h_orig = np.array(
                    yolo_result.orig_img.shape[1::-1], dtype=np.float32
                )
                if box:
                    logger.debug(
                        f"Cropped image {yolo_result.path} to box: {box.xyxy[0].cpu().numpy()}"
                    )
                    xc, yc, w, h = box.xywh[0].cpu().numpy()

                    # normalize by dividing by image width and height
                    xc /= w_orig
                    yc /= h_orig
                    w /= w_orig
                    h /= h_orig
                else:
                    xc, yc, w, h = (w_orig / 2, h_orig / 2, w_orig, h_orig)

                result = {
                    "image_path": yolo_result.path,
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(w),
                    "height": float(h),
                    "confidence": float(box.conf.item()) if box else 0,
                    "flags": ["low_confidence"] if box and box.conf.item() < 0.7 else [],
                }
                results.append(result)
    
    # Add entries for files not detected by YOLO (not in results)
    detected_paths = {res["image_path"] for res in results}
    for file in files:
        if file not in detected_paths:
            result = {
                "image_path": file,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 1.0,
                "height": 1.0,
                "confidence": 0,
            }
            results.append(result)
    return results


if __name__ == "__main__":
    folder = os.getenv("SCAN_DATA_PATH")
    files = sorted(os.listdir(folder))[:100]
    files = [os.path.join(folder, f) for f in files if f.endswith((".tif"))]

    results = []
    batch_size = 16
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        batch_result = CROP_MODEL.predict(
            batch, conf=0.1, iou=0, max_det=2, agnostic_nms=True
        )

        for result in batch_result:
            results.append(result)

    # display results
    st.title("AutoCrop Visualization")
    for result in results:
        im = result.plot()
        outer_box = bbox_from_image_contours(result.orig_img)
        outer_box = add_margin(outer_box, margin=(10, 10))
        cv2.rectangle(
            im,
            (int(outer_box[0]), int(outer_box[1])),
            (int(outer_box[2]), int(outer_box[3])),
            (0, 255, 0),
            3,
        )

        confidence = result.boxes.conf[0] if result.boxes else 0

        st.image(im, width=500, caption=f"Confidence: {confidence:.2f}")
