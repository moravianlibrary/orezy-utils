import os
import cv2
import streamlit as st
from crop_model import crop_images_inner, crop_images_outer
from rotate_finetune import AngleDegModel, load_checkpoint
from rotate_model import get_skew_angle_hough
from utils import xywh_to_xyxy_denorm
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ROTATION_MODEL = AngleDegModel(angle_max=10.0)
ckpt_path = "checkpoints_deg/best.pth"
load_checkpoint(ckpt_path, ROTATION_MODEL, map_location="cpu")


def run_crop_pipeline(title: str, crop_method="inner", rotation_method="hough"):
    """Runs a pipeline which splits multi-page images, deskews them, and crops them.
    Two methods for deskewing are supported: "hough" uses classical image processing,
    "ml" uses a trained ResNET model.

    Args:
        title (str): Path to the folder containing images to process.
        method (str): "hough" or "ml" for deskewing method.
    """
    logger.info(
        f"Running crop pipeline on {title} with crop_method={crop_method}, rotation_method={rotation_method}"
    )

    if crop_method == "inner":
        results = crop_images_inner(title, batch_size=16)
    else:
        results = crop_images_outer(title)

    for result in results:
        im = cv2.imread(result["image_path"])
        h, w = im.shape[0], im.shape[1]
        x1, y1, x2, y2 = xywh_to_xyxy_denorm(
            (result["x_center"], result["y_center"], result["width"], result["height"]),
            (w, h),
        )
        crop = im[y1:y2, x1:x2]

        if rotation_method == "hough":
            angle = get_skew_angle_hough(crop)
        else:
            angle = ROTATION_MODEL.predict_image(crop)
        result["angle"] = angle

        M = cv2.getRotationMatrix2D(
            (result["x_center"] * w, result["y_center"] * h), angle, 1.0
        )
        deskewed = cv2.warpAffine(im, M, (w, h))
        deskewed_crop = deskewed[y1:y2, x1:x2]

        st.image(deskewed_crop, width=400)
        st.write(result)

    logger.info(f"Processed {len(results)} images from {title}")
    logger.debug(f"Results: {results}")


if __name__ == "__main__":
    path = os.path.join(os.getenv("SCAN_DATA_PATH"), "2618768765/rawdata/1")
    path = "/Users/lucienovotna/Downloads/knav-1"
    run_crop_pipeline(path, crop_method="outer", rotation_method="hough")
