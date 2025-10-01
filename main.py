import os
import cv2
import streamlit as st
from crop_model import crop_images_inner, crop_images_outer
from rotate_finetune import AngleDegModel, load_checkpoint
from rotate_model import get_skew_angle_hough
from utils import xywh_to_xyxy_denorm
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

        angle = get_skew_angle_hough(crop)
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

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run crop pipeline on scanned images.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.getenv("SCAN_DATA_PATH"),
        help="Path to the folder containing images to process (default: SCAN_DATA_PATH env variable)",
    )
    args = parser.parse_args()

    logger.info(f"Using SCAN_DATA_PATH from environment: {args.input_path}")

    results = run_crop_pipeline(args.input_path, crop_method="inner", rotation_method="hough")
