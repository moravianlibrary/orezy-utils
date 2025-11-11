import math
import os
import cv2
import numpy as np
import streamlit as st
import torch
from crop_model import crop_images_inner, crop_images_outer
from rotate_model import get_skew_angle_ml
from torch.utils.data import DataLoader
from rotation_model.rotate_network import AngleDegModel, load_checkpoint
from rotation_model.rotate_dataset import PageAngleDataset
from utils import xywh_to_xywh_denorm, xywh_to_xyxy_denorm
import logging
import argparse

from visualization_app.schemas import Scan

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def add_predicted_angles(scan_results: list["Scan"]) -> list["Scan"]:
    data = [
        (res.filename, bbox.xc, bbox.yc, bbox.width, bbox.height)
        for res in scan_results
        for bbox in res.predicted_pages
    ]
    images = [b[0] for b in data]
    angles = [b[1:] for b in data]
    
    test_df = PageAngleDataset(
        image_paths=images,
        image_bboxes=angles,
        is_train=False,
        image_size=640,
        angle_max=10.0,
    )
    test_loader = DataLoader(
        test_df, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    device = torch.device("mps")
    model = AngleDegModel().to(device)
    load_checkpoint("rotate_finetune_100e/best_model.pth", model, map_location=device)
    _, preds, _ = model.predict_angles(test_loader, device)

    idx = 0
    for res in scan_results:
        for bbox in res.predicted_pages:
            bbox.angle = round(float(preds[idx]), 3)
            idx += 1
    return scan_results

def resize_by_angle(w_a, h_a, angle_deg):
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    denom = c**2 - s**2
    if abs(denom) < 1e-8:
        raise ValueError("Angle too close to 45° — inversion becomes unstable.")
    w = (c * w_a - s * h_a) / denom
    h = (-s * w_a + c * h_a) / denom
    st.write(f"Original (w, h): ({w_a:.2f}, {h_a:.2f}), angle: {angle_deg:.2f}° => New (w, h): ({w:.2f}, {h:.2f})")
    return w, h

def rotate_matrix(points, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y),
            ]
            for px, py in points
        ]
    ).astype(int)


def run_crop_pipeline(folder: str, crop_method="inner", rotation_method="hough"):
    """Runs a pipeline which splits multi-page images, deskews them, and crops them.
    Two methods for deskewing are supported: "hough" uses classical image processing,
    "ml" uses a trained ResNET model.

    Args:
        title (str): Path to the folder containing images to process.
        method (str): "hough" or "ml" for deskewing method.
    """
    logger.info(
        f"Running crop pipeline on {folder} with crop_method={crop_method}, rotation_method={rotation_method}"
    )

    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files if f.lower().endswith((".jpg"))]
    files.sort()

    if crop_method == "inner":
        results = crop_images_inner(files, batch_size=16)
    else:
        results = crop_images_outer(files)
    
    results = add_predicted_angles(results)

    for result in results:
        # load image
        im = cv2.imread(result.filename)
        h, w = im.shape[0], im.shape[1]
        # rotate by its augmentation value
        M = cv2.getRotationMatrix2D((w / 2, h / 2), result.angle, 1)
        im = cv2.warpAffine(im, M, (w, h))

        for bbox in result.predicted_pages:
            bbox.width, bbox.height = resize_by_angle(
                bbox.width, bbox.height, - bbox.angle - result.angle
            )
            (
                xc,
                yc,
                ww,
                hh,
            ) = xywh_to_xywh_denorm(
                (bbox.xc, bbox.yc, bbox.width, bbox.height),
                (w, h),
            )
            rrect = ((xc, yc), (ww, hh), - bbox.angle - result.angle)
            pts = cv2.boxPoints(rrect)  # float32, order is consistent (clockwise)
            pts = np.intp(np.round(pts))  # convert to int for drawing
            cv2.polylines(im, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw dot at center
            cv2.circle(im, (int(xc), int(yc)), radius=10, color=(255, 0, 0), thickness=10)

        st.image(im, width=600)
        st.write(result)

    logger.info(f"Processed {len(results)} images from {folder}")
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

    results = run_crop_pipeline(
        args.input_path, crop_method="inner", rotation_method="hough"
    )
