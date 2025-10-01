"""Displays rotation/crop data aquired from ScanTailor"""

import json
import cv2
import numpy as np
import streamlit as st
import os


project_path = os.getenv("SCAN_DATA_PATH")
ground_truth_path = f"data/dataset/{project_path.split('/')[-1]}.json"

with open(ground_truth_path, "r") as f:
    ground_truth = json.load(f)

for image, data in list(ground_truth.items()):
    img_path = os.path.join(project_path, "rawdata", image)
    img = cv2.imread(img_path)

    for _ in range(data["orientation"] // 90):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if data["split"]:
        # draw division line
        cv2.line(img, data["split"]["p1"], data["split"]["p2"], (255, 0, 0), 4)
        # add division offset to the right page
        data["crop"][1]["x"] = data["crop"][1]["x"] + min(
            data["split"]["p1"][0], data["split"]["p2"][0]
        )

    # rotate image
    if data["crop"][0]["rotation"] != 0:
        (h, w) = img.shape[:2]
        # Compute the center of the image
        center = (w // 2, h // 2)
        angle = -data["crop"][0]["rotation"]

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute the new bounding dimensions of the image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation and resize
        img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # draw crop coordinates
    for crop in data["crop"]:
        x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    st.image(img, caption=image, use_container_width=True)
    st.write(f"Image: {image}, properties: {data}")
