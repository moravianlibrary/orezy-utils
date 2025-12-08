"""Takes data from extract_scantailor_data script and converts them to format suitable for YOLO training."""

import os
import json
import streamlit as st
import tqdm

import cv2
import numpy as np
import argparse


def list_available_two_pages(min_rotation=0.2, max_rotation=0.25):
    """Lists images with two-page splits and significant rotation in the dataset.

    Args:
        min_rotation (float): Minimum rotation angle to consider.
        max_rotation (float): Maximum rotation angle to consider.
    """
    files = sorted(os.listdir("datasets/scantailor_data"))
    input = os.getenv("SCAN_DATA_PATH")

    allowed_pages = 0
    max_rotation_in_dataset = 0
    for json_file in tqdm.tqdm(files):
        if not json_file.endswith(".json"):
            continue
        path = os.path.join("datasets/scantailor_data", json_file)
        with open(path, "r") as f:
            data = json.load(f)
            project_name = json_file.replace(".json", "")

            for k, v in data.items():
                max_rotation_in_dataset = max(
                    max_rotation_in_dataset,
                    max(abs(crop["rotation"]) for crop in v["crop"]),
                )
                if v["split"] is None:
                    continue
                if (
                    abs(v["crop"][0]["rotation"]) < min_rotation
                    or abs(v["crop"][1]["rotation"]) < min_rotation
                ):
                    continue
                if (
                    abs(v["crop"][0]["rotation"]) > max_rotation
                    or abs(v["crop"][1]["rotation"]) > max_rotation
                ):
                    continue

                allowed_pages += 1

                file = os.path.join(input, project_name, "rawdata", k)
                img = cv2.imread(file)

                v["crop"][1]["x"] = v["crop"][1]["x"] + min(
                    v["split"]["p1"][0], v["split"]["p2"][0]
                )
                h, w = img.shape[:2]
                for crop in v["crop"]:
                    x1 = crop["x"]
                    y1 = crop["y"]
                    x2 = crop["x"] + crop["width"]
                    y2 = crop["y"] + crop["height"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

                st.image(
                    img,
                    caption=f"Image with bboxes, rotation={max(abs(crop['rotation']) for crop in v['crop'])}",
                    width=400,
                )

    print(f"Max rotation: {max_rotation_in_dataset}")
    print(f"Allowed pages: {allowed_pages}")


def convert_bbox_to_yolo_format(img_properties, img_shape):
    """Converts bounding box to YOLO format.

    Args:
        bbox (list): Bounding box in [x1, y1, x2, y2] format.
        img_shape (tuple): Shape of the image (height, width).
    Returns:
        list: Bounding box in YOLO format [x_center, y_center, width, height].
    """

    h, w = img_shape
    x_center = (img_properties["x"] + img_properties["width"] / 2) / w
    y_center = (img_properties["y"] + img_properties["height"] / 2) / h
    width = img_properties["width"] / w
    height = img_properties["height"] / h
    return [x_center, y_center, width, height]


def process_project_double_page(
    input, output, project_name, metadata_folder, split="train"
):
    """Processes a ScanTailor project and saves images as two-page splits in YOLO format.

    Args:
        input (str): Path to the input directory containing ScanTailor projects.
        output (str): Path to the output directory for YOLO formatted images.
        metadata_folder (str): Path to the folder containing metadata JSON files.
    """

    with open(os.path.join(metadata_folder, f"{project_name}.json"), "r") as f:
        metadata = json.load(f)

    for image_name in metadata:
        img_properties = metadata[image_name]
        if img_properties["split"] is None:
            continue
        if (
            abs(img_properties["crop"][0]["rotation"]) > 0.25
            or abs(img_properties["crop"][1]["rotation"]) > 0.25
        ):
            continue

        file = os.path.join(input, project_name, "rawdata", image_name)
        img = cv2.imread(file)

        # fix orientation
        if img_properties["orientation"] != 0:
            for _ in range(img_properties["orientation"] // 90):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        page_name = (
            f"double_{project_name}-{image_name.replace('/', '_').split('.')[0]}"
        )
        # save bbox info in YOLO format
        with open(os.path.join(output, "labels", split, f"{page_name}.txt"), "w") as f:
            # adjust offset from page split
            img_properties["crop"][1]["x"] = img_properties["crop"][1]["x"] + min(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            # create 2 bounding boxes
            for box in img_properties["crop"]:
                normalized_bbox = convert_bbox_to_yolo_format(box, img.shape[:2])
                f.write(f"0 {' '.join(map(str, normalized_bbox))}\n")

        # resize and save the image
        scale = 640 / min(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output, "images", split, f"{page_name}.jpg"), img)


def process_project_single_page(
    input, output, project_name, metadata_folder, split="train"
):
    """Processes a ScanTailor project and saves images in YOLO format.

    Args:
        input (str): Path to the input directory containing ScanTailor projects.
        output (str): Path to the output directory for YOLO formatted images.
        metadata_folder (str): Path to the folder containing metadata JSON files.
    """

    with open(os.path.join(metadata_folder, f"{project_name}.json"), "r") as f:
        metadata = json.load(f)

    # resize each image so shorter side is 640px, rotate by degrees in metadata
    for image_name in metadata:
        file = os.path.join(input, project_name, "rawdata", image_name)
        img = cv2.imread(file)
        img_properties = metadata[image_name]

        # fix orientation
        if img_properties["orientation"] != 0:
            for _ in range(img_properties["orientation"] // 90):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # do a page split if needed
        pages = []
        if img_properties["split"]:
            left_divider = max(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            right_divider = min(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            left_page = img[:, :left_divider]
            right_page = img[:, right_divider:]

            pages += [left_page, right_page]
        else:
            pages.append(img)

        # save each page as a separate file
        for i, page in enumerate(pages):
            page_name = (
                f"{project_name}-{image_name.replace('/', '_').split('.')[0]}_p{i}"
            )
            # save bbox info in YOLO format
            with open(
                os.path.join(output, "labels", split, f"{page_name}.txt"), "w"
            ) as f:
                normalized_bbox = convert_bbox_to_yolo_format(
                    img_properties["crop"][i], page.shape[:2]
                )
                f.write(f"0 {' '.join(map(str, normalized_bbox))}\n")

            # rotate image
            if img_properties["crop"][i]["rotation"] != 0:
                (h, w) = page.shape[:2]
                center = (w // 2, h // 2)
                angle = -img_properties["crop"][i]["rotation"]

                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))

                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]

                page = cv2.warpAffine(page, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

            # resize and save the image
            scale = 640 / min(page.shape[:2])
            new_size = (int(page.shape[1] * scale), int(page.shape[0] * scale))
            page = cv2.resize(page, new_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output, "images", split, f"{page_name}.jpg"), page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ScanTailor data to YOLO format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("SCAN_DATA_PATH"),
        help="Path to the input directory containing ScanTailor projects.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/yolo2",
        help="Path to the output directory for YOLO formatted images.",
    )
    parser.add_argument(
        "--metadata_folder",
        type=str,
        default="datasets/scantailor_data",
        help="Path to the folder containing metadata JSON files.",
    )

    args = parser.parse_args()

    input = args.input
    output = args.output
    metadata_folder = args.metadata_folder

    if not os.path.exists(output):
        os.makedirs(output)
        os.makedirs(os.path.join(output, "images"))
        os.makedirs(os.path.join(output, "labels"))

    projects = os.listdir(input)

    # split into train and validation sets
    projects = [p for p in projects if os.path.isdir(os.path.join(input, p))]
    np.random.shuffle(projects)
    train_size = int(0.9 * len(projects))
    train_projects = projects[:train_size]
    val_projects = projects[train_size:]

    os.makedirs(os.path.join(output, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output, "labels", "train"), exist_ok=True)
    for project in train_projects:
        # Process each project
        print(f"Processing TRAIN project: {project}")
        process_project_single_page(
            input, output, project, metadata_folder, split="train"
        )
        process_project_double_page(
            input, output, project, metadata_folder, split="train"
        )

    os.makedirs(os.path.join(output, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output, "labels", "val"), exist_ok=True)
    for project in val_projects:
        # Process each project
        print(f"Processing VALIDATION project: {project}")
        process_project_single_page(
            input, output, project, metadata_folder, split="val"
        )
