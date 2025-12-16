"""Takes data from extract_scantailor_data script and converts them to format suitable for YOLO training."""

import os
import json

import cv2
import numpy as np
import argparse

from tqdm import tqdm


def convert_bbox_to_yolo_format(
    img_properties, img_shape, padded=False, padded_percent=0.1
):
    """Converts bounding box to YOLO format.

    Args:
        bbox (list): Bounding box in [x1, y1, x2, y2] format.
        img_shape (tuple): Shape of the image (height, width).
    Returns:
        list: Bounding box in YOLO format [x_center, y_center, width, height].
    """
    if any([v is None for v in img_properties.values()]):
        print(
            f"Warning: Incomplete image properties for image {img_properties.get('name')}"
        )
        return [0, 0, 0, 0]
    if padded:
        # adjust for padding
        img_properties["x"] += int(padded_percent * img_shape[1] / 1.2)

    h, w = img_shape
    x_center = (img_properties["x"] + img_properties["width"] / 2) / w
    y_center = (img_properties["y"] + img_properties["height"] / 2) / h
    width = img_properties["width"] / w
    height = img_properties["height"] / h

    return [x_center, y_center, width, height]


def add_padding(img, img_width, left=True, right=True, percent=0.1):
    """Adds padding to the left and/or right side of the image.

    Args:
        img (np.ndarray): Input image.
        img_width (int): Width of the image.
        left (bool): Whether to add padding to the left side.
        right (bool): Whether to add padding to the right side.
        percent (float): Percentage of the image width to use as padding.
    Returns:
        np.ndarray: Image with added padding.
    """
    padding_size = int(img_width * percent)
    left_padding = padding_size if left else 0
    right_padding = padding_size if right else 0
    padded_img = cv2.copyMakeBorder(
        img, 0, 0, left_padding, right_padding, cv2.BORDER_REFLECT_101
    )
    return padded_img


def process_book_double_page(input, output, book_name, split="train"):
    """Processes a ScanTailor book and saves images as two-page splits in YOLO format.

    Args:
        input (str): Path to the input directory containing ScanTailor books.
        output (str): Path to the output directory for YOLO formatted images.
        metadata_folder (str): Path to the folder containing metadata JSON files.
    """

    with open(os.path.join(input, book_name, "scanTailor", "metadata.json"), "r") as f:
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

        file = os.path.join(input, book_name, "images", image_name)
        img = cv2.imread(file)

        # fix orientation
        if img_properties["orientation"] != 0:
            for _ in range(img_properties["orientation"] // 90):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        page_name = f"double_{book_name}-{image_name.split('.')[0]}"
        # save bbox info in YOLO format
        with open(os.path.join(output, "labels", split, f"{page_name}.txt"), "w") as f:
            # adjust offset from page split
            img_properties["crop"][1]["x"] = img_properties["crop"][1]["x"] + min(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            # create 2 bounding boxes
            for box in img_properties["crop"]:
                normalized_bbox = convert_bbox_to_yolo_format(box, img.shape[:2])
                img_class = box["class"]
                f.write(f"{img_class} {' '.join(map(str, normalized_bbox))}\n")

        # resize and save the image
        scale = 640 / min(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output, "images", split, f"{page_name}.jpg"), img)


def process_book_single_page(input, output, book_name, split="train"):
    """Processes a ScanTailor book and saves images in YOLO format.

    Args:
        input (str): Path to the input directory containing ScanTailor books.
        output (str): Path to the output directory for YOLO formatted images.
        metadata_folder (str): Path to the folder containing metadata JSON files.
    """

    with open(os.path.join(input, book_name, "scanTailor", "metadata.json"), "r") as f:
        metadata = json.load(f)

    # resize each image so shorter side is 640px, rotate by degrees in metadata
    for image_name in metadata:
        file = os.path.join(input, book_name, "images", image_name)
        img = cv2.imread(file)
        img_properties = metadata[image_name]

        # fix orientation
        if img_properties["orientation"] != 0:
            for _ in range(img_properties["orientation"] // 90):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        pages = []
        # process left and right separately if there is a split
        if img_properties["split"]:
            left_divider = max(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            right_divider = min(
                img_properties["split"]["p1"][0], img_properties["split"]["p2"][0]
            )
            left_page = img[:, :left_divider]
            right_page = img[:, right_divider:]
            # add 10% padding to LR side
            # l_size = left_divider
            # left_page = img[:, :int(l_size * 1.1)]
            # left_page = add_padding(
            #    left_page, l_size, left=True, right=False
            # )
            # add 10% padding to LR side
            # r_size = img.shape[1] - right_divider
            # right_page = img[:, int(r_size * 0.9):]
            # right_page = add_padding(
            #    right_page, r_size, left=False, right=True
            # )

            pages += [left_page, right_page]
        else:
            # add 10% padding to LR side
            # img = add_padding(img, img.shape[1], left=True, right=True)
            pages.append(img)

        # save each page as a separate file
        for i, page in enumerate(pages):
            page_name = f"{book_name}-{image_name.split('.')[0]}_p{i}"
            # save bbox info in YOLO format
            with open(
                os.path.join(output, "labels", split, f"{page_name}.txt"), "w"
            ) as f:
                normalized_bbox = convert_bbox_to_yolo_format(
                    img_properties["crop"][i],
                    page.shape[:2],  # padded=True
                )
                img_class = img_properties["crop"][i]["class"]
                f.write(f"{img_class} {' '.join(map(str, normalized_bbox))}\n")

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
        default="ai-orezy-compressed",
        help="Path to the input directory containing ScanTailor books.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/yolo-all-batches-no-padding",
        help="Path to the output directory for YOLO formatted images.",
    )

    args = parser.parse_args()

    input = args.input
    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)
        os.makedirs(os.path.join(output, "images"))
        os.makedirs(os.path.join(output, "labels"))

    all_books = os.listdir(input)
    all_books = [book for book in all_books if os.path.isdir(os.path.join(input, book))]

    # split into train validation and test sets
    np.random.shuffle(all_books)
    train_size = int(0.8 * len(all_books))
    val_size = int(0.1 * len(all_books))
    test_size = len(all_books) - train_size - val_size

    train = all_books[:train_size]
    val = all_books[train_size : train_size + val_size]
    test = all_books[train_size + val_size :]

    # ensure book 2610055011 is in TEST set
    if "2610055011" not in test:
        test.append("2610055011")
    if "2610055011" in val:
        val.remove("2610055011")
    if "2610055011" in train:
        train.remove("2610055011")

    # Iterate through train / val / test splits and process each book
    for split_name, books in [("train", train), ("val", val), ("test", test)]:
        # Create output directories for the split
        os.makedirs(os.path.join(output, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(output, "labels", split_name), exist_ok=True)

        for book in tqdm(books, desc=f"Processing {split_name} set"):
            # Process each book
            print(f"Processing book: {book}")
            process_book_single_page(input, output, book, split=split_name)
            process_book_double_page(input, output, book, split=split_name)
