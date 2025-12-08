from collections import defaultdict
from enum import Enum
import os
import json


class ClassificationClass(Enum):
    page = 0
    back_cover = 1
    double_page = 2


def is_valid_page(data: dict) -> bool:
    if data["crop"][1]["width"] is None:
        return False
    if data["crop"][1]["width"] == 0:
        return False
    if data["crop"][0]["width"] / data["crop"][1]["width"] > 2.0:
        return False
    return True


def is_back_cover(data: dict) -> bool:
    # Height must be similar
    if abs(data["crop"][0]["height"] - data["crop"][1]["height"]) > 100:
        return False
    # Widths must be different by at least 10%
    if data["crop"][0]["width"] / data["crop"][1]["width"] < 1.1:
        return False
    return True


def walk_through_dataset():
    results = {
        "invalid": defaultdict(list),
        "back_covers": defaultdict(list),
        "double_pages": defaultdict(list),
    }
    books = [
        d
        for d in os.listdir("/Users/lucienovotna/Documents/ai-orezy-compressed")
        if os.path.isdir(
            os.path.join("/Users/lucienovotna/Documents/ai-orezy-compressed", d)
        )
    ]

    for book in books:
        with open(
            os.path.join(
                "/Users/lucienovotna/Documents/ai-orezy-compressed",
                book,
                "scanTailor",
                "metadata.json",
            ),
            "r",
        ) as f:
            data = f.read()
            data = json.loads(data)

        all_scans = len(list(data.values()))
        single_pages = len([d for d in list(data.values()) if d["split"] is None])
        single_ratio = single_pages / all_scans

        for key, value in data.items():
            key = key.replace(".tif", ".jpg").replace("/", "-")
            if value["split"] is None:
                if (
                    single_ratio < 0.9
                    and value["crop"][0]["width"] > value["crop"][0]["height"]
                ):
                    results["double_pages"][book] += [key]
                    print(f"Double page detected in {book} at {key}")
                    continue
            else:
                if not is_valid_page(value):
                    results["invalid"][book] += [key]
                    print(f"Invalid page detected in {book} at {key}")
                    continue

                if is_back_cover(value):
                    results["back_covers"][book] += [key]
                    print(f"Back cover detected in {book} at {key}")
                    continue

    print(results)
    return results


def update_dataset(results):
    base_path = "/Users/lucienovotna/Documents/orezy-utils/datasets/yolo-all-batches"
    for book, images in results["invalid"].items():
        directory = ""
        for split in ["train", "val", "test"]:
            path = os.path.join(base_path, "labels", split)
            files = os.listdir(path)
            for file in files:
                if file.startswith(book + "-"):
                    directory = split
                    break
        for image in images:
            label_file = os.path.join(
                base_path, "labels", directory, image.replace(".jpg", ".txt")
            )
            if os.path.exists(label_file):
                os.remove(label_file)
                print(f"Removed invalid label file: {label_file}")


results = walk_through_dataset()
update_dataset(results)
