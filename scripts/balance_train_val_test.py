# move all images starting with 2610078027, 2610267219, 2619387078, 2619611960, 2619711148 from val to test

import os

file_prefixes = [
    "2610063016",
    "double_2610063016",
]

from_dir = "datasets/yolo-split_per_book/images/train"
to_dir = "datasets/yolo-split_per_book/images/val"
os.makedirs(to_dir, exist_ok=True)
moved_files = 0
for filename in os.listdir(from_dir):
    if any(filename.startswith(prefix) for prefix in file_prefixes):
        src_path = os.path.join(from_dir, filename)
        dst_path = os.path.join(to_dir, filename)
        os.rename(src_path, dst_path)
        moved_files += 1

print(f"Moved {moved_files} files from {from_dir} to {to_dir}.")

from_dir = "datasets/yolo-split_per_book/labels/train"
to_dir = "datasets/yolo-split_per_book/labels/val"
os.makedirs(to_dir, exist_ok=True)
moved_files = 0
for filename in os.listdir(from_dir):
    if any(filename.startswith(prefix) for prefix in file_prefixes):
        src_path = os.path.join(from_dir, filename)
        dst_path = os.path.join(to_dir, filename)
        os.rename(src_path, dst_path)
        moved_files += 1

print(f"Moved {moved_files} files from {from_dir} to {to_dir}.")


prefix = "datasets/yolo-split_per_book/images/"

for folder in ["train", "val", "test"]:
    path = os.path.join(prefix, folder)
    num_files = len(
        [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )
    print(f"{folder}: {num_files} files")
    percent = (
        num_files
        / sum(
            len(
                [
                    f
                    for f in os.listdir(os.path.join(prefix, d))
                    if os.path.isfile(os.path.join(prefix, d, f))
                ]
            )
            for d in ["train", "val", "test"]
        )
    ) * 100
    print(f"{folder}: {percent:.2f}%")
