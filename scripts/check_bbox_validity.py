import os


def fix_yolo_bbox(line: str) -> bool:
    obj_class, x_center, y_center, width, height = line.strip().split()

    # check if corners are out of bounds
    x1, y1 = float(x_center) - float(width) / 2, float(y_center) - float(height) / 2
    x2, y2 = float(x_center) + float(width) / 2, float(y_center) + float(height) / 2
    update = False
    if x1 < 0 or y1 < 0:
        print(x1, y1, x2, y2)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        update = True

    if x2 > 1 or y2 > 1:
        print(x1, y1, x2, y2)
        x2 = min(x2, 1)
        y2 = min(y2, 1)
        update = True

    if update:
        new_width = x2 - x1
        new_height = y2 - y1
        new_x_center = (x1 + x2) / 2
        new_y_center = (y1 + y2) / 2
        print(f"Old bbox: {line.strip()}")
        print(
            f"New bbox: {obj_class} {new_x_center} {new_y_center} {new_width} {new_height}"
        )
        return new_x_center, new_y_center, new_width, new_height
    return None


splits = ["train", "val", "test"]
path = "datasets/yolo-all-batches-no-padding/labels"
for split in splits:
    split_path = os.path.join(path, split)
    for filename in os.listdir(split_path):
        if filename.endswith(".txt"):
            with open(os.path.join(split_path, filename), "r") as f:
                lines = f.readlines()
                correct_lines = []
                replace = False
                for line in lines:
                    # Check if the line is a valid YOLO bbox format
                    fix = fix_yolo_bbox(line)
                    if fix is not None:
                        line = line.strip().split()
                        new_x_center, new_y_center, new_width, new_height = fix
                        line[1] = str(new_x_center)
                        line[2] = str(new_y_center)
                        line[3] = str(new_width)
                        line[4] = str(new_height)
                        line = " ".join(line) + "\n"
                        replace = True
                    correct_lines.append(line)

            if replace:
                # with open(os.path.join(split_path, filename), "w") as f:
                #    f.writelines(correct_lines)
                print(f"Updated file: {os.path.join(split_path, filename)}")
