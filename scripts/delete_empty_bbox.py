import os

path = "datasets/yolo-split_per_book/labels"

for dir in ["train", "val", "test"]:
    for file in os.listdir(os.path.join(path, dir)):
        file_path =  os.path.join(path, dir, file)

        with open(file_path, "r") as f:
            line = f.readline().strip()
            coords = list(map(float, line.split(" ")[1:]))

            if not len(coords) == 4:
                raise ValueError(f"Invalid bbox vector in file {file_path}: {line}")
            
            if tuple(coords) == (0.0, 0.0, 0.0, 0.0):
                print(f"Deleting empty bbox file: {file_path}")
                img_file = file_path.replace("labels", "images").replace(".txt", ".jpg")
                os.remove(img_file)
                os.remove(file_path)