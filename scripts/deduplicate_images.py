import os

img_path = "datasets/yolo-all-batches-no-padding/images"
labels_path = "datasets/yolo-all-batches-no-padding/labels"

for split in ["train", "val", "test"]:
    images = sorted(os.listdir(os.path.join(img_path, split)))
    for image1, image2 in zip(images, images[1:]):
        # compare if images are identical
        with open(os.path.join(img_path, split, image1), "rb") as f1:
            with open(os.path.join(img_path, split, image2), "rb") as f2:
                if f1.read() == f2.read():
                    print(f"Duplicate images found: {image1} and {image2}")

                    # combine labels from both images
                    label1_path = os.path.join(
                        labels_path, split, image1.replace(".jpg", ".txt")
                    )
                    label2_path = os.path.join(
                        labels_path, split, image2.replace(".jpg", ".txt")
                    )
                    with open(label1_path, "r") as lf1:
                        labels1 = lf1.readlines()
                    with open(label2_path, "r") as lf2:
                        labels2 = lf2.readlines()

                    # combine labels from both images
                    combined_labels = labels1 + labels2
                    if len(combined_labels) == 2:
                        print(combined_labels)
                        with open(label1_path, "w") as lf1:
                            lf1.writelines(combined_labels)
                        with open(label2_path, "w") as lf2:
                            lf2.writelines(combined_labels)
