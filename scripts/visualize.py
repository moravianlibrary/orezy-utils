import cv2
import os
import streamlit as st
from tqdm import tqdm


def show_pages(show_id, split):
    images = sorted(
        [
            f
            for f in os.listdir(f"datasets/yolo-all-batches/images/{split}")
            if f.endswith(".jpg")
        ]
    )
    labels = sorted(
        [
            f
            for f in os.listdir(f"datasets/yolo-all-batches/labels/{split}")
            if f.endswith(".txt")
        ]
    )

    for label in tqdm(labels):
        with open(
            os.path.join(f"datasets/yolo-all-batches/labels/{split}", label), "r"
        ) as f:
            lines = f.readlines()
            class_id = [int(line.strip().split()[0]) for line in lines][0]
            if int(class_id) == show_id:
                x_center, y_center, box_w, box_h = [
                    float(val) for val in lines[0].strip().split()[1:]
                ]
                img = cv2.imread(
                    os.path.join(
                        f"datasets/yolo-all-batches/images/{split}",
                        images[labels.index(label)],
                    )
                )
                w, h = img.shape[1], img.shape[0]
                x_center *= w
                y_center *= h
                box_w *= w
                box_h *= h
                cv2.rectangle(
                    img,
                    (int(x_center - box_w / 2), int(y_center - box_h / 2)),
                    (int(x_center + box_w / 2), int(y_center + box_h / 2)),
                    (255, 0, 0),
                    2,
                )

                st.image(img)
                st.write(f"Image: {label}, Split: {split}, Class ID: {class_id}")


with st.spinner("Processing images..."):
    st.title("Cover Visualizer")
    for split in ["train", "val", "test"]:
        show_pages(2, split)
