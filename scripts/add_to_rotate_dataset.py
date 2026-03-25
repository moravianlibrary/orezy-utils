

import os
from PIL import Image, ImageOps
import numpy as np
import requests

API_KEY = os.getenv("OREZY_API_KEY")
TITLE_MAPPING = {
    "69b7fec9186765560d780a2c": "/Users/lucienovotna/Downloads/Kram - malé/200-1000",
    "69b804a0186765560d780d56": "/Users/lucienovotna/Downloads/public-archivedwl-828"
}

def get_scans_for_title(title_id: str):
    response = requests.get(
        url=f"https://api.ai-orezy.trinera.cloud/{title_id}/scans",
        headers={"X-API-Key": API_KEY},
    )
    response.raise_for_status()
    return response.json()["scans"]

def resize_and_copy_image(input_folder: str, image: str):
    image_path = os.path.join(input_folder, image)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        img.thumbnail((640, 640))
        img.save(os.path.join("rotate_dataset_temp/images", os.path.basename(image_path.replace(".tif", ".jpg"))))

def download_label(scan, image):
    label_path = os.path.join("rotate_dataset_temp/labels", image.replace(".tif", ".txt"))
    with open(label_path, "w") as f:
        for page in scan["pages"]:
            f.write(f"0 {page['xc']} {page['yc']} {page['width']} {page['height']}\n")

def split_train_val_test(output_folder: str):
    # Split the dataset into train, val, and test sets
    all_images = os.listdir("rotate_dataset_temp/images")
    np.random.shuffle(all_images)

    train_images = all_images[:int(len(all_images) * 0.8)]
    val_images = all_images[int(len(all_images) * 0.8):int(len(all_images) * 0.9)]
    test_images = all_images[int(len(all_images) * 0.9):]

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}, Test images: {len(test_images)}")

    for image in train_images:
        os.rename(os.path.join("rotate_dataset_temp/images", image),
                  os.path.join(output_folder, "images", "train", image))
        
        label = image.replace(".jpg", ".txt")
        os.rename(os.path.join("rotate_dataset_temp/labels", label),
                  os.path.join(output_folder, "labels", "train", label))
    for image in val_images:
        os.rename(os.path.join("rotate_dataset_temp/images", image),
                  os.path.join(output_folder, "images", "val", image))
        
        label = image.replace(".jpg", ".txt")
        os.rename(os.path.join("rotate_dataset_temp/labels", label),
                  os.path.join(output_folder, "labels", "val", label))
    for image in test_images:
        os.rename(os.path.join("rotate_dataset_temp/images", image),
                  os.path.join(output_folder, "images", "test", image))
        
        label = image.replace(".jpg", ".txt")
        os.rename(os.path.join("rotate_dataset_temp/labels", label),
                  os.path.join(output_folder, "labels", "test", label))

def main():
    # create temp dir
    os.makedirs("rotate_dataset_temp", exist_ok=True)
    os.makedirs("rotate_dataset_temp/images", exist_ok=True)
    os.makedirs("rotate_dataset_temp/labels", exist_ok=True)
    
    for title_id, input_folder in TITLE_MAPPING.items():
        # zip bounding boxes and image path
        scans = get_scans_for_title(title_id)
        images = sorted([f for f in os.listdir(input_folder) if f.endswith(".tif")])
        
        for scan, image in zip(scans, images):
            if len(scan["pages"]) != 1:
                print(f"Title {title_id} has scans with multiple pages, skipping.")
                continue
            
            resize_and_copy_image(input_folder, image)
            download_label(scan, image)
    
    split_train_val_test("datasets/yolo-all-batches-rotate")

    # remove temp dir
    for root, dirs, files in os.walk("rotate_dataset_temp", topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir("rotate_dataset_temp")

main()