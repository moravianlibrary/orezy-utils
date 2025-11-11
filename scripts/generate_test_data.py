import os
import json
from PIL import Image

from visualization_app.main import run_crop_pipeline


def resize_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".gif")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + ".jpg"
            )
            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    target_size = (width // 4, height // 4)
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(output_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    file = "2610267219"
    input_dir = f"/Users/lucienovotna/Documents/AIorezy-data/{file}/rawdata/2"
    output_dir = f"{file}"
    os.makedirs(output_dir, exist_ok=True)

    # generate small version of images
    resize_images(input_dir, output_dir)

    # add json results
    results = run_crop_pipeline(
        output_dir, crop_method="inner", rotation_method="hough"
    )

    with open(os.path.join(output_dir, "transformations.json"), "w") as f:
        json.dump(results, f, indent=4)
