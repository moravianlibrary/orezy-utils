import os
from PIL import Image
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 203965792
input = "/Volumes/Trineradisk/ai-orezy-batch2"
output = "/Volumes/Trineradisk/ai-orezy-batch2-compressed"

titles = os.listdir(input)
if not os.path.exists(os.path.join(output)):
    os.makedirs(os.path.join(output))

for title in titles:
    if not os.path.isdir(os.path.join(input, title)):
        continue
    # if folder already exists, book was processed and can be skipped
    if os.path.exists(os.path.join(output, title)):
        continue

    # create output folder
    os.makedirs(os.path.join(output, title))
    os.makedirs(os.path.join(output, title, "scanTailor"))
    os.makedirs(os.path.join(output, title, "images"))

    # copy .scanTailor files into output folder
    scanTailor_files = os.listdir(os.path.join(input, title, "scanTailor"))
    for file in scanTailor_files:
        if file.endswith(".scanTailor"):
            os.system(
                f"cp {os.path.join(input, title, 'scanTailor', file)} {os.path.join(output, title, 'scanTailor', file)}"
            )

    # copy image files
    image_folders = os.listdir(os.path.join(input, title, "rawdata"))
    for prefix in image_folders:
        # if not folder, skip
        if not os.path.isdir(os.path.join(input, title, "rawdata", prefix)):
            continue
        for file in os.listdir(os.path.join(input, title, "rawdata", prefix)):
            if file.endswith(".tif") and not file.startswith("."):
                input_path = os.path.join(input, title, "rawdata", prefix, file)
                output_path = os.path.join(
                    output, title, "images", f"{prefix}-{file.replace('.tif', '.jpg')}"
                )
                # compress and save
                with Image.open(input_path) as img:
                    rgb_img = img.convert("RGB")
                    rgb_img.save(output_path, "JPEG", quality=50)
                print(f"Compressed and saved {output_path}")
