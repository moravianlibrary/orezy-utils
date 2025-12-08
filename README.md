# Dataset creation

## Input

The dataset is created based on ScanTailor metadata files. Your folders should follow this structure:

```text
scan-id/
├─ rawdata/
│  ├─ 1/
│  │  └─ <*.tif images>
│  ├─ 2/
│  └─ ...
└─ scanTailor/
    ├─ 1.scanTailor
    ├─ 2.scanTailor
    └─ ...
```

## Steps

1. Compress the input data
   - Run:
   ```
   src/create_dataset/compress_input_images.py
   ```
   - The script compressed the input structure described above from tifs into jpgs, and saves in in a format used by other scripts. 

2. Extract ScanTailor metadata
   - Run:
     ```
     src/create_dataset/extract_scantailor_data.py
     ```
   - This script extracts crop coordinates and other metadata from the `.scanTailor` files and saves them as metadata.json files, stored in their respective folder.
   - It also cleans mistakes in the training data and assigns objects to classes: `page`, `back title cover`, and `unified doublepage`.

3. Create dataset structure
   - Run:
     ```
     src/create_dataset/create_yolo_dataset.py
     ```
   - It consumes the JSONs produced in step 1 and arranges files into the structure expected by Ultralytics YOLO (train / val / test). See: https://docs.ultralytics.com/datasets/detect/
   - Images are padded by 10 % from left/right. This ensures rotation augmentation can be applied without getting page edges out ouf frame.

4. Assign classes and clean up
   - Run:
     ```
     src/create_dataset/assign_classes_and_cleanup.py
     ```
   - This script cleans mistakes in the training data and assigns objects to classes: `page`, `back title cover`, and `unified doublepage`.

5. (Optional) Rebalance splits
   - Run:
     ```
     src/create_dataset/balance_train_val_test.py
     ```
   - Use this to manually rebalance or refine the train/validation/test splits.


## Output

You can use the output structure as an input for rotate and crop finetune nets.

# Training

Scripts for model finetuning are stored in `src.training.crop_train` and `src.training.rotate_train`. Both models utilize the same dataset.
Training reports are periodically saved to CometML, set your environment variable COMET_ML_API_KEY to enable this.

## YOCO model: You Only Crop Once

YOCO is a finetuned YOLO network based on YOLO11s (see: https://docs.ultralytics.com/models/yolo11/). It is used to predict the number of pages in a document together with its location.

## RotateNET

RotateNET is a ResNET based model used to predict angle of each page.