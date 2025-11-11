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

1. Extract ScanTailor metadata
   - Run:
     ```
     scripts/extract_scantailor_data.py
     ```
   - This script extracts crop coordinates and other metadata from the `.scanTailor` files and saves them as JSON files named by `scan-id`.

2. Create dataset structure
   - Run:
     ```
     scripts/create_yolo_dataset.py
     ```
   - It consumes the JSONs produced in step 1 and arranges files into the structure expected by Ultralytics YOLO (train / val / test). See: https://docs.ultralytics.com/datasets/detect/

3. Assign classes and clean up
   - Run:
     ```
     scripts/assign_classes_and_cleanup.py
     ```
   - This script cleans mistakes in the training data and assigns objects to classes: `page`, `back title cover`, and `unified doublepage`.

4. (Optional) Rebalance splits
   - Run:
     ```
     scripts/balance_train_val_test.py
     ```
   - Use this to manually rebalance or refine the train/validation/test splits.


## Output

You can use the output structure as an input for rotate and crop finetune nets.

