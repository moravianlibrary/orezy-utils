import comet_ml
import os
from ultralytics import YOLO


comet_ml.login(
    project_name="crop_finetune_model", api_key=os.getenv("COMET_ML_API_KEY")
)
"""
model = YOLO("models/yolov10n.pt")
results = model.train(
    data="datasets/dataset.yaml",
    project="crop_finetune_model",
    epochs=150,
    imgsz=960,
    batch=16,
    device="mps",
    single_cls=True,
    flipud=0.2,
    mosaic=0,
    degrees=6,
    shear=1.0,
    save_period=1,
    save_json=True,
)"""

model = YOLO("crop_finetune_model/train11/weights/last.pt")
results = model.train(resume=True)

# Save the model after training
model.save("crop_finetune_model/yolov10n_finetuned.pt")
