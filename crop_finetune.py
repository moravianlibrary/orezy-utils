import comet_ml
import os
from ultralytics import YOLO


comet_ml.login(project_name="crop_finetune_model", api_key=os.getenv("COMET_ML_API_KEY"))

model = YOLO("crop_finetune_model/train2/weights/last.pt")
results = model.train(
    data="models/dataset.yaml",
    project="crop_finetune_model",
    epochs=30,
    imgsz=640,
    batch=16,
    device="mps",
    single_cls=True,
    close_mosaic=20,
    flipud=0.2,
)

# model = YOLO("crop_finetune_model/train2/weights/last.pt")
# results = model.train(resume=True)

# Save the model after training
model.save("crop_finetune_model/yolov10n_finetuned.pt")
