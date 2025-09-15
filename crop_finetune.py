import comet_ml
import os
from ultralytics import YOLO


comet_ml.login(project_name="crop_finetune", api_key=os.getenv("COMET_ML_API_KEY"))

model = YOLO("models/yolov10n.pt")
results = model.train(
    data="models/dataset.yaml",
    project="crop_finetune",
    epochs=25,
    imgsz=640,
    batch=16,
    device="mps",
    single_cls=True,
    close_mosaic=20,
    flipud=0.2,
    patience=5,
)

# Save the model after training
model.save("models/yolov10n_finetuned.pt")
