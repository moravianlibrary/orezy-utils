import comet_ml
import os
from ultralytics import YOLO


comet_ml.login(
    project_name="crop_finetune_model", api_key=os.getenv("COMET_ML_API_KEY")
)

model = YOLO("base_models/yolo26s.pt")
results = model.train(
    data="datasets/dataset.yaml",
    project="crop_finetune_model",
    name="100e-32b-mosaic-no-pad",
    epochs=100,
    imgsz=640,
    batch=32,
    scale=0.5,
    flipud=0.1,
    fliplr=0.5,
    close_mosaic=25,
    degrees=4,
    shear=1.0,
    save_period=10,
    save_json=True,
    max_det=2,
    single_cls=True,
)

# model = YOLO("crop_finetune_model/train11/weights/last.pt")
# results = model.train(resume=True)

# Save the model after training
# model.save("crop_finetune_model/yolov10n_finetuned.pt")
