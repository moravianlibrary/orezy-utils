# Cropilot API tools

Cropilot API tools is a set of scripts which allow communication with https://orezy.test.trinera.cloud .

## Uploader

Uploader.py script allows upload and download of new books. The workflow is as follows: First, a folder of uncropped scans is downscaled and uploaded for processing to Cropilot queue. Then, a job outputs a set of page predictions. When completed, the book becomes available on the web for review. User can change any prediction bounding box in the editor. Predictions can be downloaded and applied to the original folder of images, resulting in high quality cropped scans.

### How to run

1. Install dependencies
```
pip install -r requirements-uploader.txt
```

2. Upload your folder of images. The script outputs a link where your predictions will become available. You can 
edit them to your needs before calling the second script.
```
python3 uploader.py upload --api-key <GROUP API KEY> --input-folder sample_input

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     API key for authentication within given group, obtain from group settings in the web app
  --api-url API_URL     [Optional] Base URL of the Page Trace API, defaults to https://api.ai-orezy.trinera.cloud
  --input-folder INPUT_FOLDER
                        Input folder path (containing images to process)
  --model MODEL         [Optional] Model name to use for prediction, defaults to group default
  --name NAME           [Optional] Custom title name, defaults to input folder name
```

3. Download predictions and crop your original images from the local folder.
```
python3 uploader.py download --api-key <GROUP API KEY> --title <TITLE ID> --input-folder sample_input

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     API key for authentication within given group, obtain from group settings in the web app
  --api-url API_URL     [Optional] Base URL of the Page Trace API, defaults to https://api.ai-orezy.trinera.cloud
  --input-folder INPUT_FOLDER
                        Input folder path (containing images to process)
  --title TITLE         Title ID
  --output-folder OUTPUT_FOLDER
                        [Optional] Output folder path (to save cropped images), defaults to "output"
```

## Trainer



### Requirements

*Create a batch of labelled data*: Before running the trainer, label (create a correct set of crop boxes) one or more titles in the Cropilot editor. We recommend the batch has 100 boxes in total. If you are training from multiple titles, ensure that they belong to the same group.

*Have enough resources*: The script needs approx. 10 GB of GPU memory.

*Track progress (Optional)*: The metadata and metrics during training are uploaded to https://www.comet.com/. To enable experiment tracking, set an environment variable `COMET_ML_API_KEY`

### How to run

1. Install dependencies
```
pip install -r requirements-trainer.txt
```

2. Start trainer script. Provide the group API key, a name for the new model, and the list of title IDs that should be used for training. After the script finishes, you will be able to access the new model in Cropilot UI (or by using it in the `--model` parameter in `uploader`).
```
python3 trainer.py --api_key <GROUP API KEY> --model_name my-new-model --title_ids title1 title2 title3

options:
  -h, --help            show this help message and exit
  --api_url API_URL     [Optional] Base URL of the API, defaults to https://api.ai-orezy.trinera.cloud/
  --base_model BASE_MODEL
                        [Optional] Path to the base YOLO model to fine-tune, defaults to base_models/default.pt
  --api_key API_KEY     Group API key for authentication
  --model_name MODEL_NAME
                        New name of the fine-tuned model
  --title_ids TITLE_IDS [TITLE_IDS ...]
                        List of title IDs to train on
```