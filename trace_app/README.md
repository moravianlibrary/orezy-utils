# Page Tracer

Page Tracer is a set of scripts which allow upload and download of new books to https://orezy.test.trinera.cloud .
The workflow is as follows: First, a folder of uncropped scans is downscaled and uploaded for processing to
Hatchet queue. Then, a Hatchet job outputs a set of page predictions. When completed, the book becomes
available on the web for review. User can change any prediction bounding box in the editor. Lastly, the predictions
can be downloaded and applied to the original folder of images, resulting in high quality cropped scans.

## How to use

1. Install dependencies
```
pip install -r requirements.txt
```

2. Upload your folder of images. The script outputs a link where your predictions will become available. You can 
edit them to your needs before calling the second script.
```
python3 page_tracer.py upload --api-key <GROUP API KEY> --input-folder sample_input

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     API key for authentication within given group, obtain from group settings in the web app
  --api-url API_URL     Base URL of the Page Trace API, defaults to https://api.ai-orezy.trinera.cloud
  --input-folder INPUT_FOLDER
                        Input folder path (containing images to process)
  --model MODEL         Model name to use for prediction, currently available: [inner, outer]
  --name NAME           Custom title name, defaults to input folder name
```

3. Download predictions and crop your original images from the local folder.
```
python3 page_tracer.py download --api-key <GROUP API KEY> --title <TITLE ID> --input-folder sample_input

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     API key for authentication within given group, obtain from group settings in the web app
  --api-url API_URL     Base URL of the Page Trace API, defaults to https://api.ai-orezy.trinera.cloud
  --input-folder INPUT_FOLDER
                        Input folder path (containing images to process)
  --output-folder OUTPUT_FOLDER
                        Output folder path (to save cropped images), defaults to "output"
  --title TITLE         Title ID
```