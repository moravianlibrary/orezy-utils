Automatically crop document pages from your images using the Page Trace API.

1. Install dependencies
```
pip install -r requirements.txt
```

2. Upload your folder of images
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

3. Download predictions and crop original images
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