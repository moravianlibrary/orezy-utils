import requests
import os

url = "https://api.ai-orezy.trinera.cloud"


token = os.getenv("BASIC_AUTH_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}
response = requests.post(url + "/create", json={}, headers=headers)

if response.status_code != 200:
    requests.delete(url + f"/{id}", headers=headers)
    raise Exception("Failed to create session", response.text)

id = response.json()["id"]


for file in os.listdir(os.getenv("SCAN_DATA_PATH")):
    response = requests.post(
        url + f"/{id}/upload-scan",
        files={"scan_data": open(os.path.join(os.getenv("SCAN_DATA_PATH"), file), "rb")},
        headers=headers,
    )
    if response.status_code != 200:
        requests.delete(url + f"/{id}", headers=headers)
        raise Exception("Upload failed", response.text)

response = requests.post(url + f"/{id}/process", json={}, headers=headers)
if response.status_code != 200:
    requests.delete(url + f"/{id}", headers=headers)
    raise Exception("Processing failed", response.text)