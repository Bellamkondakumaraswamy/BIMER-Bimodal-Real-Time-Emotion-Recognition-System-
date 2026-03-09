import requests
import os

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded {target_path}")

models = {
    "https://huggingface.co/spaces/Vishnu539/emotion-recognition/resolve/main/saved_models/ser_classifier.pt": "saved_models/ser_classifier.pt",
    "https://huggingface.co/spaces/Vishnu539/emotion-recognition/resolve/main/saved_models/ter_model.pt": "saved_models/ter_model.pt"
}

for url, path in models.items():
    if not os.path.exists(path):
        download_file(url, path)
    else:
        print(f"Already exists: {path}")
