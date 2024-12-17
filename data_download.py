from datasets import load_dataset
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from hashlib import sha256
import os
from fine_tune_smolvlm import logger
# Load the dataset
data = load_dataset("allenai/pixmo-points", split="train")

# Ensure the directory exists
output_dir = "/data/image"
os.makedirs(output_dir, exist_ok=True)

# Create a requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Define headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
} 

# Iterate over the dataset
for i, example in enumerate(data):
    try:
        image_url = example["image_url"]


        # Generate a hash of the image URL
        image_hash = sha256(image_url.encode()).hexdigest()
        output_path = os.path.join(output_dir, f"{image_hash}.jpg")

        # Skip download if the image already exists
        if os.path.exists(output_path):
            print(f"Image {output_path} already exists. Skipping download.")
            continue

        # Download the image
        response = session.get(image_url, headers=headers, timeout=10)
        image_bytes = response.content

        # Save the image
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        if last_good_url!=image_url:
            with open("right_url.txt","a") as f:
                f.write(f"{image_url}")
            last_good_url= image_url    
    except requests.exceptions.RequestException as e:
        if last_bad_url!=image_url:
            with open("Unaccessible_url.txt","a") as file:
                file.write(f"{image_url}")
            last_bad_url=image
                
        continue
