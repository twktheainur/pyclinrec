# Download https://github.com/r-mal/umls-embeddings/raw/master/embeddings.csv.zip, and read the embeddings from the file line by line

import os
import torch

from medmentions.umls import (
    UMLSEmbeddings,
    # UMLSEmbeddingsVectorStore,
    load_umls_embeddings_to_redis,
)

import requests
import zipfile
import io

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Fetch and convert UMLS embeddings.")
parser.add_argument(
    "--url",
    type=str,
    default="https://github.com/r-mal/umls-embeddings/raw/master/embeddings.csv.zip",
    help="URL to the zip file containing the embeddings.",
)
parser.add_argument(
    "--format",
    type=str,
    choices=["pt", "redis"],
    required=True,
    help="Output format: 'pt' for PyTorch or 'deeplake' for DeepLake.",
)

args = parser.parse_args()

# Step 1: Download the zip file after checking the file is not already downloaded

zip_file_path = "data/umls_embeddings.csv.zip"
if not os.path.exists(zip_file_path):
    url = args.url
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    response.raise_for_status()  # Ensure we notice bad responses
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading corpus embeddings.csv.zip",
    ) as progress_bar:
        with open("data/umls_embeddings.csv.zip", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


# Step 2: Extract the file from the zip archive and parse its contents
with zipfile.ZipFile("data/umls_embeddings.csv.zip") as the_zip:
    from tqdm import tqdm

    with the_zip.open("embeddings.csv") as csvfile:
        if args.format == "pt":
            # Process and save as PyTorch .pt format
            embeddings = UMLSEmbeddings(csvfile)
            torch.save(embeddings, "data/umls_embeddings.pt")
        elif args.format == "redis":
            import redis

            # Process and save as DeepLake format
            client = redis.Redis(host="localhost", port=6379, decode_responses=True)

            load_umls_embeddings_to_redis(csvfile, client)
