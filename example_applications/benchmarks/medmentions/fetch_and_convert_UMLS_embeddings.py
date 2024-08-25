# Download https://github.com/r-mal/umls-embeddings/raw/master/embeddings.csv.zip, and read the embeddings from the file line by line

import torch
from medmentions.umls import UMLSEmbeddings

import requests
import zipfile
import io

from tqdm import tqdm

# Step 1: Download the zip file
url = "https://github.com/r-mal/umls-embeddings/raw/master/embeddings.csv.zip"
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
        umls_embeddings = UMLSEmbeddings(csvfile)

    # Step 3: Serialize the embeddings to disk
    torch.save(umls_embeddings, "umls_embeddings.pt")
