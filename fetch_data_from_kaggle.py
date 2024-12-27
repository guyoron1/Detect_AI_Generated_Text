import os
import kaggle

def download_kaggle_dataset(dataset_url_or_identifier: str, download_path: str):
    """Downloads a dataset from Kaggle using the Kaggle API.

    Args:
        dataset_url_or_identifier (str): The URL of the Kaggle dataset page or dataset identifier (username/dataset_name).
        download_path (str): The local directory where the dataset will be saved.
    """
    # Extract the dataset identifier if a URL is provided
    if dataset_url_or_identifier.startswith("https://"):
        dataset_identifier = "/".join(dataset_url_or_identifier.split("/")[-2:])
    else:
        dataset_identifier = dataset_url_or_identifier

    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)

    try:
        # Use the Kaggle API to download the dataset
        kaggle.api.dataset_download_files(dataset_identifier, path=download_path, unzip=True)
        print(f"Dataset downloaded and extracted to {download_path}")
    except kaggle.rest.ApiException as e:
        print(f"Failed to download dataset '{dataset_identifier}': {e}")

# List of datasets to download with corrected dataset identifiers
datasets = [
    {
        "url_or_identifier": "conjuring92/fpe-processed-dataset",
        "path": "./fpe_dataset"
    },
    {
        "url_or_identifier": "thedrcat/daigt-v2-train-dataset",
        "path": "./daigt_dataset"
    },
    {
        "url_or_identifier": "nbroad/persaude-corpus-2",
        "path": "./persaude_corpus"
    }
]

# Download each dataset
for dataset in datasets:
    download_kaggle_dataset(dataset["url_or_identifier"], dataset["path"])
