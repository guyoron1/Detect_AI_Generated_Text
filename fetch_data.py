import pickle
import os
import kaggle
import requests
from tqdm import tqdm

DATASET_TYPES = ('train', 'test', 'valid')
HOME = os.getcwd()
OUTFOX_DATA_PATH = os.path.join(HOME, "external_sources/OUTFOX/data/")
KAGGLE_DATASETS = [
    {
        "url_or_identifier": "https://www.kaggle.com/datasets/conjuring92/fpe-processed-dataset",
        "path": "./external_sources/fpe"
    },
    {
        "url_or_identifier": "https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset",
        "path": "./external_sources/daigt"
    },
    {
        "url_or_identifier": "https://www.kaggle.com/datasets/nbroad/persaude-corpus-2",
        "path": "./external_sources/persuade"
    }
]
def fetch_llm_data_outfox():
    """
    Fetch only the LLM-generated essays from Outfox.
    Returns dictionary with keys test, train and validation and dataframe for each key.
    """
    OUTFOX_LLM_SOURCES = (
        'chatgpt','common',
        'dipper\\chatgpt',
        'dipper\\flan_t5_xxl',
        'dipper\\text_davinci_003',
        'flan_t5_xxl',
        'text_davinci_003'
    )
    data_list = []
    for source in OUTFOX_LLM_SOURCES:
        for type in DATASET_TYPES:
            path = os.path.join(OUTFOX_DATA_PATH, source, type, f"{type}_lms.pkl")
            try:
                with open(path, 'rb') as file:
                    data = pickle.load(file)
            except FileNotFoundError:
                continue
            data_list.append(data)

    return data_list


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


def fetch_llm_data_persuade_daigt():
    for dataset in KAGGLE_DATASETS:
        download_kaggle_dataset(dataset["url_or_identifier"], dataset["path"])

def fetch_gpt2_data():
    subdir = 'data'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for ds in [
        'webtext',
        'small-117M', 'small-117M-k40',
        'medium-345M', 'medium-345M-k40',
        'large-762M', 'large-762M-k40',
        'xl-1542M', 'xl-1542M-k40',
    ]:
        for split in ['train', 'valid', 'test']:
            filename = ds + "." + split + '.jsonl'
            r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(os.path.join(subdir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


if __name__ == '__main__':
    fetch_gpt2_data()