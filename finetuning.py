#todo: add Mistral pipeline with loading from .json
from fetch_data import download_kaggle_dataset
import pandas as pd
# assume that .json data is saved to

# .pt

# Data file will be called "train_v160125.pickle"
# Identify modelname saved after finetuning by same version.

def pull_kaggle_example_data():
    path = "./external_sources/llm-detect-ai-generated-text"
    df = pd.read_csv(path + "/train_essays.csv")
    prompts = pd.read_csv(path + "/train_prompts.csv")
    prompt_dict = prompts.set_index('prompt_id')['instructions'].to_dict()
    df['prompt_text'] = df['prompt_id'].map(prompt_dict)
    df.rename(columns={'text': 'essay_text'}, inplace=True)
    df.drop(columns=['id','prompt_id'], inplace=True)
    return df



def finetune():
    # Perform finetuning and save model to known location with indicative name
    # "./models"
    pass

def inference():
    # Load finetuned model form "./models" and perform inference
    pass


if __name__ == '__main__':
    df = pull_kaggle_example_data()
    print('a')