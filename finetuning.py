import argparse
import torch
from fetch_data import download_kaggle_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from loguru import logger
import json
from typing import List
import format
from torch.utils.data import DataLoader

from format import DATASET_NAME_TO_PATH, dataset_version
# .pt

# Data file will be called "train_v160125.pickle"
# Identify modelname saved after finetuning by same version.


def merge_data_for_finetuning(sources:List[str], sample_size:int):
    """
    Sources can be what was implemented in format (assert takes care of that).
    """
    list_of_dfs_to_merge = []
    for source in sources:
        assert source in DATASET_NAME_TO_PATH.keys()
    for source in sources:
        path = DATASET_NAME_TO_PATH[source]
        if source == 'fpe':
            df = format.format_fpe_to_df(path)
        elif source == 'daigt':
            df = format.format_daigt_to_df(path)
        elif source == 'persuade':
            df = format.format_persuade_to_df(path)
        else:
            raise Exception(f"Unrecognized data source/")
        list_of_dfs_to_merge.append(df)

    merged_data = pd.concat(list_of_dfs_to_merge,ignore_index=True) # todo: add sampling mechanism (percentages) in merge.
    merged_filtered_sampled_data = format.filter_and_sample_merged_df(merged_data, sample_size)
    return merged_filtered_sampled_data



def write_classifier_format(dataset: pd.DataFrame, output_path: str, write_json=False):
    """
    Writes the dataset into the classifier format as a JSON file.
    Returns dataframe as well.
    Args:
        dataset (pd.DataFrame): The input dataset with columns
                                'prompt_text', 'essay_text', 'generated', and 'source'.
        output_path (str): Path to the output data dir.
    """
    output_file = output_path + ".json"
    classifier_data = []
    for _, row in dataset.iterrows():
        label = 1 if row['generated'] == 1 else 0
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"
        classifier_data.append({"input": input_text, "label": label})

    df = pd.DataFrame(classifier_data)
    if write_json:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classifier_data, f, ensure_ascii=False, indent=4)
    return df
def pull_kaggle_test_set():
    path = "./external_sources/llm-detect-ai-generated-text"
    df = pd.read_csv(path + "/train_essays.csv")
    prompts = pd.read_csv(path + "/train_prompts.csv")
    prompt_dict = prompts.set_index('prompt_id')['instructions'].to_dict()
    df['prompt_text'] = df['prompt_id'].map(prompt_dict)
    df.rename(columns={'text': 'essay_text'}, inplace=True)
    df.drop(columns=['id','prompt_id'], inplace=True)
    return df


def finetune(dataset_df: pd.DataFrame,
             model_name: str,
             output_dir: str,
             epochs: int = 3,
             batch_size: int = 8,
             access_token=None,
):
    """
    Receives dataset as dataframe.
    Assumes model is naturally classifier.
    Performs datasplit to train-validation inside function.
    """
    # Load tokenizer and model
    logger.debug("Loading tokenizer and model.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Replace `2` with your number of classes
    logger.debug("Loaded tokenizer and model successfully.")

    dataset = Dataset.from_pandas(dataset_df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    def preprocess_function(examples):
        # Tokenize the inputs and create attention masks
        inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=128)
        labels = examples["label"]
        print(labels)
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    logger.debug("Dataset tokenized successfully.")
    print(tokenized_dataset["train"][0])  # Check the structure
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    # Perform finetuning and save model to known location with indicative name
    # "./models"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # Use a separate validation set in practice
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}")


def inference(test_set: pd.DataFrame, model_path: str):
    """
    Perform inference using a fine-tuned DistilBERT classifier model and compute the loss.

    Args:
        test_set (pd.DataFrame): DataFrame containing the test data.
                                 It must have 'input' and 'label' columns.
        model_path (str): Path to the fine-tuned model.

    Returns:
        float: Computed loss on the test set.
    """
    # Load the tokenizer and model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

    # Convert the test set DataFrame to a HuggingFace Dataset
    dataset = Dataset.from_pandas(test_set)

    # Preprocess the test set
    def preprocess_function(examples):
        return tokenizer(examples["input"], truncation=True, padding=True, max_length=128)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Prepare the DataLoader for inference
    test_loader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=lambda x: {
        "input_ids": torch.tensor([item["input_ids"] for item in x]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in x]),
        "labels": torch.tensor([item["label"] for item in x]),
    })

    total_loss = 0.0
    total_samples = 0

    # Perform inference
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "labels": batch["labels"].to(model.device),
            }

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss  # Cross-entropy loss
            total_loss += loss.item() * len(inputs["labels"])  # Multiply by batch size
            total_samples += len(inputs["labels"])

    # Compute the average loss
    average_loss = total_loss / total_samples

    return average_loss


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
         '--sources',
        nargs="+",
        type=str,
     )

    argparser.add_argument(
        '--base_model',
        type=str,
        default='distilbert-base-uncased'
    )
    argparser.add_argument(
        '--save_dataset',
        action='store_true',
        default=False,
    )
    argparser.add_argument(
        '--load_dataset_from_path',
        type=str,
        help='Insert relative path to dataset pickle file.'
    )
    argparser.add_argument(
        '--path_to_model',
        type=str,
        help='If you want to perform inference on a model that was already finetuned, insert path to it here.'
    )
    args = argparser.parse_args()
    sources = args.sources
    sample_size = 10000
    # Loading and formatting training data.
    if args.load_dataset_from_path:
        data_in_df_format = pd.read_pickle(args.load_from_path)
    else:
        data_in_df_format = merge_data_for_finetuning(sources, sample_size=sample_size)
    output_path = f"./data/training_data_version_{dataset_version}_size_{sample_size}_sources_{'-'.join(sources)}"
    if args.save_dataset:
        data_in_df_format.to_pickle(f"{output_path}.pickle")
    classifier_input_data = write_classifier_format(data_in_df_format,output_path,args.save_dataset)

    # Perform finetuning.
    logger.debug("Loaded and saved datasets successfuly. Performing finetuning.")
    model_output_dir = f"./models/modelname_{args.base_model}_version_{dataset_version}_sources_{'-'.join(sources)}"
    finetune(classifier_input_data,model_name=args.base_model, output_dir=model_output_dir)

    # Perform inference.
    logger.debug("Finetuning successful. Performing inference.")
    test_output_path = f"./data/test_data_version_{dataset_version}_size_{sample_size}_sources_{'-'.join(sources)}"
    test_set = write_classifier_format(pull_kaggle_test_set(), output_path=test_output_path)
    path_to_model = args.path_to_model if args.path_to_model else model_output_dir
    results = inference(test_set, path_to_model)
    logger.debug(f"Average loss on test set is: {results}.")
