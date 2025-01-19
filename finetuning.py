from fetch_data import download_kaggle_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from loguru import logger
import json
# assume that .json data is saved to

# .pt

# Data file will be called "train_v160125.pickle"
# Identify modelname saved after finetuning by same version.


def write_classifier_format(dataset: pd.DataFrame, output_path: str):
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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classifier_data, f, ensure_ascii=False, indent=4)
    return df
def pull_kaggle_example_data():
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


def inference():
    # Load finetuned model form "./models" and perform inference
    pass


if __name__ == '__main__':
    model_name = "distilbert-base-uncased"
    output_dir = f"./models/{model_name}"
    data_output_path = "./data"
    df = write_classifier_format(pull_kaggle_example_data(),data_output_path)

    finetune(
        dataset_df=df,
        model_name=model_name,
        output_dir=output_dir,
    )
