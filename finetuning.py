from fetch_data import download_kaggle_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from loguru import logger
import json
import os
# assume that .json data is saved to

# .pt

# Data file will be called "train_v160125.pickle"
# Identify modelname saved after finetuning by same version.


def write_classifier_format_and_split(dataset: pd.DataFrame, output_path: str, test_size: float = 0.2, seed: int = 42):
    """
    Writes the dataset into the classifier format as a JSON file and splits it into 80% training and 20% testing data.
    Args:
        dataset (pd.DataFrame): The input dataset with columns 'prompt_text', 'essay_text', 'generated', and 'source'.
        output_path (str): Path to the output data dir.
        test_size (float): Proportion of the dataset to use as the test set (default is 0.2 for 20%).
        seed (int): Random seed for reproducibility.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Format the dataset into classifier format
    classifier_data = []
    for _, row in dataset.iterrows():
        label = 1 if row['generated'] == 1 else 0
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"
        classifier_data.append({"input": input_text, "label": label})

    formatted_df = pd.DataFrame(classifier_data)

    # Split the dataset into 80% training and 20% testing
    train_df, test_df = train_test_split(formatted_df, test_size=test_size, random_state=seed)

    # Write the train and test data to JSON files
    train_output_file = output_path + "_train.json"
    test_output_file = output_path + "_test.json"

    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

    with open(test_output_file, 'w', encoding='utf-8') as f:
        json.dump(test_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

    return train_df, test_df
def pull_kaggle_example_data():
    path = "./external_sources/llm-detect-ai-generated-text"
    df = pd.read_csv(path + "/train_essays.csv")
    prompts = pd.read_csv(path + "/train_prompts.csv")
    prompt_dict = prompts.set_index('prompt_id')['instructions'].to_dict()
    df['prompt_text'] = df['prompt_id'].map(prompt_dict)
    df.rename(columns={'text': 'essay_text'}, inplace=True)
    df.drop(columns=['id','prompt_id'], inplace=True)
    return df


def finetune(train_df: pd.DataFrame,
             test_df: pd.DataFrame,
             model_name: str,
             output_dir: str,
             epochs: int = 3,
             batch_size: int = 8,
):
    """
    Fine-tunes a model using the provided training and validation datasets.
    """
    # Load tokenizer and model
    logger.debug("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Update num_labels if needed
    logger.debug("Loaded tokenizer and model successfully.")

    # Convert train and test dataframes to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Define preprocessing function to tokenize inputs
    def preprocess_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=128)
        inputs["labels"] = examples["label"]  # Add labels to the tokenized inputs
        return inputs

    # Apply tokenization to train and test datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    logger.debug("Datasets tokenized successfully.")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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

    # Initialize Trainer with train and validation datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"{output_dir}")
    logger.info("Model fine-tuned and saved successfully.")


def inference(test_df: pd.DataFrame, model_name: str, output_dir: str):
    """
    Perform inference using the fine-tuned model on the test dataset.

    Args:
        test_df (pd.DataFrame): The test dataset as a DataFrame.
        model_name (str): The name of the pre-trained model.
        output_dir (str): The directory where the fine-tuned model is saved.
    """
    logger.info("Loading fine-tuned model for inference...")

    # Load the tokenizer and model from the output directory
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    logger.info("Model and tokenizer loaded successfully.")

    # Convert the test dataset to a Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # Define the preprocessing function to tokenize inputs
    def preprocess_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=128)
        return inputs

    # Apply tokenization to the test dataset
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    logger.info("Test dataset tokenized successfully.")

    # Use the Trainer's predict method for inference
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(tokenized_test_dataset)

    # Get the predicted class labels
    predicted_labels = predictions.predictions.argmax(axis=-1)

    # Add predictions to the test DataFrame
    test_df['predicted_label'] = predicted_labels
    logger.info("Inference completed successfully.")

    return test_df


if __name__ == '__main__':
    model_name = "distilbert-base-uncased"
    output_dir = f"./models/{model_name}"
    data_output_path = "./data"

    # Step 1: Fetch and process data
    full_df = pull_kaggle_example_data()

    # Step 2: Format and split data into train and test sets
    train_df, test_df = write_classifier_format_and_split(full_df, data_output_path)

    # Step 3: Fine-tune the model using the training dataset and validate on the test dataset
    finetune(
        train_df=train_df,
        test_df=test_df,
        model_name=model_name,
        output_dir=output_dir,
        epochs=3,
        batch_size=8,
    )

    # Step 4: Perform inference on the test dataset
    test_results = inference(test_df, model_name, output_dir)

    # Save the results to a file
    test_results.to_csv("./data/test_results.csv", index=False)

    # Print a confirmation
    print("Inference results saved to ./data/test_results.csv")



