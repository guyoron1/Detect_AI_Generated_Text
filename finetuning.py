import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from loguru import logger
import os


def load_merged_dataset(path: str):
    """
    Loads the merged dataset from the given path.
    Args:
        path (str): Path to the merged dataset CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The dataset at path '{path}' was not found.")
    logger.info(f"Loading merged dataset from {path}.")
    return pd.read_csv(path)


def format_dataset(dataset: pd.DataFrame):
    """
    Formats the dataset into the classifier format required for fine-tuning.
    Args:
        dataset (pd.DataFrame): The input dataset with columns 'prompt_text', 'essay_text', 'generated', and 'source'.

    Returns:
        pd.DataFrame: A formatted dataset ready for fine-tuning.
    """
    classifier_data = []
    for _, row in dataset.iterrows():
        label = 1 if row['generated'] == 1 else 0
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"
        classifier_data.append({"input": input_text, "label": label})

    return pd.DataFrame(classifier_data)


def finetune(full_df: pd.DataFrame,
             model_name: str,
             output_dir: str,
             epochs: int = 3,
             batch_size: int = 8):
    """
    Fine-tunes a model using the entire dataset provided.
    Args:
        full_df (pd.DataFrame): The entire dataset formatted for fine-tuning.
        model_name (str): The name of the pre-trained model to fine-tune.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
    """
    # Load tokenizer and model
    logger.debug("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    logger.debug("Loaded tokenizer and model successfully.")

    # Convert the dataset to a Hugging Face Dataset
    dataset = Dataset.from_pandas(full_df)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Define preprocessing function to tokenize inputs
    def preprocess_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=128)
        inputs["labels"] = examples["label"]
        return inputs

    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    logger.debug("Dataset tokenized successfully.")

    # Define training arguments
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",  # No evaluation as we're using the full dataset
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=False,  # Disable this since evaluation is disabled
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"{output_dir}")
    logger.info("Model fine-tuned and saved successfully.")


if __name__ == '__main__':
    model_name = "distilbert-base-uncased"
    output_dir = f"./models/{model_name}"
    dataset_path = "./data/merged_fine_tuning_data.csv"

    # Step 1: Load the merged dataset
    merged_df = load_merged_dataset(dataset_path)

    # Step 2: Format the merged dataset
    formatted_df = format_dataset(merged_df)

    # Step 3: Fine-tune the model using the entire dataset
    finetune(
        full_df=formatted_df,
        model_name=model_name,
        output_dir=output_dir,
        epochs=3,
        batch_size=8,
    )

    # Print a confirmation
    print(f"Model fine-tuned and saved in {output_dir}.")
