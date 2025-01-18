import json
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def pull_kaggle_example_data():
    """
    Pulls example data from Kaggle and formats it for the workflow.

    Returns:
        pd.DataFrame: A DataFrame with columns 'prompt_text', 'essay_text', 'generated', and 'source'.
    """
    path = "./external_sources/llm-detect-ai-generated-text"
    df = pd.read_csv(path + "/train_essays.csv")
    prompts = pd.read_csv(path + "/train_prompts.csv")
    prompt_dict = prompts.set_index('prompt_id')['instructions'].to_dict()
    df['prompt_text'] = df['prompt_id'].map(prompt_dict)
    df.rename(columns={'text': 'essay_text'}, inplace=True)
    df.drop(columns=['id', 'prompt_id'], inplace=True)
    return df


def write_gpt2_format(dataset: pd.DataFrame, output_file: str):
    """
    Writes the dataset into the GPT-2 format as a JSON file.

    Args:
        dataset (pd.DataFrame): The input dataset with columns
                                'prompt_text', 'essay_text', 'generated', and 'source'.
        output_file (str): Path to the output JSON file.
    """
    gpt2_data = []
    for _, row in dataset.iterrows():
        label = 1 if row['generated'] == 1 else 0
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"
        gpt2_data.append({"input": input_text, "label": label})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gpt2_data, f, ensure_ascii=False, indent=4)


def fine_tune_gpt2(dataset_path: str, model_name: str, output_dir: str, epochs: int = 3, batch_size: int = 8):
    """
    Fine-tune a Hugging Face-compatible GPT-2 model on a classification task.
    """
    # Load the dataset
    dataset = Dataset.from_json(dataset_path)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print("Dataset split into train and validation sets.")

    # Load tokenizer and ensure padding token is set
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    print(f"Padding token set to: {tokenizer.pad_token}")

    # Define the preprocessing function
    def preprocess_function(examples):
        # Tokenize the inputs and create attention masks
        inputs = tokenizer(examples["input"], truncation=True, padding=True, max_length=512)
        inputs['attention_mask'] = [1 if i != tokenizer.pad_token_id else 0 for i in inputs['input_ids']]
        return inputs

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print("Dataset tokenized successfully.")

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")



if __name__ == "__main__":
    # Step 1: Pull data from Kaggle
    example_df = pull_kaggle_example_data()
    gpt2_json_path = "gpt2_formatted_data.json"
    write_gpt2_format(example_df, gpt2_json_path)
    print(f"Example dataset has been written to {gpt2_json_path}.")

    # Step 2: Fine-tune the GPT-2 model using the generated JSON dataset
    fine_tune_gpt2(
        dataset_path=gpt2_json_path,
        model_name="gpt2",  # Example compatible model
        output_dir="fine_tuned_gpt2",
        epochs=3,
        batch_size=8
    )
