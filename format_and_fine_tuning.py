import json
import pandas as pd
from transformers import AutoModel, pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sample_data_frame import df


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


def write_mistral_format(dataset: pd.DataFrame, output_file: str):
    """
    Writes the dataset into the Mistral format as a JSON file.

    Args:
        dataset (pd.DataFrame): The input dataset with columns
                                'prompt_text', 'essay_text', 'generated', and 'source'.
        output_file (str): Path to the output JSON file.
    """
    mistral_data = []
    for _, row in dataset.iterrows():
        label = "LLM" if row['generated'] == 1 else "Human"
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"
        mistral_data.append({"input": input_text, "label": label})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mistral_data, f, ensure_ascii=False, indent=4)


def inference_with_mistral(dataset_path: str, model_name: str):
    """
    Run inference using a pipeline with the Mistral GGUF model.

    Args:
        dataset_path (str): Path to the dataset in JSON format.
        model_name (str): Name of the Hugging Face model to load.
    """
    dataset = Dataset.from_json(dataset_path)
    model = AutoModel.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model)

    for example in dataset:
        prompt = example["input"]
        result = pipe(prompt, max_length=100)
        print(f"Input: {prompt}\nGenerated: {result[0]['generated_text']}\n")


def fine_tune_mistral(dataset_path: str, model_name: str, output_dir: str, epochs: int = 3, batch_size: int = 8):
    """
    Fine-tune a Hugging Face-compatible Mistral model on a classification task.
    """
    dataset = DatasetDict.from_json(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preprocess_function(examples):
        return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

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
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    # Step 1: Pull data from Kaggle
    example_df = df
    mistral_json_path = "mistral_formatted_data.json"
    write_mistral_format(example_df, mistral_json_path)
    print(f"Example dataset has been written to {mistral_json_path}.")

    # Step 2: Use GGUF model for inference
    inference_with_mistral(
        dataset_path=mistral_json_path,
        model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    )

    # Step 3 (Optional): Fine-tune a compatible Hugging Face model
    # Uncomment the code below for fine-tuning if using a different model
    # fine_tune_mistral(
    #     dataset_path=mistral_json_path,
    #     model_name="mistralai/Mistral-7B-Instruct-v0.3",  # Example compatible model
    #     output_dir="fine_tuned_mistral",
    #     epochs=3,
    #     batch_size=8
    # )
