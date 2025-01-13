import json
import pandas as pd
from Detect_AI_Generated_Text.format import format_all_datasets


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
        # Determine the label based on the 'generated' column
        label = "LLM" if row['generated'] == 1 else "Human"

        # Construct the input string
        input_text = f"Prompt Text: {row['prompt_text']}. Essay Text: {row['essay_text']}"

        # Add the entry to the Mistral data list
        mistral_data.append({
            "input": input_text,
            "label": label
        })

    # Write the JSON data to the specified output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mistral_data, f, ensure_ascii=False, indent=4)


# Example usage
if __name__ == "__main__":
    # Assuming `df` is the formatted DataFrame obtained from `format_all_datasets()`
    df = format_all_datasets()

    # Path to the output JSON file
    output_file_path = "mistral_formatted_data.json"

    # Write the dataset into the required format
    write_mistral_format(df, output_file_path)

    print(f"Dataset has been written to {output_file_path} in Mistral format.")

import pandas as pd

# Create a small example DataFrame
data = {
    "prompt_text": [
        "What are the benefits of exercise?",
        "Describe the importance of technology in education."
    ],
    "essay_text": [
        "Exercise improves mental and physical health, helping people lead a balanced life.",
        "Technology enhances learning by providing access to resources and enabling remote education."
    ],
    "generated": [0, 1],  # 0 for Human, 1 for LLM
    "source": ["human_dataset", "llm_dataset"]
}

# Convert the dictionary to a DataFrame
example_df = pd.DataFrame(data)

# Display the DataFrame
print(example_df)

# Run the write_mistral_format function on the example DataFrame
output_file_path = "example_mistral_formatted.json"
write_mistral_format(example_df, output_file_path)

print(f"Example dataset has been written to {output_file_path}.")
