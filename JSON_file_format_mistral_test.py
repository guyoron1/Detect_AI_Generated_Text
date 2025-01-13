import json
import pandas as pd


# Define the function to write the Mistral format
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


# Main block to test with a sample DataFrame
if __name__ == "__main__":
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

    # Path to the output JSON and CSV files
    json_output_file = "example_mistral_formatted.json"
#    dataframe_output_file = "example_dataframe.csv"  # <--- Added this line

    # Save the DataFrame to a CSV file
#    example_df.to_csv(dataframe_output_file, index=False)  # <--- Added this line
#    print(f"Sample DataFrame has been written to {dataframe_output_file}.")  # <--- Added this line

    # Run the write_mistral_format function on the example DataFrame
    write_mistral_format(example_df, json_output_file)
    print(f"Example dataset has been written to {json_output_file}.")
