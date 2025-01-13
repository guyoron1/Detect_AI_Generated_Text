import pandas as pd
import json

# Create a small example dataset (merged from your earlier request)
data = {
    "prompt_text": [
        "What are the benefits of exercise?", "Describe the importance of technology in education.",
        "Why is persuasion important?", "Explain the role of teamwork in success.", "What challenges arise with remote work?",
        "How can one improve communication skills?", "What is the impact of climate change on agriculture?",
        "How does technology affect social interaction?", "What motivates people to volunteer?", "What is the importance of time management?",
        "How do different cultures view mental health?", "What is the role of government in addressing poverty?",
        "How does artificial intelligence affect society?", "What is the value of creativity in education?", "Why is critical thinking important?",
        "How does exercise improve cognitive function?", "What role does leadership play in success?", "How can one enhance public speaking skills?",
        "What is the future of remote work?", "How does social media impact self-esteem?", "What are the effects of sleep deprivation?",
        "Why is ethical decision-making important?", "What are the benefits of reading books?", "What is the impact of online learning on education?",
        "How do you build trust in a relationship?", "What is the role of sports in promoting teamwork?", "How can one develop emotional intelligence?",
        "What motivates people to work hard?", "Why is sustainability important?", "What is the impact of globalization on culture?"
    ],
    "essay_text": [
        "Exercise improves mental health and helps in maintaining physical fitness.", "Technology makes education more accessible and personalized.",
        "Persuasion is vital in debates and negotiation to convey ideas effectively.", "Teamwork helps individuals achieve goals by leveraging collective effort.",
        "Remote work can lead to challenges in collaboration and time management.", "Effective communication involves listening actively and expressing clearly.",
        "Climate change leads to decreased crop yields and threatens food security.", "Technology can both enhance and diminish the quality of social interactions.",
        "Volunteering is driven by a sense of purpose and the desire to help others.", "Time management helps individuals prioritize tasks and increase productivity.",
        "Mental health is viewed differently across cultures, with varying approaches to treatment.", "Governments play a crucial role in addressing poverty through policy and aid programs.",
        "AI has the potential to revolutionize industries, but also poses ethical challenges.", "Creativity fosters problem-solving skills and promotes innovation in education.", "Critical thinking is essential for making informed decisions and solving complex problems.",
        "Exercise enhances brain function by increasing blood flow and stimulating neural activity.", "Leadership inspires others to follow a vision and motivates teams to achieve common goals.",
        "Public speaking skills are crucial for effective communication and influencing others.", "Remote work is becoming more popular, offering flexibility and work-life balance.",
        "Social media can lead to unrealistic comparisons and lower self-esteem in individuals.", "Lack of sleep affects cognitive function, mood, and overall health.",
        "Ethical decision-making helps guide actions in line with moral principles and societal norms.", "Reading books improves knowledge, vocabulary, and mental stimulation.",
        "Online learning provides flexibility but lacks face-to-face interaction and hands-on experiences.", "Building trust requires consistency, honesty, and mutual respect in relationships.",
        "Sports promote teamwork by fostering collaboration, communication, and mutual support.", "Emotional intelligence helps individuals navigate social interactions and manage emotions.",
        "Motivation to work hard comes from a sense of purpose, ambition, and personal goals.", "Sustainability ensures that resources are used responsibly, protecting future generations.",
        "Globalization influences culture by promoting cultural exchange and creating new opportunities."
    ],
    "generated": [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    "source": ["fpe", "daigt", "persuade", "fpe", "daigt", "persuade", "fpe", "daigt", "persuade", "fpe",
               "daigt", "persuade", "fpe", "daigt", "persuade", "fpe", "daigt", "persuade", "fpe", "daigt",
               "persuade", "fpe", "daigt", "persuade", "fpe", "daigt", "persuade", "fpe", "daigt", "persuade"]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file (optional, for inspection)
df.to_csv("sample_dataset.csv", index=False)

# Display the DataFrame
print("Sample dataset:")
print(df)


# Function to write data into Mistral format
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


# Function to sample by percentages
def sample_by_percentages(df: pd.DataFrame, percentage_dict: dict):
    sampled_data = []

    # Loop over each source in the percentage_dict
    for source, percentage in percentage_dict.items():
        # Filter the DataFrame by the source
        source_df = df[df['source'] == source]

        # Calculate the sample size based on the percentage
        sample_size = int(len(source_df) * percentage)

        # Sample the data
        sampled_data.append(source_df.sample(n=sample_size, random_state=42))  # Random state for reproducibility

    # Concatenate all sampled data
    sampled_df = pd.concat(sampled_data)
    return sampled_df


# Example usage: Sample by percentages
datamix = {"fpe": 0.5, "daigt": 0.3, "persuade": 0.2}
sampled_df = sample_by_percentages(df, datamix)

# Print the sampled dataset
print("\nSampled dataset:")
print(sampled_df)

# Write the dataset into Mistral format
output_file_path = "mistral_formatted_data.json"
write_mistral_format(sampled_df, output_file_path)

print(f"Sampled dataset has been written to {output_file_path} in Mistral format.")
