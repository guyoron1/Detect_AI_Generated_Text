import pandas as pd
from Detect_AI_Generated_Text.format import format_all_datasets


def sample_by_percentages(df: pd.DataFrame, percentages: dict) -> pd.DataFrame:
    """
    Samples the dataset based on the given percentages for each source.

    Args:
        df (pd.DataFrame): The input dataset with a 'source' column.
        percentages (dict): A dictionary mapping sources to their respective percentages (0-1).

    Returns:
        pd.DataFrame: A new DataFrame with the sampled data.
    """
    sampled_data = []

    for source, percentage in percentages.items():
        # Filter rows belonging to the current source
        source_df = df[df['source'] == source]

        # Calculate the number of samples
        num_samples = int(len(source_df) * percentage)

        # Sample the data
        sampled_df = source_df.sample(n=num_samples, random_state=42)  # Setting random_state for reproducibility

        # Add the sampled data to the list
        sampled_data.append(sampled_df)

    # Concatenate all sampled dataframes
    result_df = pd.concat(sampled_data, ignore_index=True)
    return result_df


# Example usage
if __name__ == "__main__":
    # Assuming `df` is obtained from the `format_all_datasets()` function
    df = format_all_datasets()

    # Define the sampling percentages
    sampling_percentages = {
        "fpe": 0.5,
        "daigt": 0.3,
        "persuade": 0.2,
    }

    # Sample the dataset
    sampled_df = sample_by_percentages(df, sampling_percentages)

    # Save the sampled data to a CSV file (optional)
    sampled_output_file = "sampled_dataset.csv"
    sampled_df.to_csv(sampled_output_file, index=False)
    print(f"Sampled dataset has been written to {sampled_output_file}.")
