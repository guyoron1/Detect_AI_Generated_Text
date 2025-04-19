import format
from finetuning import write_classifier_format, pull_kaggle_test_set, inference
import fetch_data
import pandas as pd

baseline_model = "distilbert-base-uncased"
version = "v10-04-2025"
finetune_size = 5000
sources = ['fpe', 'persuade']
test_size = 700
TEST_GENERATED_PERCENTAGE = 0.5 # This is based on our coin-toss model submitted to Kaggle - test distribution is 50-50 over labels.ס
ft_model_path = f"./models/modelname_{baseline_model}_version_{version}_size_{finetune_size}_sources_{'-'.join(sources)}"
baseline_model_path = f"./models/{baseline_model}"

# Load test set - currently, test set is train dataset provided in Kaggle challange
test_output_path = f"./data/test_data_version_{version}_size_{finetune_size}_sources_{'-'.join(sources)}"


def sample_by_percentage(df: pd.DataFrame, generated_percentage: float, total_num_samples: int):
    """
    Sample a DataFrame by selecting a specified percentage of 'generated' values.

    Args:
        df (pd.DataFrame): The input DataFrame that contains a 'generated' column.
        generated_percentage (float): The percentage of '1's in the sample.
        total_num_samples (int): The total number of samples to select.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled rows.
    """
    # Calculate how many 1s (generated) and 0s (non-generated) we need
    num_generated = int(total_num_samples * generated_percentage)  # Number of 1s
    num_non_generated = total_num_samples - num_generated  # Number of 0s

    # Separate the DataFrame into generated (1) and non-generated (0)
    generated_df = df[df['generated'] == 1]
    non_generated_df = df[df['generated'] == 0]

    # Sample the required number of rows from each group
    sampled_generated = generated_df.sample(n=num_generated, random_state=71)
    sampled_non_generated = non_generated_df.sample(n=num_non_generated, random_state=71)

    # Concatenate the sampled data and shuffle them (if needed)
    final_sampled_df = pd.concat([sampled_generated, sampled_non_generated], ignore_index=True)

    # Shuffle the final sample to mix 1s and 0s randomly
    final_sampled_df = final_sampled_df.sample(frac=1, random_state=71).reset_index(drop=True)

    return final_sampled_df
def main():
    # First, dataset is generated in dataframe format
    test_set_df_format = pull_kaggle_test_set()
    df = format.format_dataset('daigt')
    df = df[['essay_text', 'generated', 'prompt_text']]
    test_set_df_format = sample_by_percentage(pd.concat([test_set_df_format, df], ignore_index=True), TEST_GENERATED_PERCENTAGE, test_size)

    # Then, it is formatted to classifier format (input + label)
    test_set = write_classifier_format(test_set_df_format, output_path=test_output_path)
    test_set.head()

    # Perform inference on test set in classifier format.
    average_loss = inference(test_set, ft_model_path)
    print(f"Average loss on test set for finetuned model is {average_loss}.")

    # What is the loss for the baseline, distilbert model, before our finetuning?
    average_loss_baseline = inference(test_set, baseline_model_path)
    print(f"Average loss on test set for baseline model is {average_loss_baseline}")

    health_org_essay = "i think that donal trump decision to quit the health organization is not good because people are sick and\
    they cant do anything about it. i think that all the countries need to be in the health organization to cure the cancer and other disease that is\
    very important for mankind"
    vacation_essay = "The perfect vacation is not a one-size-fits-all concept;\
                      it is deeply personal and shaped by individual desires and priorities.\
                      For some, it might involve basking under the sun on a quiet tropical beach, where time seems to slow,\
                      and the only sound is the gentle lapping of waves. For others, it could mean exploring bustling cities filled with art,\
                      history, and culinary delights, where every corner reveals a new adventure."

    money_essay = "I think a society without money could work, but it would be really hard.\
    People would have to find a new way to trade things, like using food, skills, or services instead of money.\
    For example, if someone wanted bread, they could trade vegetables or help the baker fix something.\
    This sounds simple, but it could get really confusing if people don’t agree on what is fair."

    example_essays = [health_org_essay, vacation_essay, money_essay]

    generated_prompts = format.generate_prompts_for_texts(example_essays, format.GLOBAL_PIPE, batch_size=1)
    print(generated_prompts)

def trying_out_generating_prompts():
    df = format.format_dataset('')

if __name__ == '__main__':
    main()