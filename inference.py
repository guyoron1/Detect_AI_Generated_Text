from finetuning import pull_kaggle_example_data


def inference():
    # Load finetuned model form "./models" and perform inference
    test_data = pull_kaggle_example_data("test")
    # todo: perform inference on test data, print loss to terminal / plot loss