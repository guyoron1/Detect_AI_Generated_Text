from transformers import pipeline

# Initialize the global pipeline once for efficiency
global_pipe = pipeline("text2text-generation", model="google/flan-t5-large")


def generate_prompt_for_text(text,
                             pipe,
                             prompt_prefix="This is an essay written by a student in response to an assignment:\n",
                             prompt_suffix= "\nWrite the assignment that you think was given to the student."):
    """
    Generates an instructional prompt for the input text.
    """
    # Combine prefix and input text
    model_input = f"{prompt_prefix} {text} {prompt_suffix}"
    generated = pipe(
        model_input,
        max_new_tokens=200,  # Control output length
        num_return_sequences=1,
        do_sample=True,  # Enable sampling for diverse outputs
        temperature=0.8  # Adjust creativity (lower = more focused, higher = more creative)
    )

    # Extract the generated text
    generated_prompt = generated[0]['generated_text'].strip()
    return generated_prompt


def main():
    input_text = """I think that there is a lot of reason to stop the electoral college one reason is that nobody wants to vote for
    the people, and also they are not fair, i think that the electoral college should be abolished soon."""
    generated_prompt = generate_prompt_for_text(input_text, global_pipe)
    print("Input Text:", input_text)
    print("Generated Prompt:", generated_prompt)

if __name__ == "__main__":
    main()
