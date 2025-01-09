from transformers import pipeline

# Initialize the global pipeline once for efficiency
global_pipe = pipeline("text-generation", model="openai-community/gpt2")


def generate_prompt_for_text(text, pipe, prompt_prefix="Create an instructional task or prompt..."):
    """
    Generates an instructional prompt for the input text.
    """
    # Combine prefix and input text
    model_input = f"{prompt_prefix}{text}"

    # Generate output with adjusted parameters
    generated = pipe(
        model_input,
        max_new_tokens=75,  # Ensures up to 75 tokens are allocated for the output
        num_return_sequences=1,
        do_sample=True
    )

    # Extract and clean the generated text
    generated_prompt = generated[0]['generated_text'][len(prompt_prefix):].strip()

    # Ensure the output is not truncated
    if not generated_prompt.endswith('.'):
        generated_prompt += '...'  # Mark incomplete sentences for clarity

    return generated_prompt


def main():
    input_text = ("""Artificial intelligence impacts modern education in several ways. This essay will focus on artificial intelligence in schools in the United States and specifically on children between ages 10-16.""")
    generated_prompt = generate_prompt_for_text(input_text, global_pipe)

    print("Input Text:", input_text)
    print("Generated Prompt:", generated_prompt)


if __name__ == "__main__":
    main()
