from transformers import pipeline

# Initialize the global pipeline once, so that the proccess is faster
global_pipe = pipeline("text-generation", model="openai-community/gpt2")

def generate_prompt_for_text(text, pipe, prompt_prefix="Generate a logical prompt for: "):
    """
        text - a string : The input text for which to generate a prompt.
        pipe: The Hugging Face pipeline for text generation.
        prompt_prefix - a string: Prefix to guide the text generation.
    """
    # Combine the prefix and text to create the input for the model
    model_input = f"{prompt_prefix}{text}"

    # Use the pipeline to generate text
    # need to adjust the parameters
    #generated = pipe(
    #    model_input,
    #    max_new_tokens=50,  # Generate up to 50 new tokens
    #    num_return_sequences=1,g
    #    do_sample=True
    #)
    generated = pipe(model_input, max_length=50, num_return_sequences=1, do_sample=True)


    # Extract the generated text and remove the prefix
    generated_prompt = generated[0]['generated_text'][len(prompt_prefix):].strip()

    return generated_prompt


def main():

    input_text = "artificial intelligence impact modern education in several way. this essay will focus on artificial intelligence in schools in united states and specifically on children between ages 10-16"
    generated_prompt = generate_prompt_for_text(input_text, global_pipe)

    print("Input Text:", input_text)
    print("Generated Prompt:", generated_prompt)

if __name__ == "__main__":
    main()
