from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import argparse

def generate_from_prompt(prompt, max_new_tokens = 50, seed = 42):
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    # trying different settings
    # params that can be used: do_sample to sample from model's predicted probabilities, 
    # num_beams to control the number of beams used for beam search
    # no_repeat_ngram_size to avoid repeating n-grams in the output
    # early_stopping to stop generation when the model predicts EOS token.
    # temperature to control the randomness of the output : higher temperature -> more randomness
    # Base version
    set_seed(seed)
    text_filename = "output_gpt2.txt"
    with open(text_filename, "w") as f:
        f.write(f"Prompt:\n{prompt}\n")
        f.write("--------------------------------------------------\n\n")
    output = model.generate(**inputs, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id)
    o = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Decoded base output: {o}")
    with open(text_filename, "a") as f:
        f.write(f"Base output:\n{o}\n")
        f.write("--------------------------------------------------\n\n")
    print("--------------------------------------------------")
    # + Do_sample = True
    set_seed(seed)
    output = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=True, pad_token_id = tokenizer.eos_token_id)
    o = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Decoded 'do_sample = True' output: {o}")
    with open(text_filename, "a") as f:
        f.write(f"Do_sample = True output:\n{o}\n")
        f.write("--------------------------------------------------\n\n")
    print("--------------------------------------------------")
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
        # + temperature
        set_seed(seed)
        output = model.generate(**inputs, max_new_tokens = max_new_tokens, temperature=temp, do_sample = True, pad_token_id = tokenizer.eos_token_id)
        o = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Decoded 'do_sample = True, temperature={temp}' output: {o}")
        with open(text_filename, "a") as f:
            f.write(f"Do_sample = True, temperature={temp} output:\n{o}\n")
            f.write("--------------------------------------------------\n\n")
        print("--------------------------------------------------")
    # + no_repeat_ngram_size = 2
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
        # + temperature
        set_seed(seed)
        output = model.generate(**inputs, max_new_tokens = max_new_tokens, no_repeat_ngram_size = 2, temperature=temp, do_sample = True, pad_token_id = tokenizer.eos_token_id)
        o = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Decoded 'do_sample = True, temperature={temp}, no_repeat_ngram_size = 2' output: {o}")
        with open(text_filename, "a") as f:
            f.write(f"Do_sample = True, temperature={temp}, no_repeat_ngram_size = 2 output:\n{o}\n")
            f.write("--------------------------------------------------\n\n")
        print("--------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser = parser.parse_args()
    generate_from_prompt(parser.text, max_new_tokens=100)