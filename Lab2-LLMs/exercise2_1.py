# Exercise 2.1
from transformers import GPT2Tokenizer, set_seed

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    text = '''
    Nel mezzo del cammin di nostra vita
    mi ritrovai per una selva oscura
    che' la diritta via era smarrita.
                           '''
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    print(f"Tokenized text {tokenized_text}")
    print(f"Text length {len(text)}, tokenized length {len(tokenized_text[0])}")

    # Text length 143, tokenized length 78