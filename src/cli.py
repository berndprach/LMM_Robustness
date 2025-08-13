import textwrap

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
Tasks:
- Use past_key_values for efficiency.
- Implement gui for this project.
- Package everything up using Docker. 
"""

STARTING_TOKEN = {
    "gpt2": None,
    "roberta-base": None,
}


def run_cli(model_name = "gpt2", **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./data')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./data', **model_kwargs)

    # text = ""
    # encoded_input = tokenizer(text, return_tensors='pt')
    # print(encoded_input)
    text = input("Provide the first word: ").replace(r"\n", "\n")

    while True:
        # print(text)
        # print(textwrap.fill(text, width=66))
        print("\n".join(textwrap.fill(line, width=66) for line in text.split("\n")))

        encoded_input = tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            output_tokens = model(**encoded_input)

        next_token_logits = output_tokens.logits[0, -1, :]
        # next_token_scores = next_token_logits.softmax(dim=-1)

        # past_key_values = output_tokens.past_key_values

        _, top3_indices = torch.topk(next_token_logits, 3, dim=-1, sorted=True)
        # top3_words = tokenizer.batch_decode(top3_indices, skip_special_tokens=True)
        # top3_words = tokenizer.batch_decode(top3_indices)
        top3_words = [tokenizer.decode(i, skip_special_tokens=False) for i in top3_indices]
        # print(top3_words)
        print(", ".join(f"{i+1}: {repr(word)}" for i, word in enumerate(top3_words)))

        choice = input("Select word it or provide an alternative next word:")
        # print(choice)
        if choice.isdigit() and 1 <= int(choice) <= 3:
            i = int(choice) - 1
            next_word = top3_words[i]
        else:
            next_word = choice.replace(r"\n", "\n")
            if next_word == "":
                next_word = "\n"

        if next_word == "<|endoftext|>" or next_word == "exit()":
            break

        # print(repr(next_word))
        text += next_word

    # Take token with highest probability
    # next_token = next_token_scores.argmax().unsqueeze(0).unsqueeze(0)
    # print(next_token)

    # output = tokenizer.batch_decode(next_token, skip_special_tokens=True)
    return text

    generated_sequence = torch.tensor([[tokenizer.sep_token_id]])  # initial token
    input_ids = batch["input_ids"]
    past_key_values = None

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            decoder_input_ids=generated_sequence,
            past_key_values=past_key_values
        )

    encoder_outputs = output.encoder_last_hidden_state

    # Generation loop
    while True:
        # From here on, use cached attention
        past_key_values = output.past_key_values
        next_token_logits = output.logits[:, -1, :]
        next_token_scores = next_token_logits.softmax(dim=-1)
        next_token = next_token_scores.argmax().unsqueeze(0).unsqueeze(0)  # greedy decoding
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        # Stop if EOS token generated
        if (generated_sequence.squeeze()[-1] == tokenizer.eos_token_id):
            break
        with torch.no_grad():
            output = model(
                decoder_input_ids=torch.tensor([[generated_sequence.squeeze()[-1]]]),
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs
            )

    summary = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)



if __name__ == "__main__":
    # full_text = run_cli()
    # full_text = run_cli("HuggingFaceTB/SmolLM-135M")
    full_text = run_cli("HuggingFaceTB/SmolLM-360M")
    # full_text = run_cli("HuggingFaceTB/SmolLM-1.7B")
    # full_text = run_cli("HuggingFaceTB/SmolLM3-3B)
    # full_text = run_cli("roberta-base", is_decoder=True)

    print("Full text:")
    print("="*66)
    # print(full_text)
    print("\n".join(textwrap.fill(line, width=66) for line in full_text.split("\n")))
    print("="*66)
