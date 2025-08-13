from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from lmm_robustness.data import cifar10

MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
GENERATION_CONFIG = GenerationConfig.from_pretrained(MODEL_PATH)
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

USER_START = '<|user|>'
USER_END = '<|end|>'
ASSISTANT_START = '<|assistant|>'


def load_lmm():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation='eager',
    ).cuda()
    print(f"Model loaded from {MODEL_PATH} on device {model.device}")
    return model


def get_prompt(categories=cifar10.CLASS_NAMES):
    categories_str = ", ".join(categories[:-1]) + f" or {categories[-1]}"
    prompt = (
        f"{USER_START} This image shows either a {categories_str}."
        f"<|image_1|> Which one is it? {USER_END}"
        f"{ASSISTANT_START}It shows a"  # No tailing space!
    )
    print(f"\nPrompt: \"{prompt}\"\n")
    return prompt


def generate_responses(images, prompts, model, response_length=1):
    inputs = PROCESSOR(text=prompts, images=images, return_tensors='pt')
    inputs = inputs.to('cuda:0')

    all_token_ids = model.generate(
        **inputs,
        max_new_tokens=response_length,
        generation_config=GENERATION_CONFIG,
        num_logits_to_keep=1,  # Bug fix!
    )
    prompt_length = inputs['input_ids'].shape[1]
    generated_ids = all_token_ids[:, prompt_length:]
    responses = PROCESSOR.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return responses


