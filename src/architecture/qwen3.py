from transformers import AutoModelForCausalLM, AutoTokenizer

""" https://huggingface.co/Qwen/Qwen3-0.6B """

MODEL_NAME = "Qwen/Qwen3-0.6B"

class Qwen:
    def __init__(self):
        print("Loading Qwen.")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        self.prompt = "Give me a short introduction to large language model."
        self.messages = [{"role": "user", "content": self.prompt}]

    def __call__(self):
        model_inputs = self.tokenize(self.messages)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        thinking_content, content = self.decode(output_ids)

        print("thinking content:", thinking_content)
        print("content:", content)

    def tokenize(self, messages, think=False):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return model_inputs

    def decode(self, output_ids):
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content

