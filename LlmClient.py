import platform

from dotenv import load_dotenv
from openai import OpenAI
if platform.system() == "Darwin":
    from mlx_lm import load, generate

# In this file, history is a list of dictionaries, as defined in the OpenAI api
# An example could be:
# his = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Hello, who are you?"},
# ]


SYSTEM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
USER = "<|start_header_id|>user<|end_header_id|>"
ASSISTENT = "<|start_header_id|>assistant<|end_header_id|>"

def generate_system_prompt(prompt):
    return f"{SYSTEM}\n\n{prompt}<|eot_id|>\n"


def generate_user_prompt(prompt):
    return f"{USER}\n\n{prompt}<|eot_id|>\n"


def generate_assistant_prompt(prompt):
    return f"{ASSISTENT}\n\n{prompt}<|eot_id|>\n"


def generate_final_prompt(prompt):
    return f"{prompt}\n\n{ASSISTENT}\n\n"


def generate_final_prompt_no_assistent(prompt):
    return f"{prompt}\n\n"


def history_to_prompt(history):
    prompt = ""
    for m in history:
        if m["role"] == "system":
            prompt += generate_system_prompt(m["content"])
        if m["role"] == "user":
            prompt += generate_user_prompt(m["content"])
        elif m["role"] == "assistant":
            prompt += generate_assistant_prompt(m["content"])

    prompt += f"{ASSISTENT}\n\n"

    return prompt


class MlxLlama():
    def __init__(self, temp, model_name="mlx-community/Meta-Llama-3-8B-Instruct-8bit"):
        model, tokenizer = load(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.temp = temp

    def run(self, history):
        response = generate(
            self.model,
            self.tokenizer,
            prompt=history_to_prompt(history),
            temp=self.temp,
            repetition_penalty=1.3,
            repetition_context_size=4000,
            max_tokens=1000,
        )

        return response


class OpenAi():
    def __init__(self, temp, model_name="gpt-3.5-turbo"):
        self.temp = temp

        load_dotenv()
        self.client = OpenAI()
        # Recommended to use 'gpt-3.5-turbo' or 'gpt-4o'
        self.model_name = model_name

    def run(self, history):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=self.temp
        )

        return response.choices[0].message.content
