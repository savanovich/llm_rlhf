import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_eval(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, eval_prompts):
    print("Starting Evaluation...")
    model = model.to(device)
    model.eval()
    for eval_prompt in eval_prompts:
        batch = tokenizer(eval_prompt, return_tensors="pt").to(device)

        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=128)

        print("\n\n", textwrap.fill(tokenizer.decode(output_tokens[0], skip_special_tokens=False)))
        print("*" * 100)