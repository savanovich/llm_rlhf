import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_eval import generate_eval

device = "cuda" if torch.cuda.is_available() else "cpu"


EVAL_PROMPTS = [
    "Question: How to create an attention layer in pytorch? \n\n Answer: ",
    "Question: I recieve TypeError with a following code: ```a=[1, 2, 3]    a.extend(4)```. How can I fix it?  \n\n Answer: ",
    "Question: What library would you recommend for visualizing 3D points in python? \n\n Answer: ",
]


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    generate_eval(model, tokenizer, EVAL_PROMPTS)
