import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.training import generate_eval

device = "cuda" if torch.cuda.is_available() else "cpu"


EVAL_PROMPTS = [
"Question: Is there any security issue if I use for example my Gmail account to store passwords from another resources "
"instead of using any password manager like a KeePass? \n\n Answer: ",

"""Question: My WordPress website received a couple of fake subscriptions to the newsletter. I identified the logs, most of them with the same form as below:
```
xx.xx.xx.xx example.com - [04/Feb/2023:06:01:42 +0100] "POST / HTTP/1.1" 200 207 "https://example.com/" "curl/7.54.0"
```
Is there any way to block this?  \n\n Answer: """,
]


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    generate_eval(model, tokenizer, EVAL_PROMPTS)
