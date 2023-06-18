import textwrap

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def print_gpu_utilization(devId=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(devId)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()