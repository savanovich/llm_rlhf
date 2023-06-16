from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.pretraining_eval import EVAL_PROMPTS
from utils.utils_eval import generate_eval

# Load the Lora model
model = AutoModelForCausalLM.from_pretrained(f"models/sft-4000steps/snapshots/beae5adc278664ba0b94c5f2e68dbe24bb34c685", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

generate_eval(model, tokenizer, EVAL_PROMPTS)