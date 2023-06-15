from dataclasses import dataclass, field
from typing import Optional

import torch
import addict
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    AutoModelForSequenceClassification
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


ppo_config = addict.Dict({
    "model_name": "djalexj/gpt-neo-1.3B-sft-se-4000steps",
    "tokenizer_name": "EleutherAI/gpt-neo-1.3B",
    "reward_model_name": "djalexj/bert-base-cased-rm-se-100000steps",
    "reward_tokenizer_name": "bert-base-cased",
    ######
    "learning_rate": 1.4e-5,
    "output_max_length": 128,
    "mini_batch_size": 2,
    "batch_size": 8,
    "ppo_epochs": 3,
    "gradient_accumulation_steps": 4,
    "adafactor": False,
    "early_stopping": True,
    "target_kl": 0.1,
    "reward_baseline": 0.0,
    #######
    "batched_gen": True,
    "save_freq": 50,
    "output_dir": "gpt-neo-1.3B-rlhf-se",
    "seed": 0,
    "log_with": None,

})



tqdm.pandas()

reward_model_name = ppo_config.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"

rl_config = PPOConfig(
    model_name=ppo_config.model_name,
    learning_rate=ppo_config.learning_rate,
    log_with=ppo_config.log_with,
    batch_size=ppo_config.batch_size,
    mini_batch_size=ppo_config.mini_batch_size,
    gradient_accumulation_steps=ppo_config.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=ppo_config.early_stopping,
    target_kl=ppo_config.target_kl,
    ppo_epochs=ppo_config.ppo_epochs,
    seed=ppo_config.seed,
)


train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
train_dataset = train_dataset.select(range(100000))
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16, "truncation": True}

reward_tokenizer = AutoTokenizer.from_pretrained(ppo_config.reward_tokenizer_name)
tokenizer = AutoTokenizer.from_pretrained(ppo_config.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer, dataset_name="lvwerra/stack-exchange-paired", input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.
    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 2

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(rl_config.seed)



# Now let's build the model, the reference model, and the tokenizer.
pretrained_model = AutoModelForCausalLM.from_pretrained(
    rl_config.model_name,
    load_in_8bit=True,
    device_map={"": torch.cuda.current_device()}
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = prepare_model_for_int8_training(pretrained_model)
pretrained_model.enable_input_require_grads()
pretrained_model = get_peft_model(pretrained_model, lora_config)

pretrained_model.print_trainable_parameters()

# Use trl wrapper for additional ValueHead (critic)
model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

optimizer = None
if ppo_config.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=rl_config.learning_rate,
    )


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config=rl_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug


reward_model = AutoModelForSequenceClassification.from_pretrained(
    ppo_config.reward_model_name
)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device_map={"": torch.cuda.current_device()},
    tokenizer=reward_tokenizer,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 32
output_max_length = ppo_config.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    print(batch["response"])

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - ppo_config.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if ppo_config.save_freq and epoch and epoch % ppo_config.save_freq == 0:
        ppo_trainer.save_pretrained(ppo_config.output_dir + f"step_{epoch}")
    break


