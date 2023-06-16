from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import addict
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from pynvml import *
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from transformers import logging


logging.set_verbosity_error()


print_gpu_utilization(0)
print_gpu_utilization(1)

cfg_reward = addict.Dict({
    "model_name": "bert-base-cased",
    "num_train_epochs": 1,
    "train_subset": 100_000,
    "eval_subset": 20_000,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.05,
    "learning_rate": 2e-5,
    "bf16": False,
    "fp16": False,
    ######
    "per_device_eval_batch_size": 32,
    "per_device_train_batch_size": 32,
    "local_rank": 0,
})


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer = None
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


if __name__ == '__name__':
    # Load the human stack-exchange-paired dataset for tuning the reward model
    train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
    if cfg_reward.train_subset > 0:
        train_dataset = train_dataset.select(range(cfg_reward.train_subset))

    eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
    if cfg_reward.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(cfg_reward.eval_subset))

    model_name_split = cfg_reward.model_name.split("/")[-1]
    output_name = f"{model_name_split}_peft_stack-exchange-paired_rmts_{cfg_reward.train_subset}_{cfg_reward.learning_rate}"

    original_columns = train_dataset.column_names
    num_proc = 8  # Can adjust to be higher if you have more processors

    tokenizer = AutoTokenizer.from_pretrained(cfg_reward.model_name)
    config = AutoConfig.from_pretrained(cfg_reward.model_name)

    # preprocess the dataset and filter out QAs that are longer than 512
    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)

    eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=cfg_reward.learning_rate,
        per_device_train_batch_size=cfg_reward.per_device_train_batch_size,
        per_device_eval_batch_size=cfg_reward.per_device_eval_batch_size,
        num_train_epochs=cfg_reward.num_train_epochs,
        weight_decay=cfg_reward.weight_decay,
        gradient_accumulation_steps=cfg_reward.gradient_accumulation_steps,
        gradient_checkpointing=cfg_reward.gradient_checkpointing,
        lr_scheduler_type=cfg_reward.lr_scheduler_type,
        bf16=cfg_reward.bf16,
        fp16=cfg_reward.fp16,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        local_rank=cfg_reward.local_rank,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        label_names=[],
    )

    # Load reward model as a SequenceClassification model with 1 label
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_reward.model_name,
        num_labels=1,
        device_map={'': torch.cuda.current_device()}
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    model.config.use_cache = not cfg_reward.gradient_checkpointing

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=512),
    )
    trainer.train()

    # print("Saving last checkpoint of the model")
    # model.save_pretrained(output_name + "_peft_last_checkpoint")
