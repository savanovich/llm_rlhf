import os
import re

import addict
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

import augmenty
import spacy

from utils.markdown import find_non_text_intervals, apply_transformation_for_non_code

nlp = spacy.load("en_core_web_md")
keystroke_error_augmenter = augmenty.load("keystroke_error_v1", level=0.03)

device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = addict.Dict(
    {
        "dataset_name": "lvwerra/stack-exchange-paired",
        "max_steps": 4000,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "cosine",
        "fp16": True,
        "gradient_checkpointing": True,
        "weight_decay": 0.05,
        "num_warmup_steps": 100,
        ##########
        "subset": "data/finetune",
        "split": "train",
        "size_valid_set": 4000,
        "shuffle_buffer": 5000,
        "seq_length": 1024,
        "local_rank": 0,
        "seed": 0,
        "num_workers": None,
        "output_dir": "./checkpoints",
        "log_freq": 1,
        "eval_freq": 1000,
        "save_freq": 1000,
    }
)


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }


def augment(text):
    return apply_transformation_for_non_code(text, functions=[
        lambda s: list(augmenty.texts([s, ], augmenter=keystroke_error_augmenter, nlp=nlp))[0]
    ])


# TODO: check not cached
def prepare_sample_text(example):
    question = augment(example['question'])
    response = augment(example['response_j'])

    text = f"Question: {question}\n\nAnswer: {response}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def create_datasets(tokenizer, cfg):
    dataset = load_dataset(
        cfg.dataset_name,
        data_dir=cfg.subset,
        split=cfg.split,
        use_auth_token=True,
        num_proc=None,
    )
    dataset = dataset.train_test_split(test_size=0.005, seed=cfg.seed)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=cfg.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=cfg.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


if __name__ == '__main__':
    # load model in 8bit
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        load_in_8bit=True,
        device_map={'': torch.cuda.current_device()}
        # device_map="auto"
    )

    model = prepare_model_for_int8_training(model)

    # add LoRA to model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config).to(device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    print("Starting main loop")
    train_dataset, eval_dataset = create_datasets(tokenizer, cfg)
    train_dataset.start_iteration = 0

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=cfg.max_steps,
        eval_steps=cfg.eval_freq,
        save_steps=cfg.save_freq,
        logging_steps=cfg.log_freq,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.num_warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        weight_decay=cfg.weight_decay,
    )

    print("Training...")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(cfg.output_dir, "final_checkpoint/"))



