#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29601"  # Change to any unused port

import sys
from typing import Optional
from functools import partial
import datasets
import torch
import torch.distributed as dist
import deepspeed
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, List, Union
import pandas as pd
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from utils.neft import NEFTune

from trainer import SFTTrainer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
    loss: str = field(
        default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "gem_triton","ts2", "neft"]}
    )
    gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM. A value between 0 and 1. A value close to 1.0 makes GEM behave more like CE, while a value close to 0.0 preserves more diversity."})
    gem_h: str = field(
        default="linear", metadata={"help": "Function $h$ in GEM. The 'logsigmoid' function is more adaptive, but the difference between 'logsigmoid' and 'linear' is usually negligible.", "choices": ["logsigmoid", "linear"]}
    )
    ts2_alpha: float = field(default=10.0)
    ts2_temperature: float = field(default=1.0)
    print_entropy: bool = field(
        default=False, metadata={"help": "Print entropy during training"}
    )
    max_grad_norm: float = field(
        default=1.0
    )
    warmup_epochs:int=field(
        default=0
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


@dataclass
class DataArguments:
    train_tokenized_file: str = field(
        default=None, metadata={"help": "huggingface dataset name or local data path"}
    )
    test_tokenized_file: str = field(
        default=None, metadata={"help": "huggingface dataset name or local data path"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


class CustomDataset(Dataset):
    def __init__(
        self,
        training_args,
        data_args,
        model_args,
        train_tokenized_file,
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

        raw_datasets = load_dataset(
            "json",
            data_files=[train_tokenized_file],
            cache_dir=self.model_args.cache_dir,
        )
        self.data = raw_datasets["train"]

        if self.data_args.max_train_samples is not None:
            max_samples = min(len(self.data), self.data_args.max_train_samples)
            self.data = self.data.select(range(max_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        assert "input_ids" in example
        assert "labels" in example
        example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
        return example


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    global_rank = dist.get_rank()
    logger.warning(
        f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
        tokenizer.pad_token_id = len(tokenizer) - 1
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation=(
            "flash_attention_2" if model_args.use_flash_attn else "eager"
        ),
    )
    if training_args.loss=='neft':
        model = NEFTune(model,5.0)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # set up datasets
    train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
    if data_args.test_tokenized_file:
        test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
    else:
        test_dataset = None

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
        model.generation_config.eos_token_id = [128001, 128009]
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
