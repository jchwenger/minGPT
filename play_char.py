#!/usr/bin/env python
# coding: utf-8

# Train a character-level GPT on some text data
# The inputs here are simple text files, which we chop up to individual characters and then train GPT on. So you could say this is a char-transformer instead of a char-rnn. Doesn't quite roll off the tongue as well. In this example we will feed it some Shakespeare, which we'll get it to predict character-level.

import os
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.trainer import Trainer
from mingpt.trainer import TrainerConfig

from mingpt.model import GPT
from mingpt.model import GPTConfig
from mingpt.model import MinConfig

from char_dataset import BytesDataset
from char_dataset import CharDataset

from mingpt.utils import set_seed
from mingpt.utils import load_json
from mingpt.utils import check_name
from mingpt.utils import check_model
from mingpt.utils import pretty_log_dict
from mingpt.utils import print_state_dict

# make deterministic
set_seed(42)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "input", metavar="PATH", type=str, help="input file, all in one",
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="""the path to the model (the dir should include the model.vocab.json & model.json.""",
)

parser.add_argument(
    "--byte_level",
    action="store_true",
    help="""train on bytes instead of chars.
    Defaults to false """,
)

parser.add_argument(
    "--block_size", default=64, type=int, help="the attention window, default: 64",
)

parser.add_argument(
    "--n_layer",
    type=int,
    help="the number of layers. If None, will default to tiny config: 4",
)

parser.add_argument(
    "--n_head",
    type=int,
    help="the number of heads. If None, will default to tiny config:: 4",
)

parser.add_argument(
    "--n_embd",
    type=int,
    help="the embedding dims. If None, will default to tiny config:: 128",
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="""the batch size. If resuming training, the
    last batch sized will be used unless a new one is provided. When creating a
    model, the default is 10.""",
)

parser.add_argument(
    "--sample_every",
    default=0,
    type=int,
    help="print a sample every N steps, default: 0 (disabled)",
)

parser.add_argument(
    "--train_test_split",
    "--test_train_split",
    default=100,
    type=int,
    help="""The divisor used to split the dataset. Defaults to 100, which
    means a test set randomly sampled to amount to a 100th the size of the
    total dataset: `test_len = dataset_len // divisor, train = remainder`""",
)

args = parser.parse_args()

logging.basicConfig(
    format="%(levelname)s: %(asctime)s %(name)s | %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def make_dataset(fname):
    with open(fname, "r") as i:
        text = i.read()
    # block_size: spatial extent of the model for its context
    if args.byte_level == True:
        return BytesDataset(text, args.block_size)
    else:
        return CharDataset(text, args.block_size)


def split_dataset(dataset, divisor=100, seed=42):
    ds_len = len(dataset)
    test_len = ds_len // divisor
    test, train = torch.utils.data.random_split(
        dataset,
        [test_len, ds_len - test_len],
        generator=torch.Generator().manual_seed(seed),
    )
    return {
        "test_dataset": test,
        "train_dataset": train,
    }


def restore_model(model_name):
    mconf = GPTConfig()
    mconf.load(model_name)
    model = GPT(mconf)
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt["model_state_dict"])
    tconf = TrainerConfig(**load_json(f"{model_name }.train.json"))
    if args.batch_size is not None:
        tconf.batch_size = args.batch_size
    pretty_log_dict(tconf.as_dict(), title="Training config:")
    print_state_dict(model)
    return {
        "model": model,
        "config": tconf,
        "opt": ckpt["optimizer_state_dict"] if "optimizer_state_dict" in ckpt else None,
    }


def create_model(model_name, dataset):
    # default to a tiny network if nothing specified
    mconf = MinConfig()

    mconf.block_size = args.block_size

    if args.n_layer is not None:
        mconf.n_layer = args.n_layer
    if args.n_head is not None:
        mconf.n_head = args.n_head
    if args.n_embd is not None:
        mconf.n_embd = args.n_embd

    mconf.vocab_size = dataset.vocab_size

    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        max_epochs=1,
        batch_size=args.batch_size if args.batch_size is not None else 10,
        learning_rate=6e-4,
        lr_decay=True,
        ckpt_path=model_name,
        warmup_tokens=args.batch_size * 20,
        final_tokens=2 * len(dataset) * args.block_size,
        num_workers=4,
    )

    pretty_log_dict(tconf.as_dict(), title="Training config:")
    print_state_dict(model)

    return {
        "model": model,
        "config": tconf,
        "opt": None,
    }


if args.model is not None:
    # name given
    args.model, model_name = check_name(args.model)
    # continue training
    if check_model(model_name):
        logger.info(f"found model: {args.model}, restoring.")
        trainer = Trainer(
            **{
                **split_dataset(
                    make_dataset(args.input), divisor=args.train_test_split
                ),
                **restore_model(model_name),
            }
        )
    # new model with said name
    else:
        logger.info(
            f"model: {args.model} not found, creating a new one with this name."
        )

        train_dataset = make_dataset(args.input)
        trainer = Trainer(
            **{
                **split_dataset(train_dataset, divisor=args.train_test_split),
                **create_model(args.model, train_dataset),
            }
        )
# default new model
else:
    args.model = "le_model.pt"
    logging.info("no model specified, training 'le_model'")
    train_dataset = make_dataset(args.input)
    trainer = Trainer(
        **{
            **split_dataset(train_dataset, divisor=args.train_test_split),
            **create_model("le_model", train_dataset),
        }
    )

try:
    logger.info("training")
    trainer.train(sample_every=args.sample_every)
except KeyboardInterrupt:
    logger.info("saving model, vocab, optimizer")
    trainer.save_checkpoint()
