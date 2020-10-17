#!/usr/bin/env python
# coding: utf-8

# ## Train a character-level GPT on some text data
#
# The inputs here are simple text files, which we chop up to individual characters and then train GPT on. So you could say this is a char-transformer instead of a char-rnn. Doesn't quite roll off the tongue as well. In this example we will feed it some Shakespeare, which we'll get it to predict character-level.

import os
import json
import logging
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(
    description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "input",
    metavar="PATH",
    type=str,
    help="input file, all in one",
)
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed
set_seed(42)

from torch.nn import functional as F
from char_dataset import CharDataset
import torch.nn as nn
import numpy as np
import torch
import math


block_size = 64  # spatial extent of the model for its context

text = open(args.input, "r").read()  # don't worry we won't run out of file handles

train_dataset = CharDataset(
    text, block_size
)

from mingpt.model import GPT, GPTConfig

mconf = GPTConfig(
    train_dataset.vocab_size, train_dataset.block_size, n_layer=2, n_head=2, n_embd=256
)
model = GPT(mconf)

# load model
mod_name = "le_model"
logger.info(f"found model: {mod_name}")
if os.path.isfile(mod_name):
    model.load_state_dict(torch.load(mod_name))

from mingpt.trainer import Trainer, TrainerConfig

bs = 256
# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=1,
    batch_size=bs,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=bs * 20,
    final_tokens=2 * len(train_dataset) * block_size,
    num_workers=4,
)
trainer = Trainer(model, train_dataset, None, tconf)

# Print model's state_dict
logger.info("Model's state_dict:")
for param_tensor in model.state_dict():
    logger.info(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")

try:
    logger.info("training")
    trainer.train()
except KeyboardInterrupt:
    logger.error("interrupted, saving...")
    mconf.save(mod_name)
    train_dataset.save(mod_name)
    torch.save(model.state_dict(), mod_name + ".pt")
