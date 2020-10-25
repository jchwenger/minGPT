import os
import json
import random
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def check_name(name):
    if name.endswith(".pt"):
        model_name = os.path.splitext(name)[0]
        model_path = name
    else:
        model_name = name
        model_path = model_name + ".pt"
    return model_path, model_name


def check_file(fname):
    if os.path.isfile(fname):
        return True
    else:
        logger.info(f"could not find: {fname}")
        return False


def check_model(name):
    found = True
    if not all(
        [
            check_file(name + ".pt"),
            check_file(name + ".vocab.json"),
            check_file(name + ".json"),
            check_file(name + ".train.json"),
        ]
    ):
        found = False
    return found


def pretty_log_dict(le_dict, title=None):
    logger.info("-" * 40)
    if title is not None:
        logger.info(title)
        logger.info("-" * len(title))
    longest = len(max(le_dict.keys(), key=len))
    [logger.info(f"{k:{longest}}: {v}") for k, v in le_dict.items()]
    logger.info("-" * 40)
    logger.info("")


def print_state_dict(model):
    logger.info("-" * 40)
    msg = "Model's state_dict:"
    logger.info(msg)
    logger.info("-" * len(msg))
    longest = len(max(model.state_dict().keys(), key=len))
    for param_tensor in model.state_dict():
        logger.info(
            f"{param_tensor:{longest}} {list(model.state_dict()[param_tensor].size())}"
        )
    logger.info("-" * 40)
    logger.info("")


def load_json(fname):
    with open(fname) as i:
        j = json.load(i)
    return j


def save_json(obj, fname):
    with open(fname, "w") as o:
        json.dump(obj, o)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
