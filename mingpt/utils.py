import os
import json
import random
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class Loggable:

    def __init__(self):
        pass

    # https://stackoverflow.com/questions/61517/python-dictionary-from-an-objects-fields
    # https://stackoverflow.com/a/21945171
    def as_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if attr[:2] + attr[-2:] != "____" and not callable(getattr(self, attr))
        }

    def log(self, title=None):
        pretty_log_dict(
            {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr[:2] + attr[-2:] != "____" and not callable(getattr(self, attr))
            },
            title=f"{self.__class__.__name__}:",
        )


def check_file(fname):
    if os.path.isfile(fname):
        return True
    else:
        logger.warning(f"could not find: {fname}")
        return False


def check_model(model_dir, name):
    found = True
    if not all(
        [
            check_file(os.path.join(model_dir, name, "model.pt")),
            check_file(os.path.join(model_dir, name, "model.vocab.json")),
            check_file(os.path.join(model_dir, name, "model.json")),
            check_file(os.path.join(model_dir, name, "model.train.json")),
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
