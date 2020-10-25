
import os
import json
import torch
import logging
import argparse

from mingpt.utils import sample
from mingpt.utils import load_json
from mingpt.utils import check_name
from mingpt.utils import print_state_dict

from mingpt.model import GPT, GPTConfig

from char_dataset import BytesDataset

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "model",
    metavar="PATH",
    type=str,
    help="model",
)

parser.add_argument(
    "-c", "--context",
    default=None,
    type=str,
    help="Context to be given to the model for sampling (beginning).",
)

args = parser.parse_args()

args.model, model_name = check_name(args.model)

model = GPT(GPTConfig(**load_json(f"{model_name}.json")))

model.load_state_dict(torch.load(args.model)["model_state_dict"])

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

model.to(device)

stoi = load_json(model_name + ".vocab.json")
bytes_level = True if len(stoi) == 1 and "bytes" in stoi else False
if not bytes_level:
    itos = {int(v): k for k, v in stoi.items()}

print_state_dict(model)

logger.info("inferring on device:", device)

context = "o god, o god!" if args.context is None else args.context

if bytes_level:
    context = BytesDataset.encode(context)
else:
    context = [stoi[s] for s in context]
x = torch.tensor(context, dtype=torch.long)[None, ...].to(device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]

if bytes_level:
    completion = BytesDataset.decode(y)
else:
    completion = "".join([itos[int(i)] for i in y])

print(completion)
