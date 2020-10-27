import os
import json
import torch
import logging
import argparse

from mingpt.utils import sample
from mingpt.utils import load_json

from mingpt.model import GPT, GPTConfig

from char_dataset import BytesDataset

logging.basicConfig(
    format="%(levelname)s: %(asctime)s %(name)s | %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "model", metavar="PATH", type=str, help="model",
)

parser.add_argument(
    "--model_dir",
    type=str,
    default="models",
    help="""The model dir, if not 'models' (the default).""",
)

parser.add_argument(
    "-c",
    "--context",
    default=None,
    type=str,
    help="Context to be given to the model for sampling (beginning).",
)

args = parser.parse_args()

mconf = GPTConfig(**load_json(os.path.join(args.model_dir, args.model, "model.json")))
model = GPT(mconf)
model.load_state_dict(
    torch.load(os.path.join(args.model_dir, args.model, "model.pt"))["model_state_dict"]
)

mconf.log()
# model.log()

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

model.to(device)

stoi = load_json(os.path.join(args.model_dir, args.model, "model.vocab.json"))
bytes_level = True if len(stoi) == 1 and "bytes" in stoi else False
if not bytes_level:
    itos = {int(v): k for k, v in stoi.items()}

logger.info(f"inferring on device: {device}",)

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
