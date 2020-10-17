import os
import json
import torch
import argparse
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig

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

assert os.path.isfile(args.model), f"{args.model} not found"
mod_config = os.path.splitext(args.model)[0] + ".json"
assert os.path.isfile(mod_config), f"{mod_config} not found"
stoi_config = os.path.splitext(args.model)[0] + ".stoi.json"
assert os.path.isfile(mod_config), f"{stoi_config} not found"
itos_config = os.path.splitext(args.model)[0] + ".itos.json"
assert os.path.isfile(mod_config), f"{itos_config} not found"

with open(mod_config) as i:
    data = json.load(i)

mconf = GPTConfig(
    data["vocab_size"],
    data["block_size"],
    n_layer=data["n_layer"],
    n_head=data["n_head"],
    n_embd=data["n_embd"],
)
model = GPT(mconf)
model.load_state_dict(torch.load(args.model))

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

model.to(device)

with open(stoi_config) as i:
    stoi = json.load(i)

with open(itos_config) as i:
    itos = json.load(i)
    itos = {int(k): v for k, v in itos.items()}

# Print model's state_dict
print("-" * 40)
msg = "Model's state_dict:"
print(msg)
print("-" * len(msg))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("-" * 40)

print("inferring on device:", device)

context = "O God, O God!" if args.context is not None else args.context

x = torch.tensor([stoi[s] for s in context], dtype=torch.long)[
    None, ...
].to(device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = "".join([itos[int(i)] for i in y])
print(completion)
