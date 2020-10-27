"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import os
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import sample
from mingpt.utils import Loggable
from mingpt.utils import save_json

logger = logging.getLogger(__name__)


class TrainerConfig(Loggable):
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, opt=None):
        self.model = model
        self.dataset = (
            train_dataset.dataset
            if hasattr(train_dataset, "dataset")
            else train_dataset
        )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.tpu = False

        # take over whatever gpus are on the system
        self.device = "cpu"
        if "COLAB_TPU_ADDR" in os.environ:
            self.setup_tpu()
            self.device = xm.xla_device()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.urmodel = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer = self.urmodel.configure_optimizers(config)
        if opt is not None:
            logger.info("loading optimizer dict")
            # [logger.info(f"{k}: {v}") for k, v in opt.items()]
            self.optimizer.load_state_dict(opt)

    def setup_tpu():
        # install tpu requirements
        tpu_client_version = "cloud-tpu-client==0.10"
        wheel = "https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl"
        logger.info(f"Found tpu: {os.environ['COLAB_TPU_ADDR']}. Attempting to install requirements:")
        logger.info(f"{tpu_client_version} | {wheel}")
        logger.info(f"(these can be changed in mingpt/trainer.py)")
        # https://stackoverflow.com/a/50255019
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", tpu_client_version, wheel])
        # import xla
        import torch_xla
        import torch_xla.core.xla_model as xm
        self.tpu = True


    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        logger.info(f"saving model {os.path.split(self.config.ckpt_path)[-1]}")
        if not os.path.isdir(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        self.urmodel.save(*os.path.split(self.config.ckpt_path))
        save_json(
            self.dataset.stoi, os.path.join(self.config.ckpt_path, "model.vocab.json")
        )
        save_json(
            self.config.as_dict(),
            os.path.join(self.config.ckpt_path, "model.train.json"),
        )
        save_fn = xm if self.tpu else torch
        save_fn.save(
            {
                "model_state_dict": self.urmodel.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.config.ckpt_path, "model.pt",),
        )

    def train(self, sample_every=0):
        model, config = self.model, self.config

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:

                # get global step from optimizer
                # https://discuss.pytorch.org/t/current-step-from-optimizer/19370/3
                try:
                    step = self.optimizer.state[
                        self.optimizer.param_groups[0]["params"][-1]
                    ]["step"]
                except:
                    step = 0

                if sample_every > 0 and step != 0 and step % sample_every == 0:
                    context = "\n"

                    inp = torch.tensor(
                        self.train_dataset.encode(context), dtype=torch.long
                    )[None, ...].to(self.device)

                    outp = sample(
                        self.urmodel, inp, 200, temperature=1.0, sample=True, top_k=10
                    )[0]

                    completion = self.train_dataset.decode(outp, errors="replace")

                    print("-" * 40)
                    print(completion)
                    print("-" * 40)

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    # collapse all losses if they are scattered on multiple gpus
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    self.optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} step {step}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float("inf")
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
