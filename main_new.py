import os
import argparse
import warnings
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from lib.data import ImageTokenDatasetClassLabelNew
from lib.models import (
    ConditionalTransformerDecoderConfig, ConditionalTransformerDecoder,
    VanillaTransformerDecoderConfig, VanillaTransformerDecoder
)
from lib.training import (
    ConditionalTransformerTrainer,
    UnconditionalTransformerTrainer
)

device = torch.device("cuda:0")

###

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["flowers", "tinyimagenet"], required=True)
parser.add_argument("--benchmark", type=str, choices=["unc", "cond"], required=True)
parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--save_every_epoch", type=int, required=True)
parser.add_argument("--savepath", type=str, required=True)
parser.add_argument("--loadpath", type=str, required=False)

###

trainer = dict(
    unc=UnconditionalTransformerTrainer,
    cond=ConditionalTransformerTrainer
)

model = dict(
    unc=VanillaTransformerDecoder,
    cond=ConditionalTransformerDecoder
)

###

if __name__ == "__main__":
    args = parser.parse_args()

    model_benchmark_configs = yaml.safe_load(open("./model_benchmark_configs.yaml", "r"))

    dataset = ImageTokenDatasetClassLabelNew(dataset=args.dataset)

    model_config = None
    match args.benchmark:
        case "unc":
            model_config = VanillaTransformerDecoderConfig(
                **model_benchmark_configs["unc"]
            )
        case "cond":
            model_config = ConditionalTransformerDecoderConfig(
                n_classes=dataset.n_classes,
                **model_benchmark_configs["cond_l"]
            )

    model = model[args.benchmark](model_config)

    trainer = trainer[args.benchmark](
        train_dataset=dataset,
        batch_size=args.batch_size,
        model=model,
        lr=2.25e-5,
        save_every_epoch=args.save_every_epoch,
        savepath=args.savepath,
        device=device
    )

    if args.loadpath:
        trainer.load_checkpoint(args.loadpath)

    trainer.train(args.n_epochs)