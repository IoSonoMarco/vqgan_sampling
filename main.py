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
from lib.data import (
    ImageTokenDataset, 
    ImageTokenDatasetClassLabel,
    ImageTokenDatasetSemanticLabel
)
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
parser.add_argument("--benchmark", type=str, choices=["unc", "cond_l", "cond_s"], required=True)
parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--save_every_epoch", type=int, required=True)
parser.add_argument("--savepath", type=str, required=True)
parser.add_argument("--loadpath", type=str, required=False)

###

dataset = dict(
    unc=ImageTokenDataset,
    cond_l=ImageTokenDatasetClassLabel,
    cond_s=ImageTokenDatasetSemanticLabel
)

trainer = dict(
    unc=UnconditionalTransformerTrainer,
    cond_l=ConditionalTransformerTrainer,
    cond_s=ConditionalTransformerTrainer
)

model = dict(
    unc=VanillaTransformerDecoder,
    cond_l=ConditionalTransformerDecoder,
    cond_s=ConditionalTransformerDecoder
)

if __name__ == "__main__":
    args = parser.parse_args()

    model_benchmark_configs = yaml.safe_load(open("./model_benchmark_configs.yaml", "r"))

    dataset = dataset[args.benchmark]()

    model_config = None
    match args.benchmark:
        case "unc":
            model_config = VanillaTransformerDecoderConfig(
                **model_benchmark_configs["unc"]
            )
        case "cond_l":
            model_config = ConditionalTransformerDecoderConfig(
                n_classes=dataset.n_classes,
                **model_benchmark_configs["cond_l"]
            )
        case "cond_s":
            model_config = ConditionalTransformerDecoderConfig(
                n_classes=dataset.n_classes,
                **model_benchmark_configs["cond_s"]
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




