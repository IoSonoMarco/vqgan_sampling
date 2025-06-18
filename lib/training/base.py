import os
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from dataclasses import asdict


class BaseTrainer(ABC):
    def __init__(self,
        train_dataset: Dataset,
        batch_size: int,
        model: nn.Module,
        lr: float,
        save_every_epoch: int,
        savepath: str,
        device: torch.device
    ):
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.model = model
        self.lr = lr
        self.save_every_epoch = save_every_epoch
        self.savepath = savepath
        self.device = device

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.loss_history = None
        self.current_epoch = 0

    @abstractmethod
    def step(self, batch: torch.Tensor):
        pass

    def train(self, n_epochs: int):

        if self.loss_history is None:
            self.loss_history = dict(epoch_losses=[], batch_losses=[])

        for epoch in range(self.current_epoch, n_epochs):
            self.model.train()
            epoch_loss = []

            with tqdm(self.train_dataloader) as t_epoch:
                t_epoch.set_description(f"Train Epoch: {epoch + 1}")

                for batch in t_epoch:
                    batch_stats = self.step(batch)
                    epoch_loss.append(batch_stats["loss"])

                    t_epoch.set_postfix(dict(
                        loss=float(np.mean(epoch_loss).round(3))
                    ))

                    self.loss_history["batch_losses"].append(float(batch_stats["loss"]))
            self.loss_history["epoch_losses"].append(float(np.mean(epoch_loss)))

            if (epoch + 1) % self.save_every_epoch == 0:
                self.save_checkpoint(filename=f"ckpt_e{(epoch+1):03d}.pt", data=dict(
                    model_state_dict=self.model.state_dict(),
                    model_config=asdict(self.model.config),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    loss_history=self.loss_history,
                    current_epoch=epoch+1
                ))

    def save_checkpoint(self, filename: str, data: Dict[str, Any]):
        os.makedirs(self.savepath, exist_ok=True)
        torch.save(data, os.path.join(self.savepath, filename))

    def load_checkpoint(self, filepath):
        ckpt = torch.load(filepath, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.loss_history = ckpt["loss_history"]
        self.current_epoch = ckpt["current_epoch"]
        
        print(f"...checkpoint [{filepath}] loaded!")