from typing import Tuple, Dict
import torch
from lib.training.base import BaseTrainer


class UnconditionalTransformerTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, batch: torch.LongTensor) -> Dict:
        batch = batch.to(self.device)

        output = self.model(batch)
        
        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()

        return dict(
            loss=float(output.loss)
        )
    

class ConditionalTransformerTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, batch: Tuple[torch.LongTensor]) -> Dict:
        tokens, labels = batch
        tokens = tokens.to(self.device)
        labels = labels.to(self.device)

        output = self.model(tokens, labels)
        
        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()

        return dict(
            loss=float(output.loss)
        )