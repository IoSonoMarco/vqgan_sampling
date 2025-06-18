import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TransformerDecoderOutput:
    hidden_states: torch.tensor
    logits: torch.tensor
    loss: torch.tensor


class BaseTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config       


    def _get_number_of_parameters(self):
        print(f"#params: {sum(p.numel() for p in self.parameters())}")


    @torch.no_grad()
    def sample_topk(self, logits, k: int = None):
        if k is not None:
            topk_vals, topk_idx = torch.topk(logits, k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(0, topk_idx, topk_vals)
        else:
            mask = logits
        probs = torch.softmax(mask, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze()
        return int(next_token)


    @torch.no_grad()
    def predict_next_token(self, inputs, k: int = None):
        """
        inputs: long tensor of shape (n_tokens,) 
        where n_tokens is the number previous tokens
        """
        inputs = inputs[None]
        inputs = self.embedding(inputs)
        n = inputs.size(1)
        inputs += self.positional_encoding[:n]
        inputs = self.encoder(inputs, mask=self.causal_mask[:n,:n])[0]
        logits = self.prediction_head(inputs[-1][None])[0]
        next_token = self.sample_topk(logits, k)
        return next_token