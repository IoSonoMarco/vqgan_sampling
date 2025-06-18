import torch
import math


def get_sinusoidal_positional_encoding(d_model: int, max_len: int = 512):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) *
        -(math.log(10000.0) / d_model)
    )  
    pe[:, 0::2] = torch.sin(position * div_term)  # even
    pe[:, 1::2] = torch.cos(position * div_term)  # odd
    return pe


def build_prefix_causal_mask(N: int, S: int):
    """
    N: number of pre-pended tokens for conditioning
    S: number of tokens in the actual sequence
    """
    L = N + S
    mask = torch.full((L, L), float('-inf'))

    # prompt ↔ prompt (allow)
    mask[:N, :N] = 0

    # sequence ↔ prompt (allow)
    mask[N:, :N] = 0           

    # sequence ↔ sequence (causal)
    causal = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
    mask[N:, N:] = causal
    return mask