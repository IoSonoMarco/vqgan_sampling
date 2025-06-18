from dataclasses import dataclass
import torch
import torch.nn as nn
from lib.models.base import BaseTransformerDecoder, TransformerDecoderOutput
from lib.models.utils import get_sinusoidal_positional_encoding


@dataclass
class VanillaTransformerDecoderConfig:
    d_model: int 
    n_layers: int 
    n_heads: int 
    n_tokens: int = 256
    vocab_size: int = 1024


class VanillaTransformerDecoder(BaseTransformerDecoder):
    def __init__(self, config: VanillaTransformerDecoderConfig):
        super().__init__(config)

        self.register_buffer(
            "positional_encoding", 
            get_sinusoidal_positional_encoding(
                self.config.d_model, 
                self.config.n_tokens + 1
            )
        )

        self.register_buffer(
            "causal_mask", 
            nn.Transformer.generate_square_subsequent_mask(self.config.n_tokens + 1)
        )

        self.register_buffer(
            "sos_token_id", 
            torch.tensor(self.config.vocab_size)
        )

        self.embedding = nn.Embedding(
            self.config.vocab_size + 1, 
            self.config.d_model
        )

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model*4,
            batch_first=True
        ), num_layers=self.config.n_layers)

        self.prediction_head = nn.Linear(self.config.d_model, self.config.vocab_size)

        self._get_number_of_parameters()

    def embed(self, x):
        """
        x: long tensor of shape (batch_size, n_tokens)
        """
        x = self.embedding(x)
        x += self.positional_encoding
        return x

    def encode(self, x):
        """
        x: long tensor of shape (batch_size, n_tokens, d_model)
        """
        x = self.encoder(x, mask=self.causal_mask)
        return x

    def forward(self, x):
        """
        x: long tensor of shape (batch_size, n_tokens)
        """
        
        # prepend sos token
        x = torch.cat([self.sos_token_id.repeat(x.size(0),1), x], dim=-1)
        
        # embed + encode
        hidden_states = self.embed(x)
        hidden_states = self.encode(hidden_states)
        
        # predict next token + loss
        logits = self.prediction_head(hidden_states[:, :-1, :])

        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            x[:, 1:].contiguous().view(-1)
        )

        return TransformerDecoderOutput(
            hidden_states=hidden_states,
            logits=logits,
            loss=loss
        )