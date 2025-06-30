from dataclasses import dataclass
import torch
import torch.nn as nn
from lib.models.base import BaseTransformerDecoder, TransformerDecoderOutput
from lib.models.utils import get_sinusoidal_positional_encoding, build_prefix_causal_mask


@dataclass
class ConditionalTransformerDecoderConfig:
    d_model: int 
    n_layers: int 
    n_heads: int 
    class_prompt_length: int 
    n_classes: int
    n_tokens: int = 256
    vocab_size: int = 1024


class ConditionalTransformerDecoder(BaseTransformerDecoder):
    def __init__(self, config: ConditionalTransformerDecoderConfig):
        super().__init__(config)

        self.register_buffer(
            "positional_encoding", 
            get_sinusoidal_positional_encoding(
                self.config.d_model, 
                self.config.n_tokens
            )
        )

        self.register_buffer(
            "causal_mask", 
            build_prefix_causal_mask(
                self.config.class_prompt_length,
                self.config.n_tokens
            )
        )

        self.embedding = nn.Embedding(
            self.config.vocab_size, 
            self.config.d_model
        )

        self.class_prompt_embedding = nn.Embedding(
            self.config.n_classes,
            int(self.config.d_model * self.config.class_prompt_length)
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
    
    def forward(self, x, y):
        """
        x: long tensor of shape (batch_size, n_tokens)
        y: long tensor of class conditioning labels of shape (batch_size, )
        """

        # embed class conditioning
        prompts = self.class_prompt_embedding(y).view(x.size(0), -1, self.config.d_model) # (batch_size, prompt_length, d_model)
        
        # embed input sequence
        hidden_states = self.embed(x)
        
        # prepend class condition prompts + encoder
        hidden_states = torch.cat([prompts, hidden_states], dim=1)
        hidden_states = self.encode(hidden_states)
        
        # predict next token + loss
        logits = self.prediction_head(
            hidden_states[:, (self.config.class_prompt_length-1):-1, :]
        )

        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            x.contiguous().view(-1)
        )

        return TransformerDecoderOutput(
            hidden_states=hidden_states,
            logits=logits,
            loss=loss
        )
    
    @torch.no_grad()
    def predict_next_token(self, prompt, inputs, k: int = None):
        """
        prompt: tensor of shape (prompt_length, d_model)
        inputs: long tensor of shape (n_tokens,) 
        where n_tokens is the number previous tokens
        """
        if len(inputs) == 0:
            inputs = prompt[None]
            n = inputs.size(1)
        else:
            inputs = self.embedding(inputs[None])
            n = inputs.size(1)
            inputs += self.positional_encoding[:n]
            inputs = torch.cat([prompt[None], inputs], dim=1)
            n = inputs.size(1)
        inputs = self.encoder(inputs, mask=self.causal_mask[:n,:n])[0]
        logits = self.prediction_head(inputs[-1][None])[0]
        next_token = self.sample_topk(logits, k)
        return next_token

