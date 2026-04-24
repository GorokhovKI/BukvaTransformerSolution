from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from constants import (
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_LAYERS,
    SEQUENCE_LENGTH,
)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the sign language model."""

    input_size: int = INPUT_SIZE
    sequence_length: int = SEQUENCE_LENGTH
    hidden_size: int = HIDDEN_SIZE
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    num_classes: int = NUM_CLASSES
    ff_multiplier: int = 2


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic positional encoding with zero runtime allocations."""

    def __init__(self, hidden_size: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / hidden_size)
        )

        pe = torch.zeros(max_len, hidden_size, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("encoding", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer encoder block with explicit residual paths."""

    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        return x + self.dropout2(ff_output)


class SignLanguageTransformer(nn.Module):
    """Compact transformer classifier optimized for edge/mobile deployment."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.input_projection = nn.Linear(self.config.input_size, self.config.hidden_size)
        self.position = SinusoidalPositionalEncoding(
            hidden_size=self.config.hidden_size,
            max_len=self.config.sequence_length + 32,
        )

        ff_size = self.config.hidden_size * self.config.ff_multiplier
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    ff_size=ff_size,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.num_classes),
        )

    @staticmethod
    def masked_mean_pool(sequence: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        if padding_mask is None:
            return sequence.mean(dim=1)

        valid = (~padding_mask).to(dtype=sequence.dtype).unsqueeze(-1)
        summed = (sequence * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.input_projection(x)
        x = self.position(x)

        for block in self.encoder:
            x = block(x, padding_mask=src_key_padding_mask)

        pooled = self.masked_mean_pool(x, src_key_padding_mask)
        return self.head(pooled)
