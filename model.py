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
    input_size: int = INPUT_SIZE
    sequence_length: int = SEQUENCE_LENGTH
    hidden_size: int = HIDDEN_SIZE
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    num_classes: int = NUM_CLASSES
    ff_multiplier: int = 4
    temporal_kernel_size: int = 5


class SinusoidalPositionalEncoding(nn.Module):
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


class TemporalStem(nn.Module):
    """Adds motion features + local temporal mixing before transformer blocks."""

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size * 2, hidden_size)
        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.pointwise_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        x = torch.cat([x, delta], dim=-1)

        x = self.input_projection(x)
        conv = x.transpose(1, 2)
        conv = self.depthwise_conv(conv)
        conv = self.pointwise_conv(conv).transpose(1, 2)

        x = self.norm(x + conv)
        x = self.activation(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
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
            nn.GELU(),
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
    """Transformer with temporal stem + CLS pooling for higher gesture accuracy."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.stem = TemporalStem(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            kernel_size=self.config.temporal_kernel_size,
            dropout=self.config.dropout,
        )
        self.position = SinusoidalPositionalEncoding(
            hidden_size=self.config.hidden_size,
            max_len=self.config.sequence_length + 64,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))

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
            nn.Linear(self.config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.num_classes),
        )

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.stem(x)
        x = self.position(x)

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        for block in self.encoder:
            x = block(x, padding_mask=src_key_padding_mask)

        pooled = x[:, 0]
        return self.head(pooled)
