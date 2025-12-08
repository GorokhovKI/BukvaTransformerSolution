import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from constants import INPUT_SIZE, NUM_CLASSES, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT, SEQUENCE_LENGTH


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


class SignLanguageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.pos_encoder = PositionalEncoding(HIDDEN_SIZE, max_len=SEQUENCE_LENGTH + 50)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_SIZE * 2,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )


    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask_float = (~src_key_padding_mask).float().unsqueeze(-1)
            output = output * mask_float
            output = output.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        else:
            output = output.mean(dim=1)

        return self.fc(output)