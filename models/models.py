# Generic imports
import gc
import math
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE), optionally learnable.

    Applies rotary position encoding to even/odd pairs in Q/K vectors,
    for use in attention modules.

    Args:
        dim (int): Embedding dimension (must be even).
        max_seq_len (int): Max sequence length for precomputed rotation matrix.
        learnable (bool): Whether to learn the sin/cos parameters.
    """

    def __init__(self, dim, max_seq_len=2048, learnable=False):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE."
        self.dim = dim
        self.learnable = learnable

        # Compute fixed inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]
        positions = torch.arange(0, max_seq_len).float()  # [seq_len]
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, dim/2]

        # sin and cos buffers/parameters
        sin = torch.sin(freqs)  # [seq_len, dim/2]
        cos = torch.cos(freqs)  # [seq_len, dim/2]

        if learnable:
            self.sin = nn.Parameter(sin)
            self.cos = nn.Parameter(cos)
        else:
            self.register_buffer("sin", sin, persistent=False)
            self.register_buffer("cos", cos, persistent=False)

    def forward(self, x):
        """
        Apply rotary embedding to input tensor.

        Args:
            x (Tensor): Shape [seq_len, batch, dim]

        Returns:
            Tensor: Same shape as input, with rotary transformation applied.
        """
        seq_len = x.shape[0]
        sin = self.sin[:seq_len].unsqueeze(1)  # [seq_len, 1, dim/2]
        cos = self.cos[:seq_len].unsqueeze(1)  # [seq_len, 1, dim/2]

        # Split into even/odd channels
        x_even = x[..., ::2]  # [seq_len, batch, dim/2]
        x_odd = x[..., 1::2]  # [seq_len, batch, dim/2]

        # Apply rotation
        x_rot = torch.cat(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )

        return x_rot


class MovingAvg1d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TSDecomposition(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg1d(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, num_heads=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.LayerNorm(expansion_factor * embed_dim),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

    def forward(self, x, pad=None, return_attn=False):
        x_res = x
        x = self.norm1(x)
        if return_attn:
            attn_output, attn_weights = self.mha(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=pad,
            )
        else:
            attn_output, _ = self.mha(
                x,
                x,
                x,
                need_weights=False,
                key_padding_mask=pad,
            )

        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return (z + h, attn_weights) if return_attn else (z + h)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=1,
        embed_dim=64,
        num_layers=4,
        num_heads=4,
        expansion_factor=4,
        dropout=0.1,
        context_len=48,
        horizon=48,
        decompose=True,
        kernel_size=4,
    ):
        super().__init__()
        self.context_len = context_len
        self.horizon = horizon
        self.decompose = decompose

        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.pe = RotaryPositionalEmbedding(embed_dim, learnable=False)

        # A Learnable Token
        # self.token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.decompose:
            self.decomp = TSDecomposition(kernel_size=kernel_size)
            self.trend_encoder = nn.ModuleList(
                [
                    TransformerEncoderBlock(
                        embed_dim, expansion_factor, num_heads, dropout
                    )
                    for _ in range(num_layers)
                ]
            )
            self.res_encoder = nn.ModuleList(
                [
                    TransformerEncoderBlock(
                        embed_dim, expansion_factor, num_heads, dropout
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.encoder_layers = nn.ModuleList(
                [
                    TransformerEncoderBlock(
                        embed_dim, expansion_factor, num_heads, dropout
                    )
                    for _ in range(num_layers)
                ]
            )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, horizon),
        )

    def forward(self, x, pad=None):
        x = self.embed(x)  # [B, context_len, embed_dim]

        # --- Add a learnable token ---
        # bs = x.shape[0]
        # token = self.token.expand(bs, -1, -1)  # shape: [B, 1, embed_dim]
        # x = torch.cat((token, x), dim=1)  # shape: [B, 1 + T, embed_dim]
        # --- Positional Embedding ---
        # x = self.pe(x)

        if self.decompose:
            res, trend = self.decomp(x)
            for layer in self.res_encoder:
                res = layer(res, pad)
            for layer in self.trend_encoder:
                trend = layer(trend, pad)
            x = res + trend
        else:
            for layer in self.encoder_layers:
                x = layer(x, pad)

        # pooled = x[:, 0, :]
        pooled = x.mean(1)
        return self.head(pooled)


class TransformerForecaster(LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        context_len: int = 48,
        horizon: int = 48,
        embed_dim: int = 256,
        num_heads: int = 2,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        lr: float = 1e-4,
        decompose: bool = True,
        kernel_size: int = 25,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = Transformer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_enc_layers,
            num_heads=num_heads,
            context_len=context_len,
            horizon=horizon,
            dropout=dropout,
            decompose=decompose,
            kernel_size=kernel_size,
        )

        self.loss_fn = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.lr = lr

    def forward(self, x, pad):
        return self.model(x, pad)

    def training_step(self, batch, batch_idx):
        x, y, pad = batch
        y_hat = self(x, pad)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(
            "train_mae",
            self.mae(y_hat, y),
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pad = batch
        y_hat = self(x, pad)
        val_loss = self.loss_fn(y_hat, y)

        self.log(
            "val_loss", val_loss, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.log(
            "val_mae",
            self.mae(y_hat, y),
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

