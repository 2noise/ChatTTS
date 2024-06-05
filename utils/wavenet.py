"""https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/vqgan/modules/wavenet.py"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearNorm(nn.Module):
    """LinearNorm Projection"""

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(
        self,
        residual_channels,
        use_linear_bias=False,
        dilation=1,
        condition_channels=None,
    ):
        super(ResidualBlock, self).__init__()
        self.conv_layer = ConvNorm(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

        if condition_channels is not None:
            self.diffusion_projection = LinearNorm(
                residual_channels, residual_channels, use_linear_bias
            )
            self.condition_projection = ConvNorm(
                condition_channels, 2 * residual_channels, kernel_size=1
            )

        self.output_projection = ConvNorm(
            residual_channels, 2 * residual_channels, kernel_size=1
        )

    def forward(self, x, condition=None, diffusion_step=None):
        y = x

        if diffusion_step is not None:
            diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
            y = y + diffusion_step

        y = self.conv_layer(y)

        if condition is not None:
            condition = self.condition_projection(condition)
            y = y + condition

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        residual_channels: int = 512,
        residual_layers: int = 20,
        dilation_cycle: Optional[int] = 4,
        is_diffusion: bool = False,
        condition_channels: Optional[int] = None,
    ):
        super().__init__()

        # Input projection
        self.input_projection = None
        if input_channels is not None and input_channels != residual_channels:
            self.input_projection = ConvNorm(
                input_channels, residual_channels, kernel_size=1
            )

        if input_channels is None:
            input_channels = residual_channels

        self.input_channels = input_channels

        # Residual layers
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    use_linear_bias=False,
                    dilation=2 ** (i % dilation_cycle) if dilation_cycle else 1,
                    condition_channels=condition_channels,
                )
                for i in range(residual_layers)
            ]
        )

        # Skip projection
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1
        )

        # Output projection
        self.output_projection = None
        if output_channels is not None and output_channels != residual_channels:
            self.output_projection = ConvNorm(
                residual_channels, output_channels, kernel_size=1
            )

        if is_diffusion:
            self.diffusion_embedding = DiffusionEmbedding(residual_channels)
            self.mlp = nn.Sequential(
                LinearNorm(residual_channels, residual_channels * 4, False),
                Mish(),
                LinearNorm(residual_channels * 4, residual_channels, False),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t=None, condition=None):
        if self.input_projection is not None:
            x = self.input_projection(x)
            x = F.silu(x)

        if t is not None:
            t = self.diffusion_embedding(t)
            t = self.mlp(t)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, condition, t)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)

        if self.output_projection is not None:
            x = F.silu(x)
            x = self.output_projection(x)

        return x
