import math
from typing import List, Optional

import numpy as np
import pybase16384 as b14
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import GroupedResidualFSQ


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )  # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.gamma is not None:
            y *= self.gamma
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)

        x = y + residual
        del y

        return x


class GFSQ(nn.Module):

    def __init__(
        self, dim: int, levels: List[int], G: int, R: int, eps=1e-5, transpose=True
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=levels,
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        """
        x = rearrange(
            x, "b t (g r) -> g b t r", g = self.G, r = self.R,
        )
        """
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
        feat, ind = self.quantizer(x)
        """
        ind = rearrange(
            ind, "g b t r ->b t (g r)",
        )
        """
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        embed_onehot_tmp = F.one_hot(ind.long(), self.n_ind)
        embed_onehot = embed_onehot_tmp.to(x.dtype)
        del embed_onehot_tmp
        e_mean = torch.mean(embed_onehot, dim=[0, 1])
        # e_mean = e_mean / (e_mean.sum(dim=1) + self.eps).unsqueeze(1)
        torch.div(e_mean, (e_mean.sum(dim=1) + self.eps).unsqueeze(1), out=e_mean)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + self.eps), dim=1))

        return (
            torch.zeros(perplexity.shape, dtype=x.dtype, device=x.device),
            feat.transpose_(1, 2) if self.transpose else feat,
            perplexity,
            None,
            ind.transpose_(1, 2) if self.transpose else ind,
        )


class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


class DVAE(nn.Module):
    def __init__(
        self,
        decoder_config,
        vq_config,
        dim=512,
        coef: Optional[str] = None,
    ):
        super().__init__()
        if coef is None:
            coef = torch.rand(100)
        else:
            coef = torch.from_numpy(
                np.copy(np.frombuffer(b14.decode_from_string(coef), dtype=np.float32))
            )
        self.register_buffer("coef", coef.unsqueeze(0).unsqueeze_(2))

        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        if vq_config is not None:
            self.vq_layer = GFSQ(**vq_config)
        else:
            self.vq_layer = None

    def __repr__(self) -> str:
        return b14.encode_to_string(
            self.coef.cpu().numpy().astype(np.float32).tobytes()
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            if self.vq_layer is not None:
                vq_feats = self.vq_layer._embed(inp)
            else:
                vq_feats = inp

            vq_feats = (
                vq_feats.view(
                    (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
                )
                .permute(0, 2, 3, 1)
                .flatten(2)
            )

            dec_out = self.out_conv(
                self.decoder(
                    x=vq_feats,
                ),
            )

            return torch.mul(dec_out, self.coef, out=dec_out)
