import math
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualFSQ

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel, dilation,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 
                                kernel_size=kernel, padding=dilation*(kernel//2), 
                                dilation=dilation, groups=dim
                            )  # depthwise conv
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
    


class GFSQ(nn.Module):

    def __init__(self, 
            dim, levels, G, R, eps=1e-5, transpose = True
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
        
    def _embed(self, x):
        if self.transpose:
            x = x.transpose(1,2)
        x = rearrange(
            x, "b t (g r) -> g b t r", g = self.G, r = self.R,
        )  
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose(1,2) if self.transpose else feat
        
    def forward(self, x,):
        if self.transpose:
            x = x.transpose(1,2)
        feat, ind = self.quantizer(x)
        ind = rearrange(
            ind, "g b t r ->b t (g r)",
        )  
        embed_onehot = F.one_hot(ind.long(), self.n_ind).to(x.dtype)
        e_mean = torch.mean(embed_onehot, dim=[0,1])
        e_mean = e_mean / (e_mean.sum(dim=1) + self.eps).unsqueeze(1)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + self.eps), dim=1))
        
        return (
            torch.zeros(perplexity.shape, dtype=x.dtype, device=x.device),
            feat.transpose(1,2) if self.transpose else feat,
            perplexity,
            None,
            ind.transpose(1,2) if self.transpose else ind,
        )
        
class DVAEDecoder(nn.Module):
    def __init__(self, idim, odim,
                 n_layer = 12, bn_dim = 64, hidden = 256, 
                 kernel = 7, dilation = 2, up = False
                ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1), nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1)
        )
        self.decoder_block = nn.ModuleList([
            ConvNeXtBlock(hidden, hidden* 4, kernel, dilation,)
            for _ in range(n_layer)])
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, input, conditioning=None):
        # B, T, C
        x = input.transpose(1, 2)
        x = self.conv_in(x)
        for f in self.decoder_block:
            x = f(x, conditioning)
        
        x = self.conv_out(x)
        return x.transpose(1, 2)
    

class DVAE(nn.Module):
    def __init__(
        self, decoder_config, vq_config, dim=512
    ):
        super().__init__()
        self.register_buffer('coef', torch.randn(1, 100, 1))

        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        if vq_config is not None:
            self.vq_layer = GFSQ(**vq_config)
        else:
            self.vq_layer = None

    def forward(self, inp):

        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp.detach().clone()
            
        temp = torch.chunk(vq_feats, 2, dim=1) # flatten trick :)
        temp = torch.stack(temp, -1)
        vq_feats = temp.reshape(*temp.shape[:2], -1)

        vq_feats = vq_feats.transpose(1, 2)
        dec_out = self.decoder(input=vq_feats)
        dec_out = self.out_conv(dec_out.transpose(1, 2))
        mel = dec_out * self.coef

        return mel
