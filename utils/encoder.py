
import torch
import torch.nn as nn

from ChatTTS.model.dvae import ConvNeXtBlock, DVAEDecoder

from .wavenet import WaveNet


def get_encoder_config(decoder: DVAEDecoder) -> dict[str, int | bool]:
    return {
        'idim': decoder.conv_out.out_channels,
        'odim': decoder.conv_in[0].in_channels,
        'n_layer': len(decoder.decoder_block),
        'bn_dim': decoder.conv_in[0].out_channels,
        'hidden': decoder.conv_in[2].out_channels,
        'kernel': decoder.decoder_block[0].dwconv.kernel_size[0],
        'dilation': decoder.decoder_block[0].dwconv.dilation[0],
        'down': decoder.up,
    }


class DVAEEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer: int = 12,
        bn_dim: int = 64,
        hidden: int = 256,
        kernel: int = 7,
        dilation: int = 2,
        down: bool = False,
    ) -> None:
        super().__init__()
        self.wavenet = WaveNet(
            input_channels=100,
            residual_channels=idim,
            residual_layers=20,
            dilation_cycle=4,
        )
        self.conv_in_transpose = nn.ConvTranspose1d(idim, hidden, kernel_size=1, bias=False)
        # nn.Sequential(
        #     nn.ConvTranspose1d(100, idim, 3, 1, 1, bias=False),
        #     nn.ConvTranspose1d(idim, hidden, kernel_size=1, bias=False)
        # )
        self.encoder_block = nn.ModuleList([
            ConvNeXtBlock(hidden, hidden * 4, kernel, dilation,)
            for _ in range(n_layer)])
        self.conv_out_transpose = nn.Sequential(
            nn.Conv1d(hidden, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, odim, 3, 1, 1),
        )

    def forward(
        self,
        audio_mel_specs: torch.Tensor,  # (batch_size, audio_len*2, 100)
        audio_attention_mask: torch.Tensor,  # (batch_size, audio_len)
        conditioning=None,
    ) -> torch.Tensor:
        mel_attention_mask = audio_attention_mask.unsqueeze(-1).repeat(1, 1, 2).flatten(1)
        x: torch.Tensor = self.wavenet(audio_mel_specs.transpose(1, 2))   # (batch_size, idim, audio_len*2)
        x = x * mel_attention_mask.unsqueeze(1)
        x = self.conv_in_transpose(x)   # (batch_size, hidden, audio_len*2)
        for f in self.encoder_block:
            x = f(x, conditioning)
        x = self.conv_out_transpose(x)   # (batch_size, odim, audio_len*2)
        x = x.view(x.size(0), x.size(1), 2, x.size(2) // 2).permute(0, 3, 1, 2).flatten(2)
        return x   # (batch_size, audio_len, audio_dim=odim*2)
