
import torch
from einops import rearrange
from vector_quantize_pytorch.residual_fsq import GroupedResidualFSQ

from ChatTTS.model.dvae import DVAE


def get_mel_attention_mask(
    waveform_attention_mask: torch.Tensor,  # (batch_size, time)
    mel_len: int,
):
    batch_size = waveform_attention_mask.size(0)
    mel_attention_mask = torch.ones(
        (batch_size, mel_len),
        device=waveform_attention_mask.device,
    )
    indices = waveform_attention_mask.int().sum(dim=1)  # (batch_size,)
    indices = indices * mel_len // waveform_attention_mask.size(1)
    for i in range(batch_size):
        mel_attention_mask[i, indices[i]:] = 0
    return mel_attention_mask


def dvae_encode(
    dvae: DVAE,
    mel_specs: torch.Tensor,  # (batch_size, 100, mel_len)
) -> torch.Tensor:
    x: torch.Tensor = dvae.downsample_conv(mel_specs / dvae.coef)
    x = dvae.encoder(x)
    return x  # (batch_size, audio_dim, mel_len // 2)


def dvae_quantize(
    quantizer: GroupedResidualFSQ,
    audio_latents: torch.Tensor,   # (batch_size, audio_dim=1024, mel_len // 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    # feat shape (batch_size, mel_len // 2, audio_dim)
    # ind shape (GFSQ.G, batch_size, mel_len // 2, GFSQ.R)
    # num_vq=GFSQ.G*GFSQ.R
    feat, ind = quantizer(audio_latents.transpose(1, 2))
    audio_quantized_latents = feat.transpose(1, 2)   # (batch_size, audio_dim, mel_len // 2)
    audio_input_ids = rearrange(ind, "g b t r ->b t (g r)")   # (batch_size, mel_len // 2, num_vq)
    return audio_quantized_latents, audio_input_ids


def dvae_decode(
    dvae: DVAE,
    audio_latents: torch.Tensor,  # (batch_size, audio_dim, mel_len // 2)
) -> torch.Tensor:
    reshaped_audio_latents = audio_latents.view(
        (audio_latents.size(0), 2, audio_latents.size(1) // 2, audio_latents.size(2)),
    ).permute(0, 2, 3, 1).flatten(2)  # (batch_size, audio_dim // 2, mel_len)
    x: torch.Tensor = dvae.decoder(reshaped_audio_latents)
    x = dvae.out_conv(x)
    return x * dvae.coef    # (batch_size, 100, mel_len)
