
import torch
from einops import rearrange

import ChatTTS
import ChatTTS.model.dvae


def quantize(
    vq: ChatTTS.model.dvae.GFSQ,
    audio_latents: torch.Tensor,   # (batch_size, audio_len, audio_dim=1024)
) -> tuple[torch.Tensor, torch.Tensor]:
    # feat shape (batch_size, audio_len, audio_dim)
    # ind shape (GFSQ.G, batch_size, audio_len, GFSQ.R)
    # num_vq=GFSQ.G*GFSQ.R
    feat, ind = vq.quantizer(audio_latents)
    audio_quantized_latents = feat   # (batch_size, audio_len, audio_dim)
    audio_input_ids = rearrange(   # (batch_size, audio_len, num_vq)
        ind, "g b t r ->b t (g r)",
    )
    return audio_quantized_latents, audio_input_ids
