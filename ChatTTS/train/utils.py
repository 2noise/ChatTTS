from typing import Literal

import torch
from einops import rearrange
from transformers.trainer_pt_utils import LabelSmoother
from vector_quantize_pytorch.residual_fsq import GroupedResidualFSQ

import ChatTTS
from ChatTTS.model.dvae import DVAE

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
AUDIO_EOS_TOKEN_ID: int = 0
AUDIO_PAD_TOKEN_ID: int = AUDIO_EOS_TOKEN_ID
# SPEAKER_TOKEN_ID: int = chat.tokenizer.spk_emb_ids
# AUDIO_EOS_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[Etts]')


def get_mel_specs(
    chat: ChatTTS.Chat,
    waveforms: torch.Tensor,  # (batch_size, time)
) -> torch.Tensor:
    mel_specs = chat.dvae.preprocessor_mel(waveforms)  # (batch_size, 100, mel_len)
    if mel_specs.size(2) % 2 != 0:
        mel_specs = torch.cat(
            [mel_specs, torch.zeros_like(mel_specs[:, :, :1])],
            dim=2,
        )
    return mel_specs  # (batch_size, 100, mel_len)


def get_dvae_mel_specs(
    chat: ChatTTS.Chat,
    mel_specs: torch.Tensor,  # (batch_size, 100, mel_len)
    mel_attention_mask: torch.Tensor,  # (batch_size, mel_len)
):
    audio_attention_mask = get_audio_attention_mask(
        mel_attention_mask
    )  # (batch_size, mel_len / 2)
    audio_latents = dvae_encode(
        chat.dvae, mel_specs
    )  # (batch_size, audio_dim, mel_len / 2)
    audio_latents = audio_latents * audio_attention_mask.unsqueeze(1)  # clip
    audio_quantized_latents, _ = dvae_quantize(
        chat.dvae.vq_layer.quantizer, audio_latents
    )  # (batch_size, audio_dim, mel_len / 2)
    audio_quantized_latents = audio_quantized_latents * audio_attention_mask.unsqueeze(
        1
    )
    dvae_mel_specs = dvae_decode(
        chat.dvae, audio_quantized_latents
    )  # (batch_size, 100, mel_len)
    return dvae_mel_specs  # (batch_size, 100, mel_len)


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
    indices = torch.ceil(indices * mel_len / waveform_attention_mask.size(1)).int()
    for i in range(batch_size):
        mel_attention_mask[i, indices[i] :] = 0
    return mel_attention_mask  # (batch_size, mel_len)


def get_audio_attention_mask(
    mel_attention_mask: torch.Tensor,  # (batch_size, mel_len)
):
    audio_attention_mask = mel_attention_mask[:, ::2]  # (batch_size, mel_len / 2)
    return audio_attention_mask  # (batch_size, mel_len / 2)


def dvae_encode(
    dvae: DVAE,
    mel_specs: torch.Tensor,  # (batch_size, 100, mel_len)
) -> torch.Tensor:
    x: torch.Tensor = dvae.downsample_conv(mel_specs / dvae.coef)
    x = dvae.encoder(x)
    return x  # (batch_size, audio_dim, mel_len / 2)


def dvae_quantize(
    quantizer: GroupedResidualFSQ,
    audio_latents: torch.Tensor,  # (batch_size, audio_dim=1024, mel_len / 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    # feat shape (batch_size, mel_len / 2, audio_dim)
    # ind shape (GFSQ.G, batch_size, mel_len / 2, GFSQ.R)
    # num_vq=GFSQ.G*GFSQ.R
    feat, ind = quantizer(audio_latents.transpose(1, 2))
    audio_quantized_latents = feat.transpose(
        1, 2
    )  # (batch_size, audio_dim, mel_len / 2)
    audio_input_ids = rearrange(
        ind, "g b t r ->b t (g r)"
    )  # (batch_size, mel_len / 2, num_vq)
    return audio_quantized_latents, audio_input_ids


def dvae_decode(
    dvae: DVAE,
    audio_latents: torch.Tensor,  # (batch_size, audio_dim, mel_len / 2)
) -> torch.Tensor:
    assert audio_latents.size(1) % 2 == 0
    reshaped_audio_latents = (
        audio_latents.view(
            (
                audio_latents.size(0),
                2,
                audio_latents.size(1) // 2,
                audio_latents.size(2),
            ),
        )
        .permute(0, 2, 3, 1)
        .flatten(2)
    )  # (batch_size, audio_dim / 2, mel_len)
    x: torch.Tensor = dvae.decoder(reshaped_audio_latents)
    x = dvae.out_conv(x)
    return x * dvae.coef  # (batch_size, 100, mel_len)


# TODO: a better name
def get_hidden_states_and_labels(
    chat: ChatTTS.Chat,
    mel_specs: torch.Tensor,  # (batch_size, 100, mel_len)
    mel_attention_mask: torch.Tensor,  # (batch_size, mel_len)
    text_input_ids: torch.Tensor,  # (batch_size, text_len)
    text_attention_mask: torch.Tensor,  # (batch_size, text_len)
    speakers: list[str],
    speaker_embeds: dict[str, torch.Tensor],
) -> dict[Literal["labels"] | Literal["hidden_states"], torch.Tensor]:
    audio_attention_mask = get_audio_attention_mask(
        mel_attention_mask
    )  # (batch_size, mel_len / 2)
    audio_latents = dvae_encode(
        chat.dvae, mel_specs
    )  # (batch_size, audio_dim, mel_len // 2)
    audio_latents = audio_latents * audio_attention_mask.unsqueeze(1)  # clip
    _, dvae_audio_input_ids = dvae_quantize(
        chat.dvae.vq_layer.quantizer, audio_latents
    )  # (batch_size, mel_len // 2)
    dvae_audio_input_ids[~audio_attention_mask.bool()] = AUDIO_PAD_TOKEN_ID

    batch_size = text_attention_mask.size(0)
    # add audio eos token
    extended_audio_attention_mask = torch.cat(
        [
            audio_attention_mask,
            torch.zeros(
                (batch_size, 1),
                dtype=audio_attention_mask.dtype,
                device=audio_attention_mask.device,
            ),
        ],
        dim=1,
    )  # (batch_size, mel_len+1)
    extended_audio_input_ids = torch.cat(
        [
            dvae_audio_input_ids,
            AUDIO_PAD_TOKEN_ID
            * torch.ones(
                (batch_size, 1, chat.gpt.num_vq),
                dtype=dvae_audio_input_ids.dtype,
                device=dvae_audio_input_ids.device,
            ),
        ],
        dim=1,
    )  # (batch_size, mel_len+1, num_vq)
    indices = audio_attention_mask.int().sum(dim=1)  # (batch_size,)
    for i in range(batch_size):
        extended_audio_attention_mask[i, indices[i]] = 1
        extended_audio_input_ids[i, indices[i]] = AUDIO_EOS_TOKEN_ID

    # combine text and audio
    input_ids = torch.cat(  # (batch_size, text_len + mel_len + 1, num_vq)
        [
            text_input_ids.unsqueeze(-1).repeat(
                1, 1, chat.gpt.num_vq
            ),  # (batch_size, text_len, num_vq)
            extended_audio_input_ids,  # (batch_size, mel_len, num_vq)
        ],
        dim=1,
    )
    attention_mask = torch.cat(  # (batch_size, text_len + mel_len + 1)
        [text_attention_mask, extended_audio_attention_mask],
        dim=1,
    )
    text_mask = torch.cat(  # (batch_size, text_len + mel_len + 1)
        [
            torch.ones_like(text_attention_mask, dtype=bool),
            torch.zeros_like(extended_audio_attention_mask, dtype=bool),
        ],
        dim=1,
    )
    # set labels
    labels = input_ids.clone()  # (batch_size, text_len + mel_len + 1, num_vq)
    labels[~attention_mask.bool()] = IGNORE_TOKEN_ID

    # (batch_size, text_len + mel_len, 768)
    inputs_embeds = chat.embed(input_ids, text_mask).clone()

    for i, speaker in enumerate(speakers):
        inputs_embeds[i] = chat.speaker.apply(
            emb=inputs_embeds[i].unsqueeze(0),
            spk_emb=speaker_embeds[speaker],
            input_ids=text_input_ids[i].unsqueeze(0),
            spk_emb_ids=chat.tokenizer.spk_emb_ids,
            device=chat.device,
        ).squeeze(0)
    # indices = torch.all(input_ids == SPEAKER_TOKEN_ID, dim=-1)
    # for i, speaker in enumerate(speakers):
    #     inputs_embeds[i, indices[i]] = torch.nn.functional.normalize(
    #         speaker_embeds[speaker].to(dtype=inputs_embeds.dtype),
    #         p=2.0,
    #         dim=-1,
    #         eps=1e-12,
    #     ).unsqueeze(0)

    # (batch_size, text_len + mel_len)
    outputs = chat.gpt.gpt.forward(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask
    )
    hidden_states = (
        outputs.last_hidden_state
    )  # (batch_size, text_len + mel_len + 1, 768)

    return {"labels": labels, "hidden_states": hidden_states}
