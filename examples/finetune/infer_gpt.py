"""
CUDA_VISIBLE_DEVICES=0 python examples/finetune/infer_gpt.py --text "你好，我是恬豆"
--gpt_path ./saved_models/gpt.pth --decoder_path ./saved_models/decoder.pth --speaker_embeds_path ./saved_models/speaker_embeds.npz
"""  # noqa: E501

import argparse
import os
import random

import torch.utils.data
import torch.nn
import torchaudio
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae

from tools.normalizer import load_normalizer


def main():
    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=str)
    parser.add_argument("--save_path", type=str, default="./")

    parser.add_argument("--dvae_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--gpt_path", type=str)
    parser.add_argument("--speaker_embeds_path", type=str)
    args = parser.parse_args()
    text: str = args.text
    speaker: str | None = args.speaker
    save_path: str | None = args.save_path
    dvae_path: str | None = args.dvae_path
    decoder_path: str | None = args.decoder_path
    gpt_path: str | None = args.gpt_path
    speaker_embeds_path: str | None = args.speaker_embeds_path

    chat = ChatTTS.Chat()
    chat.load(compile=False)
    # load pretrained models
    if decoder_path is not None:
        chat.decoder.load_state_dict(torch.load(decoder_path, map_location=chat.device))
    if dvae_path is not None:
        chat.dvae.load_state_dict(torch.load(dvae_path, map_location=chat.device))
    if gpt_path is not None:
        chat.gpt.load_state_dict(torch.load(gpt_path, map_location=chat.device))
    speaker_embeds: dict[str, torch.Tensor] = {}
    if speaker_embeds_path is not None:
        np_speaker_embeds: dict[str, np.ndarray] = np.load(speaker_embeds_path)
        speaker_embeds = {
            speaker: torch.from_numpy(speaker_embed).to(chat.device)
            for speaker, speaker_embed in np_speaker_embeds.items()
        }

    if speaker is None:
        if len(speaker_embeds) == 0:
            speaker_embed = chat.speaker._sample_random()
        else:
            speaker_embed = random.choice(list(speaker_embeds.values()))
    else:
        speaker_embed = speaker_embeds[speaker]

    load_normalizer(chat)

    decoder_wav = chat.infer(
        [text],
        stream=False,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=chat.speaker._encode(speaker_embed),
        ),
    )
    print(decoder_wav[0].shape)
    torchaudio.save(
        os.path.join(save_path, "infer_gpt_decoder.wav"),
        torch.from_numpy(decoder_wav[0]).view(1, -1),
        sample_rate=24_000,
    )

    dvae_wav = chat.infer(
        [text],
        stream=False,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=chat.speaker._encode(speaker_embed),
        ),
    )
    print(dvae_wav[0].shape)
    torchaudio.save(
        os.path.join(save_path, "infer_gpt_dvae.wav"),
        torch.from_numpy(dvae_wav[0]).view(1, -1),
        sample_rate=24_000,
    )


if __name__ == "__main__":
    main()
