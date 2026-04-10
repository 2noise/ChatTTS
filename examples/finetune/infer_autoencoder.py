"""
CUDA_VISIBLE_DEVICES=0 python examples/finetune/infer_autoencoder.py --data_path Bekki.list --tar_path data/Xz.tar
--dvae_path saved_models/dvae.pth
"""  # noqa: E501

import argparse
import os

import torch.utils.data
import torch.nn
import torchaudio

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from ChatTTS.train.dataset import XzListTar, AudioCollator
from ChatTTS.train.model import (
    get_mel_specs,
    get_mel_attention_mask,
    get_dvae_mel_specs,
)


def main():
    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dummy_data/xz_list_style/speaker_A.list",
        help="the data_path to json/list file",
    )
    parser.add_argument("--tar_path", type=str, help="the tarball path with wavs")
    parser.add_argument("--dvae_path", type=str)
    args = parser.parse_args()
    save_path: str = args.save_path
    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    dvae_path: str = args.dvae_path

    chat = ChatTTS.Chat()
    chat.load(compile=False)
    if dvae_path is not None:
        chat.dvae.load_state_dict(torch.load(dvae_path, map_location=chat.device))

    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.tokenizer._tokenizer,
        normalizer=chat.normalizer,
        tar_path=tar_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=AudioCollator(),
        # num_workers=4,
    )

    batch = next(iter(loader))
    waveforms: torch.Tensor = batch["waveforms"]  # (batch_size, time)
    waveform_attention_mask: torch.Tensor = batch[
        "waveform_attention_mask"
    ]  # (batch_size, time)

    waveforms = waveforms.to(chat.device, non_blocking=True)
    waveform_attention_mask = waveform_attention_mask.to(chat.device, non_blocking=True)

    mel_specs = get_mel_specs(chat, waveforms)  # (batch_size, 100, mel_len)
    mel_attention_mask = get_mel_attention_mask(
        waveform_attention_mask, mel_len=mel_specs.size(2)
    )  # (batch_size, mel_len)
    mel_specs = mel_specs * mel_attention_mask.unsqueeze(1)  # clip

    dvae_mel_specs = get_dvae_mel_specs(
        chat, mel_specs, mel_attention_mask
    )  # (batch_size, 100, mel_len)
    dvae_mel_specs = dvae_mel_specs * mel_attention_mask.unsqueeze(1)  # clip

    wav = chat.vocos.decode(dvae_mel_specs).cpu()
    org_wav = chat.vocos.decode(mel_specs).cpu()

    print("Original Waveform shape:", org_wav.shape)
    print(wav.shape)
    torchaudio.save(
        os.path.join(save_path, "infer_autoencoder_org.wav"),
        org_wav[0].view(1, -1),
        sample_rate=24_000,
    )
    torchaudio.save(
        os.path.join(save_path, "infer_autoencoder.wav"),
        wav[0].view(1, -1),
        sample_rate=24_000,
    )


if __name__ == "__main__":
    main()
