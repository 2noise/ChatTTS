"""
CUDA_VISIBLE_DEVICES=0 python examples/finetune/validate.py --color --data_path Bekki.list --tar_path data/Xz.tar --batch_size 16
--gpt_path ./saved_models/gpt.pth --decoder_path ./saved_models/decoder.pth --speaker_embeds_path ./saved_models/speaker_embeds.npz
--dvae_path ./saved_models/dvae.pth

--tar_in_memory --process_ahead
"""  # noqa: E501

import argparse
import logging

import torch.utils.data
import torch.nn
import torch.nn.functional
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from ChatTTS.train.dataset import XzListTar
from ChatTTS.train.model import train_autoencoder, train_gpt

from tools.normalizer import load_normalizer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logging.basicConfig(level=logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dummy_data/xz_list_style/speaker_A.list",
        help="the data_path to json/list file",
    )
    parser.add_argument("--tar_path", type=str, help="the tarball path with wavs")
    parser.add_argument(
        "--tar_in_memory", action="store_true", help="load tarball in memory"
    )
    parser.add_argument(
        "--process_ahead",
        action="store_true",
        help="process all data ahead during dataset initialization",
    )
    # parser.add_argument('--gpt_kbit', type=int, default=16, help='train gpt with kbit')
    parser.add_argument("--dvae_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--gpt_path", type=str)
    parser.add_argument("--speaker_embeds_path", type=str)
    parser.add_argument("--color", action="store_true", help="colorful output")
    args = parser.parse_args()
    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    tar_in_memory: bool = args.tar_in_memory
    process_ahead: bool = args.process_ahead
    # gpt_kbit: int = args.gpt_kbit

    decoder_path: str = args.decoder_path
    dvae_path: str = args.dvae_path
    gpt_path: str = args.gpt_path
    speaker_embeds_path: str = args.speaker_embeds_path

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

    load_normalizer(chat)

    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.tokenizer._tokenizer,
        normalizer=chat.normalizer,
        tar_path=tar_path,
        tar_in_memory=tar_in_memory,
        process_ahead=process_ahead,
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )
    train_autoencoder(chat=chat, dataset=dataset, validate=True)
    train_gpt(
        chat=chat,
        dataset=dataset,
        speaker_embeds=speaker_embeds,
        train_text=True,
        validate=True,
    )


if __name__ == "__main__":
    main()
