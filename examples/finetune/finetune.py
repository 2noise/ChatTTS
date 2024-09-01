"""
CUDA_VISIBLE_DEVICES=0 python examples/finetune/finetune.py --color --save_folder ./saved_models --data_path Bekki.list --tar_path data/Xz.tar --batch_size 32 --epochs 10 --train_module dvae
CUDA_VISIBLE_DEVICES=0 python examples/finetune/finetune.py --color --save_folder ./saved_models --data_path Bekki.list --tar_path data/Xz.tar --batch_size 16 --epochs 10 --train_module gpt_speaker

--gpt_lora --tar_in_memory --process_ahead

"""  # noqa: E501

import argparse
import logging
import os

import torch.nn
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from ChatTTS.train.dataset import XzListTar
from ChatTTS.train.model import TrainModule, train_autoencoder, train_gpt

from tools.normalizer import load_normalizer

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
    parser.add_argument(
        "--train_module",
        type=str,
        default="gpt",
        choices=[
            "gpt_all",
            "gpt_speaker",
            "gpt",
            "speaker",
            "dvae",
            "dvae_encoder",
            "dvae_decoder",
            "decoder",
        ],
    )
    parser.add_argument("--train_text", action="store_true", help="train text loss")
    parser.add_argument("--gpt_lora", action="store_true", help="train gpt with lora")
    # parser.add_argument('--gpt_kbit', type=int, default=16, help='train gpt with kbit')
    parser.add_argument("--dvae_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--gpt_path", type=str)
    parser.add_argument("--speaker_embeds_path", type=str)
    parser.add_argument("--save_folder", type=str, default="./")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--color", action="store_true", help="colorful output")
    args = parser.parse_args()
    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    tar_in_memory: bool = args.tar_in_memory
    process_ahead: bool = args.process_ahead
    train_module: TrainModule = args.train_module
    train_text: bool = args.train_text
    gpt_lora: bool = args.gpt_lora
    # gpt_kbit: int = args.gpt_kbit
    save_folder: str = args.save_folder
    batch_size: int = args.batch_size
    epochs: int = args.epochs

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

    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT]:
        if gpt_lora:
            import peft

            # match gpt_kbit:
            #     case 4:
            #         quantization_config = transformers.BitsAndBytesConfig(
            #             load_in_4bit=True,
            #             bnb_4bit_quant_type="nf4",
            #             bnb_4bit_use_double_quant=True,
            #             bnb_4bit_compute_dtype=torch.bfloat16,
            #         )
            #     case 8:
            #         quantization_config = transformers.BitsAndBytesConfig(
            #             load_in_8bit=True,
            #     )
            # chat.gpt.gpt = transformers.LlamaModel.from_pretrained()
            # peft.prepare_model_for_gpt_kbit_training(chat.gpt.gpt)
            lora_config = peft.LoraConfig(r=8, lora_alpha=16)
            chat.gpt.gpt = peft.get_peft_model(chat.gpt.gpt, lora_config)

    match train_module:
        case (
            TrainModule.GPT_ALL
            | TrainModule.GPT_SPEAKER
            | TrainModule.GPT
            | TrainModule.SPEAKER
            | TrainModule.DECODER
        ):
            train = train_gpt
            kwargs = {"train_text": train_text, "speaker_embeds": speaker_embeds}
        case TrainModule.DVAE | TrainModule.DVAE_ENCODER | TrainModule.DVAE_DECODER:
            train = train_autoencoder
            kwargs = {}
        case _:
            raise ValueError(f"invalid train_module: {train_module}")

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
    speaker_embeds = train(
        chat=chat,
        dataset=dataset,
        train_module=train_module,
        batch_size=batch_size,
        epochs=epochs,
        **kwargs,
    )

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    gpt_save_path = os.path.join(save_folder, "gpt.pth")
    speaker_embeds_save_path = os.path.join(save_folder, "speaker_embeds.npz")
    decoder_save_path = os.path.join(save_folder, "decoder.pth")
    dvae_save_path = os.path.join(save_folder, "dvae.pth")
    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT] and gpt_lora:
        chat.gpt.gpt = chat.gpt.gpt.merge_and_unload()
    if speaker_embeds is not None:
        np_speaker_embeds = {
            speaker: speaker_embed.detach().cpu().numpy()
            for speaker, speaker_embed in speaker_embeds.items()
        }
    match train_module:
        case TrainModule.GPT_ALL:
            torch.save(chat.gpt.state_dict(), gpt_save_path)
            torch.save(chat.decoder.state_dict(), decoder_save_path)
            np.savez(speaker_embeds_save_path, **np_speaker_embeds)
        case TrainModule.GPT_SPEAKER:
            torch.save(chat.gpt.state_dict(), gpt_save_path)
            np.savez(speaker_embeds_save_path, **np_speaker_embeds)
        case TrainModule.GPT:
            torch.save(chat.gpt.state_dict(), gpt_save_path)
        case TrainModule.DECODER:
            torch.save(chat.decoder.state_dict(), decoder_save_path)
        case TrainModule.SPEAKER:
            np.savez(speaker_embeds_save_path, **np_speaker_embeds)
        case TrainModule.DVAE | TrainModule.DVAE_ENCODER | TrainModule.DVAE_DECODER:
            torch.save(chat.dvae.state_dict(), dvae_save_path)
    print("save models to:", save_folder)


if __name__ == "__main__":
    main()
