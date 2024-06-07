"""
CUDA_VISIBLE_DEVICES=1 python eval_gpt.py --decoder_type decoder --text "你好，我是恬豆" --gpt_path ./saved_models/gpt.pth --speaker_embeds_path ./saved_models/speaker_embeds.npz
"""

import argparse
from enum import StrEnum

import vocos

import torch.utils.data
import torch.nn
import torchaudio
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae


class DecoderType(StrEnum):
    DECODER = 'decoder'
    DVAE = 'dvae'


def main():
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--speaker', type=str)
    parser.add_argument('--save_path', type=str, default='output.wav')

    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    parser.add_argument(
        '--decoder_type', type=str, default='decoder',
        choices=['decoder', 'dvae'],
    )
    parser.add_argument('--decoder_decoder_path', type=str)
    parser.add_argument('--dvae_decoder_path', type=str)
    parser.add_argument('--gpt_path', type=str)
    parser.add_argument('--speaker_embeds_path', type=str)
    args = parser.parse_args()
    text: str = args.text
    speaker: str = args.speaker
    save_path: str = args.save_path
    local_path: str | None = args.local_path
    decoder_type: DecoderType = args.decoder_type
    decoder_decoder_path: str = args.decoder_decoder_path
    dvae_decoder_path: str = args.dvae_decoder_path
    gpt_path: str = args.gpt_path
    speaker_embeds_path: str = args.speaker_embeds_path

    chat = ChatTTS.Chat()
    if local_path is None:
        chat.load_models()
    else:
        print('local model path:', local_path)
        chat.load_models('local', local_path=local_path)

    decoder_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']
    dvae_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['dvae']
    gpt: ChatTTS.model.gpt.GPT_wrapper = chat.pretrain_models['gpt']
    vocos_model: vocos.Vocos = chat.pretrain_models['vocos']

    device = next(vocos_model.parameters()).device

    # load pretrained models
    if decoder_decoder_path is not None:
        decoder_decoder.load_state_dict(torch.load(decoder_decoder_path, map_location=device))
    if dvae_decoder_path is not None:
        dvae_decoder.load_state_dict(torch.load(dvae_decoder_path, map_location=device))
    if gpt_path is not None:
        gpt.load_state_dict(torch.load(gpt_path, map_location=device))
    if speaker_embeds_path is None:
        speaker_embeds: dict[str, torch.Tensor] = {}
    else:
        np_speaker_embeds: dict[str, np.ndarray] = np.load(speaker_embeds_path)
        speaker_embeds = {
            speaker: torch.from_numpy(speaker_embed).to(device)
            for speaker, speaker_embed in np_speaker_embeds.items()
        }

    if speaker is None:
        assert len(speaker_embeds) == 1
        speaker_embed = next(iter(speaker_embeds.values()))
    else:
        speaker_embed = speaker_embeds[speaker]
    params_infer_code = {
        'spk_emb': speaker_embed,
        # 'temperature': temperature,
        # 'top_P': top_P,
        # 'top_K': top_K,
    }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    wav = chat.infer(
        text,
        skip_refine_text=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=(decoder_type == DecoderType.DECODER),
    )
    print(wav[0].shape)
    torchaudio.save(save_path, torch.from_numpy(wav[0]).view(1, -1), sample_rate=24_000)


if __name__ == '__main__':
    main()
