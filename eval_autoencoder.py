"""
CUDA_VISIBLE_DEVICES=1 python eval_autoencoder.py --data_path data/Xz/Bekki.list --tar_path data/Xz.tar --decoder_encoder_path ./saved_models/decoder_encoder.pth --dvae_encoder_path ./saved_models/dvae_encoder.pth --decoder_type decoder
"""

import argparse
from enum import StrEnum

import vocos
import transformers

import torch.utils.data
import torch.nn
import torchaudio

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import XzListTar, AudioCollator
from utils.model import quantize
from utils.encoder import DVAEEncoder, get_encoder_config


class DecoderType(StrEnum):
    DECODER = 'decoder'
    DVAE = 'dvae'


def main():
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--save_path', type=str, default='output')

    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    parser.add_argument('--data_path', type=str, default='dummy_data/xz_list_style/speaker_A.list', help='the data_path to json/list file')
    parser.add_argument('--tar_path', type=str, help='the tarball path with wavs')
    parser.add_argument('--tar_in_memory', action='store_true', help='load tarball in memory')
    parser.add_argument(
        '--decoder_type', type=str, default='decoder',
        choices=['decoder', 'dvae'],
    )
    parser.add_argument('--decoder_encoder_path', type=str)
    parser.add_argument('--decoder_decoder_path', type=str)
    parser.add_argument('--dvae_encoder_path', type=str)
    parser.add_argument('--dvae_decoder_path', type=str)
    args = parser.parse_args()
    save_path: str = args.save_path
    local_path: str | None = args.local_path
    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    tar_in_memory: bool = args.tar_in_memory
    decoder_type: DecoderType = args.decoder_type
    decoder_encoder_path: str = args.decoder_encoder_path
    decoder_decoder_path: str = args.decoder_decoder_path
    dvae_encoder_path: str = args.dvae_encoder_path
    dvae_decoder_path: str = args.dvae_decoder_path

    chat = ChatTTS.Chat()
    if local_path is None:
        chat.load_models()
    else:
        print('local model path:', local_path)
        chat.load_models('local', local_path=local_path)
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']

    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.pretrain_models['tokenizer'],
        vocos_model=chat.pretrain_models['vocos'],
        tar_path=tar_path,
        tar_in_memory=tar_in_memory,
        # device=None,
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id),
        # num_workers=4,
    )

    decoder_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']
    decoder_encoder: DVAEEncoder = DVAEEncoder(
        **get_encoder_config(decoder_decoder.decoder),
    )
    dvae_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['dvae']
    dvae_encoder: DVAEEncoder = DVAEEncoder(
        **get_encoder_config(dvae_decoder.decoder),
    )
    vocos_model: vocos.Vocos = chat.pretrain_models['vocos']

    # load pretrained models
    if decoder_encoder_path is not None:
        decoder_encoder.load_state_dict(torch.load(decoder_encoder_path, map_location=dataset.device))
    if decoder_decoder_path is not None:
        decoder_decoder.load_state_dict(torch.load(decoder_decoder_path, map_location=dataset.device))
    if dvae_encoder_path is not None:
        dvae_encoder.load_state_dict(torch.load(dvae_encoder_path, map_location=dataset.device))
    if dvae_decoder_path is not None:
        dvae_decoder.load_state_dict(torch.load(dvae_decoder_path, map_location=dataset.device))

    batch = next(iter(loader))
    audio_mel_specs: torch.Tensor = batch['audio_mel_specs']  # (batch_size, audio_len*2, 100)
    audio_attention_mask: torch.Tensor = batch['audio_attention_mask']  # (batch_size, audio_len)
    mel_attention_mask = audio_attention_mask.unsqueeze(-1).repeat(1, 1, 2).flatten(1)  # (batch_size, audio_len*2)

    if decoder_type == DecoderType.DECODER:
        encoder = decoder_encoder
        decoder = decoder_decoder
    else:
        encoder = dvae_encoder
        decoder = dvae_decoder
    vq_layer = decoder.vq_layer
    decoder.vq_layer = None

    encoder.to(dataset.device)
    decoder.to(dataset.device)

    # (batch_size, audio_len, audio_dim)
    audio_latents = encoder(audio_mel_specs, audio_attention_mask) * audio_attention_mask.unsqueeze(-1)
    # (batch_size, audio_len*2, 100)
    if vq_layer is not None:
        audio_latents, _ = quantize(vq_layer.quantizer, audio_latents)  # (batch_size, audio_len, num_vq)
    gen_mel_specs: torch.Tensor = decoder(audio_latents.transpose(1, 2)).transpose(1, 2) * mel_attention_mask.unsqueeze(-1)
    wav = vocos_model.decode(gen_mel_specs.transpose(1, 2)).cpu()
    org_wav = vocos_model.decode(audio_mel_specs.transpose(1, 2)).cpu()

    print(mel_attention_mask)
    print(org_wav.shape, wav.shape)
    torchaudio.save(save_path+'_org.wav', org_wav[0].view(1, -1), sample_rate=24_000)
    torchaudio.save(save_path+'.wav', wav[0].view(1, -1), sample_rate=24_000)


if __name__ == '__main__':
    main()
