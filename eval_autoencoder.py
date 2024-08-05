"""
CUDA_VISIBLE_DEVICES=0 python eval_autoencoder.py --data_path data/Xz/Bekki.list --tar_path data/Xz.tar
--dvae_path saved_models/dvae.pth
"""

import argparse

import torch.utils.data
import torch.nn
import torchaudio

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import XzListTar, AudioCollator
from utils.model import get_mel_attention_mask, dvae_encode, dvae_quantize, dvae_decode


def main():
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--data_path', type=str, default='dummy_data/xz_list_style/speaker_A.list', help='the data_path to json/list file')
    parser.add_argument('--tar_path', type=str, help='the tarball path with wavs')
    parser.add_argument('--dvae_path', type=str)
    args = parser.parse_args()
    save_path: str = args.save_path
    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    dvae_path: str = args.dvae_path

    chat = ChatTTS.Chat()
    chat.load()
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
    waveforms: torch.Tensor = batch['waveforms']  # (batch_size, time)
    waveform_attention_mask: torch.Tensor = batch['waveform_attention_mask']  # (batch_size, time)

    waveforms = waveforms.to(chat.device, non_blocking=True)
    waveform_attention_mask = waveform_attention_mask.to(chat.device, non_blocking=True)

    mel_specs = chat.dvae.preprocessor_mel(waveforms)
    mel_specs = mel_specs[:, :, :mel_specs.size(2) // 2 * 2]  # (batch_size, 100, mel_len)
    mel_attention_mask = get_mel_attention_mask(waveform_attention_mask, mel_len=mel_specs.size(2))  # (batch_size, mel_len)
    mel_specs = mel_specs * mel_attention_mask.unsqueeze(1)

    audio_latents = dvae_encode(chat.dvae, mel_specs)    # (batch_size, audio_dim, mel_len // 2)
    audio_latents = audio_latents * mel_attention_mask[:, ::2].unsqueeze(1)
    audio_quantized_latents, _ = dvae_quantize(chat.dvae.vq_layer.quantizer, audio_latents)  # (batch_size, audio_dim, mel_len // 2)
    audio_quantized_latents = audio_quantized_latents * mel_attention_mask[:, ::2].unsqueeze(1)
    gen_mel_specs = dvae_decode(chat.dvae, audio_quantized_latents)    # (batch_size, 100, mel_len)
    gen_mel_specs = gen_mel_specs * mel_attention_mask.unsqueeze(1)

    wav = chat.vocos.decode(gen_mel_specs).cpu()
    org_wav = chat.vocos.decode(mel_specs).cpu()

    print(mel_attention_mask)
    print(org_wav.shape, wav.shape)
    torchaudio.save(save_path+'_org.wav', org_wav[0].view(1, -1), sample_rate=24_000)
    torchaudio.save(save_path+'.wav', wav[0].view(1, -1), sample_rate=24_000)


if __name__ == '__main__':
    main()
