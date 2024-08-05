"""
CUDA_VISIBLE_DEVICES=0 python finetune.py --color --save_folder ./saved_models --data_path data/Xz/Bekki.list --tar_path data/Xz.tar --tar_in_memory --process_ahead --batch_size 32 --epochs 10 --train_module dvae
CUDA_VISIBLE_DEVICES=0 python finetune.py --color --save_folder ./saved_models --data_path data/Xz/Bekki.list --tar_path data/Xz.tar --tar_in_memory --process_ahead --batch_size 16 --epochs 10 --train_module gpt_all --gpt_lora
"""  # noqa: E501

import argparse
import os
from enum import StrEnum

import torch.utils.data
import torch.nn
import torch.nn.functional
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import XzListTar, AudioFolder, AudioCollator
from utils.logger import MetricLogger
from utils.model import get_mel_attention_mask, dvae_encode, dvae_quantize, dvae_decode
from utils.output import ansi, get_ansi_len, output_iter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class TrainModule(StrEnum):
    GPT_ALL = 'gpt_all'  # GPT + SPEAKER + DECODER
    GPT_SPEAKER = 'gpt_speaker'  # GPT + SPEAKER

    GPT = 'gpt'
    DECODER = 'decoder'
    SPEAKER = 'speaker'

    DVAE = 'dvae'
    DVAE_ENCODER = 'dvae_encoder'
    DVAE_DECODER = 'dvae_decoder'


def train_autoencoder(
    chat: ChatTTS.Chat,
    dataset: AudioFolder,
    train_module: TrainModule = TrainModule.DVAE,
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_norm_clip: float = 1.0,
):
    chat.dvae.eval().requires_grad_(False)
    match train_module:
        case TrainModule.DVAE:
            chat.dvae.train().requires_grad_()
            train_params = list(chat.dvae.parameters())
        case TrainModule.DVAE_ENCODER:
            chat.dvae.downsample_conv.train().requires_grad_()
            chat.dvae.encoder.train().requires_grad_()
            chat.dvae.vq_layer.train().requires_grad_()
            train_params = []
            train_params += list(chat.dvae.downsample_conv.parameters())
            train_params += list(chat.dvae.encoder.parameters())
            train_params += list(chat.dvae.vq_layer.parameters())
        case TrainModule.DVAE_DECODER:
            chat.dvae.decoder.train().requires_grad_()
            chat.dvae.out_conv.train().requires_grad_()
            train_params = []
            train_params += list(chat.dvae.decoder.parameters())
            train_params += list(chat.dvae.out_conv.parameters())

    optimizer = torch.optim.AdamW(train_params, lr=lr, betas=[0.8, 0.99], eps=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-7)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioCollator(),
        # num_workers=4,
    )
    logger = MetricLogger()
    logger.create_meters(loss=None)
    for _epoch in range(epochs):
        _epoch += 1
        logger.reset()
        header: str = '{blue_light}{0}: {1}{reset}'.format(
            'Epoch', output_iter(_epoch, epochs), **ansi)
        header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
        iterator = logger.log_every(loader, header=header, tqdm_header='Batch')
        for batch in iterator:
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

            loss = torch.nn.functional.mse_loss(gen_mel_specs, mel_specs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, grad_norm_clip)
            optimizer.step()
            logger.meters['loss'].update(loss.item(), n=len(waveform_attention_mask))
        lr_scheduler.step()
    optimizer.zero_grad()


def train_gpt(
    chat: ChatTTS.Chat,
    dataset: AudioFolder,
    train_module: TrainModule = TrainModule.GPT_ALL,
    batch_size: int = 16,
    epochs: int = 10,
    train_text: bool = True,
    speaker_embeds: dict[str, torch.Tensor] = {},
) -> dict[str, torch.Tensor]:
    speaker_embeds = {
        speaker: chat._sample_random_speaker().requires_grad_(
            train_module in [TrainModule.GPT_ALL, TrainModule.GPT_SPEAKER, TrainModule.SPEAKER],
        ) for speaker in dataset.speakers
    } | speaker_embeds
    for speaker_embed in speaker_embeds.values():
        speaker_embed.data = speaker_embed.data * chat.std + chat.mean
    SPEAKER_TOKEN_ID: int = chat.tokenizer.spk_emb_ids
    AUDIO_EOS_TOKEN_ID: int = 0
    # AUDIO_EOS_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[Etts]')
    AUDIO_PAD_TOKEN_ID: int = AUDIO_EOS_TOKEN_ID

    chat.dvae.eval().requires_grad_(False)
    chat.gpt.eval().requires_grad_(False)
    chat.decoder.eval().requires_grad_(False)
    match train_module:
        case TrainModule.GPT_ALL:
            chat.gpt.train().requires_grad_()
            chat.decoder.train().requires_grad_()
            train_params = []
            train_params += list(speaker_embeds.values())
            train_params += list(chat.gpt.parameters())
            train_params += list(chat.decoder.parameters())
            optimizer = torch.optim.Adam(chat.gpt.parameters(), lr=1e-3, weight_decay=0, betas=[0.9, 0.95], eps=1e-5)
            optimizer.add_param_group({'params': chat.decoder.parameters(), 'lr': 1e-3, 'weight_decay': 0, 'betas': [0.9, 0.95], 'eps': 1e-5})
            optimizer.add_param_group({'params': speaker_embeds.values(), 'lr': 1e-2, 'weight_decay': 0, 'betas': [0.9, 0.95], 'eps': 1e-5})
        case TrainModule.GPT_SPEAKER:
            train_params = []
            train_params += list(speaker_embeds.values())
            train_params += list(chat.gpt.parameters())
            optimizer = torch.optim.Adam(chat.gpt.parameters(), lr=1e-3, weight_decay=0, betas=[0.9, 0.95], eps=1e-5)
            optimizer.add_param_group({'params': speaker_embeds.values(), 'lr': 1e-2, 'weight_decay': 0, 'betas': [0.9, 0.95], 'eps': 1e-5})
        case TrainModule.GPT:
            chat.gpt.train().requires_grad_()
            train_params = list(chat.gpt.parameters())
        case TrainModule.DECODER:
            chat.decoder.train().requires_grad_()
            train_params = list(chat.decoder.parameters())
        case TrainModule.SPEAKER:
            train_params = list(speaker_embeds.values())
            optimizer = torch.optim.Adam(train_params, lr=1e-2, weight_decay=0, betas=[0.9, 0.95], eps=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-7)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial())

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioCollator(),
        # num_workers=4,
    )
    logger = MetricLogger()
    logger.create_meters(loss=None, audio_loss=None, text_loss=None, mse_loss=None)
    for _epoch in range(epochs):
        _epoch += 1
        logger.reset()
        header: str = '{blue_light}{0}: {1}{reset}'.format(
            'Epoch', output_iter(_epoch, epochs), **ansi)
        header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
        iterator = logger.log_every(loader, header=header, tqdm_header='Batch')
        for batch in iterator:
            speakers: list[str] = batch['speaker']  # (batch_size,)
            text_input_ids: torch.Tensor = batch['text_input_ids']   # (batch_size, text_len)
            text_attention_mask: torch.Tensor = batch['text_attention_mask']   # (batch_size, text_len)
            waveforms: torch.Tensor = batch['waveforms']   # (batch_size, time)
            waveform_attention_mask: torch.Tensor = batch['waveform_attention_mask']   # (batch_size, time)

            text_input_ids = text_input_ids.to(chat.device, non_blocking=True)
            text_attention_mask = text_attention_mask.to(chat.device, non_blocking=True)
            waveforms = waveforms.to(chat.device, non_blocking=True)
            waveform_attention_mask = waveform_attention_mask.to(chat.device, non_blocking=True)

            mel_specs = chat.dvae.preprocessor_mel(waveforms)
            mel_specs = mel_specs[:, :, :mel_specs.size(2) // 2 * 2]  # (batch_size, 100, mel_len)
            mel_attention_mask = get_mel_attention_mask(waveform_attention_mask, mel_len=mel_specs.size(2))  # (batch_size, mel_len)
            mel_specs = mel_specs * mel_attention_mask.unsqueeze(1)

            audio_latents = dvae_encode(chat.dvae, mel_specs)    # (batch_size, audio_dim, mel_len // 2)
            audio_latents = audio_latents * mel_attention_mask[:, ::2].unsqueeze(1)
            _, dvae_audio_input_ids = dvae_quantize(chat.dvae.vq_layer.quantizer, audio_latents)  # (batch_size, mel_len // 2)
            dvae_audio_input_ids[~mel_attention_mask[:, ::2].bool()] = AUDIO_PAD_TOKEN_ID

            batch_size, text_len = text_attention_mask.size()
            # add audio eos token
            extended_audio_attention_mask = torch.cat(
                [
                    mel_attention_mask[:, ::2],
                    torch.zeros(
                        (batch_size, 1),
                        dtype=mel_attention_mask[:, ::2].dtype,
                        device=mel_attention_mask[:, ::2].device,
                    ),
                ],
                dim=1,
            )  # (batch_size, mel_len+1)
            extended_audio_input_ids = torch.cat(
                [
                    dvae_audio_input_ids,
                    AUDIO_PAD_TOKEN_ID * torch.ones(
                        (batch_size, 1, chat.gpt.num_vq),
                        dtype=dvae_audio_input_ids.dtype,
                        device=dvae_audio_input_ids.device,
                    ),
                ],
                dim=1,
            )  # (batch_size, mel_len+1, num_vq)
            indices = mel_attention_mask[:, ::2].int().sum(dim=1)  # (batch_size,)
            for i in range(batch_size):
                extended_audio_attention_mask[i, indices[i]] = 1
                extended_audio_input_ids[i, indices[i]] = AUDIO_EOS_TOKEN_ID

            # combine text and audio
            input_ids = torch.cat(   # (batch_size, text_len + mel_len + 1, num_vq)
                [
                    text_input_ids.unsqueeze(-1).repeat(1, 1, chat.gpt.num_vq),   # (batch_size, text_len, num_vq)
                    extended_audio_input_ids,   # (batch_size, mel_len, num_vq)
                ],
                dim=1,
            )
            attention_mask = torch.cat(   # (batch_size, text_len + mel_len + 1)
                [text_attention_mask, extended_audio_attention_mask],
                dim=1,
            )
            text_mask = torch.cat(   # (batch_size, text_len + mel_len + 1)
                [
                    torch.ones_like(text_attention_mask, dtype=bool),
                    torch.zeros_like(extended_audio_attention_mask, dtype=bool),
                ],
                dim=1,
            )
            # set labels
            labels = input_ids.clone()   # (batch_size, text_len + mel_len + 1, num_vq)
            labels[~attention_mask.bool()] = IGNORE_TOKEN_ID

            # (batch_size, text_len + mel_len, 768)
            inputs_embeds = chat.gpt.forward(input_ids=input_ids, text_mask=text_mask)

            # (batch_size, text_len + mel_len)
            indices = torch.all(input_ids == SPEAKER_TOKEN_ID, dim=-1)
            for i, speaker in enumerate(speakers):
                inputs_embeds[i, indices[i]] = torch.nn.functional.normalize(
                    speaker_embeds[speaker].to(dtype=inputs_embeds.dtype),
                    p=2.0,
                    dim=-1,
                    eps=1e-12,
                ).unsqueeze(0)
            outputs = chat.gpt.gpt.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch_size, text_len + mel_len + 1, 768)
            text_hidden_states = hidden_states[:, :text_len-1]  # (batch_size, text_len-1, 768)
            audio_hidden_states = hidden_states[:, text_len-1:-1]  # (batch_size, mel_len+1, 768)

            audio_logits = torch.stack(
                [chat.gpt.head_code[i](audio_hidden_states) for i in range(chat.gpt.num_vq)],
                dim=2,
            )  # (batch_size, mel_len+1, num_vq, num_class_audio)
            audio_loss: torch.Tensor = torch.nn.functional.cross_entropy(audio_logits.flatten(0, 2), labels[:, text_len:].flatten(0, 2))
            loss: torch.Tensor = audio_loss
            if train_text:
                text_logits: torch.Tensor = chat.gpt.head_text(text_hidden_states)  # (batch_size, text_len-1, num_class_text)
                text_loss: torch.Tensor = torch.nn.functional.cross_entropy(text_logits.flatten(0, 1), labels[:, 1:text_len, 0].flatten(0, 1))
                loss += text_loss
                logger.meters['text_loss'].update(text_loss.item(), n=batch_size)

            gen_mel_specs = chat.decoder(audio_hidden_states[:, :-1].transpose(1, 2))
            mse_loss = torch.nn.functional.mse_loss(
                gen_mel_specs,
                mel_specs,
            )
            loss += 0.01 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()
            logger.meters['loss'].update(loss.item(), n=batch_size)
            logger.meters['mse_loss'].update(mse_loss.item(), n=batch_size)
            logger.meters['audio_loss'].update(audio_loss.item(), n=batch_size)
        lr_scheduler.step()
    optimizer.zero_grad()
    return speaker_embeds


def main():
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--data_path', type=str, default='dummy_data/xz_list_style/speaker_A.list', help='the data_path to json/list file')
    parser.add_argument('--tar_path', type=str, help='the tarball path with wavs')
    parser.add_argument('--tar_in_memory', action='store_true', help='load tarball in memory')
    parser.add_argument('--process_ahead', action='store_true', help='process all data ahead during dataset initialization')
    parser.add_argument(
        '--train_module', type=str, default='gpt',
        choices=['gpt_all', 'gpt_speaker', 'gpt', 'speaker', 'dvae', 'dvae_encoder', 'dvae_decoder', 'decoder'],
    )
    parser.add_argument('--train_text', action='store_true', help='train text loss')
    parser.add_argument('--gpt_lora', action='store_true', help='train gpt with lora')
    # parser.add_argument('--gpt_kbit', type=int, default=16, help='train gpt with kbit')
    parser.add_argument('--dvae_path', type=str)
    parser.add_argument('--decoder_path', type=str)
    parser.add_argument('--gpt_path', type=str)
    parser.add_argument('--speaker_embeds_path', type=str)
    parser.add_argument('--save_folder', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--color', action='store_true', help='colorful output')
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
    chat.load()
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
        case TrainModule.GPT_ALL | TrainModule.GPT_SPEAKER | TrainModule.GPT | TrainModule.SPEAKER | TrainModule.DECODER:
            train = train_gpt
            kwargs = {'train_text': train_text, 'speaker_embeds': speaker_embeds}
        case TrainModule.DVAE | TrainModule.DVAE_ENCODER | TrainModule.DVAE_DECODER:
            train = train_autoencoder
            kwargs = {}
        case _:
            raise ValueError(f'invalid train_module: {train_module}')

    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.tokenizer._tokenizer,
        normalizer=chat.normalizer,
        tar_path=tar_path,
        tar_in_memory=tar_in_memory,
        process_ahead=process_ahead,
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )
    speaker_embeds = train(chat=chat, dataset=dataset, train_module=train_module, batch_size=batch_size, epochs=epochs, **kwargs)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    gpt_save_path = os.path.join(save_folder, 'gpt.pth')
    speaker_embeds_save_path = os.path.join(save_folder, 'speaker_embeds.npz')
    decoder_save_path = os.path.join(save_folder, 'decoder.pth')
    dvae_save_path = os.path.join(save_folder, 'dvae.pth')
    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT] and gpt_lora:
        chat.gpt.gpt = chat.gpt.gpt.merge_and_unload()
    if speaker_embeds is not None:
        np_speaker_embeds = {speaker: speaker_embed.detach().cpu().numpy() for speaker, speaker_embed in speaker_embeds.items()}
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
    print('save models to:', save_folder)


if __name__ == '__main__':
    main()
