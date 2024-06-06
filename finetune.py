"""
python finetune.py --save_folder ./saved_models --data_path data/all.list --train_module encoder --decoder_type decoder
python finetune.py --save_folder ./saved_models --data_path data/all.list --train_module encoder --decoder_type dvae
python finetune.py --save_folder ./saved_models --data_path data/Bekki.list --train_module gpt_speaker --gpt_lora --decoder_encoder_path ./saved_models/decoder_encoder.pth --dvae_encoder_path ./saved_models/dvae_encoder.pth
"""

import argparse
import functools
import os
from enum import StrEnum
from tqdm import tqdm

import torch.utils.data
import torch.nn
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import ListFolder, AudioFolder, AudioCollator
from utils.model import quantize
from utils.encoder import DVAEEncoder, get_encoder_config

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class TrainModule(StrEnum):
    GPT_SPEAKER = 'gpt_speaker'
    GPT = 'gpt'
    SPEAKER = 'speaker'

    AUTOENCODER = 'autoencoder'
    ENCODER = 'encoder'
    DECODER = 'decoder'


class DecoderType(StrEnum):
    DECODER = 'decoder'
    DVAE = 'dvae'


def train_autoencoder(
        chat: ChatTTS.Chat,
        dataset: AudioFolder,
        encoder: DVAEEncoder,
        decoder: ChatTTS.model.dvae.DVAE,
        train_module: TrainModule = TrainModule.AUTOENCODER,
):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']
    encoder: DVAEEncoder = DVAEEncoder(
        **get_encoder_config(decoder.decoder),
    ).to(device=dataset.device)

    match train_module:
        case TrainModule.AUTOENCODER:
            encoder.train().requires_grad_()
            decoder.train().requires_grad_()
            train_params = list(encoder.parameters()) + list(decoder.parameters())
        case TrainModule.ENCODER:
            encoder.train().requires_grad_()
            decoder.eval().requires_grad_(False)
            train_params = list(encoder.parameters())
        case TrainModule.DECODER:
            encoder.eval().requires_grad_(False)
            decoder.train().requires_grad_()
            train_params = list(decoder.parameters())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(train_params, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-6)

    vq_layer = decoder.vq_layer
    decoder.vq_layer = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id))
    for epoch in range(10):
        for batch in tqdm(loader):
            audio_mel_specs: torch.Tensor = batch['audio_mel_specs']  # (batch_size, audio_len*2, 100)
            audio_attention_mask: torch.Tensor = batch['audio_attention_mask']  # (batch_size, audio_len)
            mel_attention_mask = audio_attention_mask.unsqueeze(-1).repeat(1, 1, 2).flatten(1)  # (batch_size, audio_len*2)

            # (batch_size, audio_len, audio_dim)
            audio_latents: torch.Tensor = encoder(audio_mel_specs, audio_attention_mask) * audio_attention_mask.unsqueeze(-1)
            # (batch_size, audio_len*2, 100)
            if vq_layer is not None:
                audio_latents, _ = quantize(vq_layer.quantizer, audio_latents)  # (batch_size, audio_len, num_vq)
            gen_mel_specs: torch.Tensor = decoder(audio_latents.transpose(1, 2)).transpose(1, 2) * mel_attention_mask.unsqueeze(-1)

            loss: torch.Tensor = loss_fn(gen_mel_specs, audio_mel_specs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    optimizer.zero_grad()
    decoder.vq_layer = vq_layer


def train_gpt(
    chat: ChatTTS.Chat,
    dataset: AudioFolder,
    decoder_encoder: DVAEEncoder,
    dvae_encoder: DVAEEncoder,
    train_module: TrainModule = TrainModule.GPT_SPEAKER,
    train_text: bool = True,
    speaker_embeds: dict[str, torch.Tensor] = {},
):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']

    decoder_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']
    decoder_decoder.eval().requires_grad_(False)
    # decoder_encoder: DVAEEncoder = DVAEEncoder(
    #     **get_encoder_config(decoder_decoder.decoder),
    # )
    decoder_encoder.to(device=dataset.device).eval().requires_grad_(False)

    dvae_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['dvae']
    dvae_decoder.eval().requires_grad_(False)
    # dvae_encoder: DVAEEncoder = DVAEEncoder(
    #     **get_encoder_config(dvae_decoder.decoder),
    # )
    dvae_encoder.to(device=dataset.device).eval().requires_grad_(False)

    gpt: ChatTTS.model.gpt.GPT_wrapper = chat.pretrain_models['gpt']
    if train_module == TrainModule.SPEAKER:
        gpt.eval().requires_grad_(False)
    else:
        gpt.train().requires_grad_()

    speaker_embeds = {
        speaker: torch.randn(
            768,
            device=dataset.device,
            requires_grad=train_module in [TrainModule.GPT_SPEAKER, TrainModule.SPEAKER],
        ) for speaker in dataset.speakers
    } | speaker_embeds
    SPEAKER_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[spk_emb]')
    AUDIO_EOS_TOKEN_ID: int = 0
    # AUDIO_EOS_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[Etts]')
    AUDIO_PAD_TOKEN_ID: int = AUDIO_EOS_TOKEN_ID

    match train_module:
        case TrainModule.GPT_SPEAKER:
            train_params = list(gpt.parameters()) + list(speaker_embeds.values())
        case TrainModule.GPT:
            train_params = list(gpt.parameters())
        case TrainModule.SPEAKER:
            train_params = list(speaker_embeds.values())

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_params, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-6)

    loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id))
    for epoch in range(10):
        for batch in tqdm(loader):
            speakers: list[str] = batch['speaker']  # (batch_size,)
            text_input_ids: torch.Tensor = batch['text_input_ids']   # (batch_size, text_len)
            text_attention_mask: torch.Tensor = batch['text_attention_mask']   # (batch_size, text_len)
            audio_mel_specs: torch.Tensor = batch['audio_mel_specs']   # (batch_size, audio_len*2, 100)
            audio_attention_mask: torch.Tensor = batch['audio_attention_mask']   # (batch_size, audio_len)

            batch_size, text_len = text_attention_mask.size()

            dvae_audio_latents = dvae_encoder(audio_mel_specs, audio_attention_mask)  # (batch_size, audio_len, audio_dim=1024)
            _, dvae_audio_input_ids = quantize(dvae_decoder.vq_layer.quantizer, dvae_audio_latents)  # (batch_size, audio_len, num_vq)
            dvae_audio_input_ids[~audio_attention_mask.bool()] = AUDIO_PAD_TOKEN_ID

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
            )  # (batch_size, audio_len+1)
            extended_audio_input_ids = torch.cat(
                [
                    dvae_audio_input_ids,
                    AUDIO_PAD_TOKEN_ID * torch.ones(
                        (batch_size, 1, gpt.num_vq),
                        dtype=dvae_audio_input_ids.dtype,
                        device=dvae_audio_input_ids.device,
                    ),
                ],
                dim=1,
            )  # (batch_size, audio_len+1, num_vq)
            indices = audio_attention_mask.int().sum(dim=1)  # (batch_size,)
            for i in range(batch_size):
                extended_audio_attention_mask[i, indices[i]] = 1
                extended_audio_input_ids[i, indices[i]] = AUDIO_EOS_TOKEN_ID

            # combine text and audio
            input_ids = torch.cat(   # (batch_size, text_len + audio_len + 1, num_vq)
                [
                    text_input_ids.unsqueeze(-1).repeat(1, 1, gpt.num_vq),   # (batch_size, text_len, num_vq)
                    extended_audio_input_ids,   # (batch_size, audio_len, num_vq)
                ],
                dim=1,
            )
            attention_mask = torch.cat(   # (batch_size, text_len + audio_len + 1)
                [text_attention_mask, extended_audio_attention_mask],
                dim=1,
            )
            text_mask = torch.cat(   # (batch_size, text_len + audio_len + 1)
                [
                    torch.ones_like(text_attention_mask, dtype=bool),
                    torch.zeros_like(extended_audio_attention_mask, dtype=bool),
                ],
                dim=1,
            )
            # set labels
            labels = input_ids.clone()   # (batch_size, text_len + audio_len + 1, num_vq)
            labels[~attention_mask.bool()] = IGNORE_TOKEN_ID

            # (batch_size, text_len + audio_len, 768)
            inputs_embeds = gpt.get_emb(input_ids=input_ids, text_mask=text_mask)

            # (batch_size, text_len + audio_len)
            indices = torch.all(input_ids == SPEAKER_TOKEN_ID, dim=-1)
            for i, speaker in enumerate(speakers):
                inputs_embeds[i, indices[i]] = torch.nn.functional.normalize(
                    speaker_embeds[speaker].to(dtype=inputs_embeds.dtype),
                    p=2.0,
                    dim=-1,
                    eps=1e-12,
                ).unsqueeze(0)
            outputs = gpt.gpt.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch_size, text_len + audio_len + 1, 768)
            text_hidden_states = hidden_states[:, :text_len-1]  # (batch_size, text_len-1, 768)
            audio_hidden_states = hidden_states[:, text_len-1:-1]  # (batch_size, audio_len+1, 768)

            audio_logits = torch.stack(
                [gpt.head_code[i](audio_hidden_states) for i in range(gpt.num_vq)],
                dim=2,
            )  # (batch_size, audio_len+1, num_vq, num_class_audio)
            loss: torch.Tensor = loss_fn(audio_logits.flatten(0, 2), labels[:, text_len:].flatten(0, 2))  # audio loss
            if train_text:
                text_logits: torch.Tensor = gpt.head_text(text_hidden_states)  # (batch_size, text_len-1, num_class_text)
                loss += loss_fn(text_logits.flatten(0, 1), labels[:, 1:text_len, 0].flatten(0, 1))  # text loss

            gpt_gen_mel_specs = decoder_decoder(audio_hidden_states[:, :-1].transpose(1, 2)).transpose(1, 2)
            loss += 0.01 * torch.nn.functional.mse_loss(
                gpt_gen_mel_specs,
                audio_mel_specs,
            )

            optimizer.zero_grad()
            loss.backward()
            if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT]:
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            optimizer.step()
        lr_scheduler.step()
    optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    parser.add_argument('--data_path', type=str, default='dummy_data/xz_list_style/speaker_A.list', help='the data_path')
    parser.add_argument(
        '--train_module', type=str, default='gpt',
        choices=['gpt_speaker', 'gpt', 'speaker', 'autoencoder', 'encoder', 'decoder'],
    )
    parser.add_argument(
        '--decoder_type', type=str, default='decoder',
        choices=['decoder', 'dvae'],
    )
    parser.add_argument('--train_text', action='store_true', help='train text loss')
    parser.add_argument('--gpt_lora', action='store_true', help='train gpt with lora')
    parser.add_argument('--gpt_kbit', type=int, default=16, help='train gpt with kbit')
    parser.add_argument('--decoder_encoder_path', type=str)
    parser.add_argument('--decoder_decoder_path', type=str)
    parser.add_argument('--dvae_encoder_path', type=str)
    parser.add_argument('--dvae_decoder_path', type=str)
    parser.add_argument('--gpt_path', type=str)
    parser.add_argument('--speaker_embeds_path', type=str)
    parser.add_argument('--save_folder', type=str, default='./')
    args = parser.parse_args()
    local_path: str | None = args.local_path
    data_path: str = args.data_path
    train_module: TrainModule = args.train_module
    decoder_type: DecoderType = args.decoder_type
    train_text: bool = args.train_text
    gpt_lora: bool = args.gpt_lora
    gpt_kbit: int = args.gpt_kbit
    save_folder: str = args.save_folder

    decoder_encoder_path: str = args.decoder_encoder_path
    decoder_decoder_path: str = args.decoder_decoder_path
    dvae_encoder_path: str = args.dvae_encoder_path
    dvae_decoder_path: str = args.dvae_decoder_path
    speaker_embeds_path: str = args.speaker_embeds_path

    chat = ChatTTS.Chat()
    if local_path is None:
        chat.load_models()
    else:
        print('local model path:', local_path)
        chat.load_models('local', local_path=local_path)

    dataset = ListFolder(
        root=data_path,
        tokenizer=chat.pretrain_models['tokenizer'],
        vocos_model=chat.pretrain_models['vocos'],
        # device=None,
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )

    decoder_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']
    decoder_encoder: DVAEEncoder = DVAEEncoder(
        **get_encoder_config(decoder_decoder.decoder),
    )
    dvae_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['dvae']
    dvae_encoder: DVAEEncoder = DVAEEncoder(
        **get_encoder_config(dvae_decoder.decoder),
    )

    # load pretrained models
    if decoder_encoder_path is not None:
        decoder_encoder.load_state_dict(torch.load(decoder_encoder_path, map_location='cpu'))
    if decoder_decoder_path is not None:
        decoder_decoder.load_state_dict(torch.load(decoder_decoder_path, map_location='cpu'))
    if dvae_encoder_path is not None:
        dvae_encoder.load_state_dict(torch.load(dvae_encoder_path, map_location='cpu'))
    if dvae_decoder_path is not None:
        dvae_decoder.load_state_dict(torch.load(dvae_decoder_path, map_location='cpu'))
    if speaker_embeds_path is None:
        speaker_embeds: dict[str, torch.Tensor] = {}
    else:
        np_speaker_embeds: dict[str, np.ndarray] = np.load(speaker_embeds_path)
        speaker_embeds = {
            speaker: torch.from_numpy(speaker_embed).to(dataset.device)
            for speaker, speaker_embed in np_speaker_embeds.items()
        }

    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT]:
        gpt: ChatTTS.model.gpt.GPT_wrapper = chat.pretrain_models['gpt']
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
            # gpt.gpt = transformers.LlamaModel.from_pretrained()
            # peft.prepare_model_for_gpt_kbit_training(gpt.gpt)
            lora_config = peft.LoraConfig(r=8, lora_alpha=16)
            gpt.gpt = peft.get_peft_model(gpt.gpt, lora_config)

    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT, TrainModule.SPEAKER]:
        train = functools.partial(
            train_gpt,
            decoder_encoder=decoder_encoder,
            dvae_encoder=dvae_encoder,
            train_text=train_text,
            speaker_embeds=speaker_embeds,
        )
    else:
        if decoder_type == DecoderType.DECODER:
            encoder = decoder_encoder
            decoder = decoder_decoder
        else:
            encoder = dvae_encoder
            decoder = dvae_decoder
        train = functools.partial(train_autoencoder, encoder=encoder, decoder=decoder)
    train(chat=chat, dataset=dataset, train_module=train_module)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    gpt_save_path = os.path.join(save_folder, 'gpt.pth')
    speaker_embeds_save_path = os.path.join(save_folder, 'speaker_embeds.npy')
    decoder_encoder_save_path = os.path.join(save_folder, 'decoder_encoder.pth')
    decoder_decoder_save_path = os.path.join(save_folder, 'decoder_decoder.pth')
    dvae_encoder_save_path = os.path.join(save_folder, 'dvae_encoder.pth')
    dvae_decoder_save_path = os.path.join(save_folder, 'dvae_decoder.pth')
    if train_module in [TrainModule.GPT_SPEAKER, TrainModule.GPT] and gpt_lora:
        gpt.gpt = gpt.gpt.merge_and_unload()
    np_speaker_embeds = {speaker: speaker_embed.detach().cpu().numpy() for speaker, speaker_embed in speaker_embeds.items()}
    match train_module:
        case TrainModule.GPT_SPEAKER:
            torch.save(gpt.state_dict(), gpt_save_path)
            torch.save(np_speaker_embeds, speaker_embeds_save_path)
        case TrainModule.GPT:
            torch.save(gpt.state_dict(), gpt_save_path)
        case TrainModule.SPEAKER:
            torch.save(np_speaker_embeds, speaker_embeds_save_path)
        case TrainModule.AUTOENCODER:
            if decoder_type == DecoderType.DECODER:
                torch.save(decoder_encoder.state_dict(), decoder_encoder_save_path)
                torch.save(decoder_decoder.state_dict(), decoder_decoder_save_path)
            else:
                torch.save(dvae_encoder.state_dict(), dvae_encoder_save_path)
                torch.save(dvae_decoder.state_dict(), dvae_decoder_save_path)
        case TrainModule.ENCODER:
            if decoder_type == DecoderType.DECODER:
                torch.save(decoder_encoder.state_dict(), decoder_encoder_save_path)
            else:
                torch.save(dvae_encoder.state_dict(), dvae_encoder_save_path)
        case TrainModule.DECODER:
            if decoder_type == DecoderType.DECODER:
                torch.save(decoder_decoder.state_dict(), decoder_decoder_save_path)
            else:
                torch.save(dvae_decoder.state_dict(), dvae_decoder_save_path)


if __name__ == '__main__':
    main()
