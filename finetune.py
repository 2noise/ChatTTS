# python finetune.py --data_path data/Bekki.list --train_module autoencoder

import argparse
import functools
from enum import StrEnum
from tqdm import tqdm

import torch.utils.data
import torch.nn
import transformers
from transformers.trainer_pt_utils import LabelSmoother

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import XzListFolder, AudioFolder, AudioCollator
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
        train_module: TrainModule = TrainModule.AUTOENCODER,
        decoder_type: str = DecoderType.DECODER,
):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']
    decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models[decoder_type]
    encoder: DVAEEncoder = DVAEEncoder(**get_encoder_config(decoder))

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

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id))
    for epoch in range(10):
        for batch in tqdm(loader):
            audio_mel_specs: torch.Tensor = batch['audio_mel_specs']  # (batch_size, audio_len*2, 100)
            audio_attention_mask: torch.Tensor = batch['audio_attention_mask']  # (batch_size, audio_len)
            mel_attention_mask = audio_attention_mask.unsqueeze(-1).repeat(1, 1, 2)  # (batch_size, audio_len*2)

            # (batch_size, audio_len, audio_dim)
            audio_latents = encoder(audio_mel_specs, audio_attention_mask) * audio_attention_mask.unsqueeze(-1)
            # (batch_size, audio_len*2, 100)
            gen_mel_specs = decoder(audio_latents.transpose(1, 2)).transpose(1, 2) * mel_attention_mask.unsqueeze(-1)

            loss: torch.Tensor = loss_fn(gen_mel_specs, audio_mel_specs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    optimizer.zero_grad()


def train_gpt(chat: ChatTTS.Chat, dataset: AudioFolder, train_module: TrainModule = TrainModule.GPT_SPEAKER, train_text: bool = True):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']

    decoder_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']
    decoder_decoder.eval().requires_grad_(False)
    decoder_encoder: DVAEEncoder = DVAEEncoder(**get_encoder_config(decoder_decoder))
    decoder_encoder.eval().requires_grad_(False)

    dvae_decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['dvae']
    dvae_decoder.eval().requires_grad_(False)
    dvae_encoder: DVAEEncoder = DVAEEncoder(**get_encoder_config(dvae_decoder))
    dvae_encoder.eval().requires_grad_(False)
    dvae_vq: ChatTTS.model.dvae.GFSQ = dvae_decoder.vq_layer

    gpt: ChatTTS.model.gpt.GPT_warpper = chat.pretrain_models['gpt']
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
    }
    SPEAKER_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[spk_emb]')
    AUDIO_EOS_TOKEN_ID: int = tokenizer.convert_tokens_to_ids('[Etts]')
    PAD_TOKEN_ID: int = tokenizer.pad_token_id

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

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id))
    for epoch in range(10):
        for batch in tqdm(loader):
            speakers: list[str] = batch['speaker']  # (batch_size,)
            text_input_ids: torch.Tensor = batch['text_input_ids']   # (batch_size, text_len)
            text_attention_mask: torch.Tensor = batch['text_attention_mask']   # (batch_size, text_len)
            audio_mel_specs: torch.Tensor = batch['audio_mel_specs']   # (batch_size, audio_len*2, 100)
            audio_attention_mask: torch.Tensor = batch['audio_attention_mask']   # (batch_size, audio_len)

            batch_size, text_len = text_attention_mask.size()

            dvae_audio_latents = dvae_encoder(audio_mel_specs, audio_attention_mask)  # (batch_size, audio_len, audio_dim=1024)
            _, dvae_audio_input_ids = quantize(dvae_vq, dvae_audio_latents)  # (batch_size, audio_len, num_vq)
            dvae_audio_input_ids[~audio_attention_mask] = PAD_TOKEN_ID

            # add audio eos token
            extended_audio_attention_mask = torch.cat([
                audio_attention_mask,
                torch.zeros(
                    (batch_size, 1),
                    dtype=audio_attention_mask.dtype,
                    device=audio_attention_mask.device,
                ),
            ])  # (batch_size, audio_len+1)
            extended_audio_input_ids = torch.cat([
                dvae_audio_input_ids,
                PAD_TOKEN_ID * torch.ones(
                    (batch_size, 1, gpt.num_vq),
                    dtype=dvae_audio_input_ids.dtype,
                    device=dvae_audio_input_ids.device,
                ),
            ])  # (batch_size, audio_len+1, num_vq)
            indices = audio_attention_mask.sum(dim=1)  # (batch_size,)
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
            labels[~attention_mask] = IGNORE_TOKEN_ID

            # (batch_size, text_len + audio_len, 768)
            inputs_embeds = gpt.get_emb(input_ids=input_ids, text_mask=text_mask)

            # (batch_size, text_len + audio_len)
            indices = torch.all(input_ids == SPEAKER_TOKEN_ID, dim=-1)
            for i, speaker in enumerate(speakers):
                inputs_embeds[i, indices[i]] = torch.nn.functional.normalize(
                    speaker_embeds[speaker].to(dtype=inputs_embeds.dtype),
                    p=2.0,
                    dim=1,
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

            gpt_gen_mel_specs = decoder_decoder(audio_hidden_states.transpose(1, 2)).transpose(1, 2)
            loss += 0.01 * torch.nn.functional.mse_loss(
                gpt_gen_mel_specs,
                audio_mel_specs,
            )

            optimizer.zero_grad()
            loss.backward()
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
        '--decoder_type', type=str, default='gpt',
        choices=['decoder', 'dvae'],
    )
    parser.add_argument('--train_text', action='store_true', help='train text loss')
    args = parser.parse_args()
    local_path: str | None = args.local_path
    data_path: str = args.data_path
    train_module: TrainModule = args.train_module
    train_text: bool = args.train_text

    chat = ChatTTS.Chat()
    if local_path is None:
        chat.load_models()
    else:
        print('local model path:', local_path)
        chat.load_models('local', local_path=local_path)

    dataset = XzListFolder(
        root=data_path,
        tokenizer=chat.pretrain_models['tokenizer'],
        vocos_model=chat.pretrain_models['vocos'],
        # device=None,
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )
    if train_module in [TrainModule.SPEAKER, TrainModule.GPT, TrainModule.GPT_SPEAKER]:
        train = functools.partial(train_gpt, train_text=train_text)
    else:
        train = train_autoencoder
    train(chat=chat, dataset=dataset, train_module=train_module)


if __name__ == '__main__':
    main()
