# python finetune.py --data_path data/Bekki.list --train_module autoencoder

import argparse
from enum import StrEnum
from tqdm import tqdm

import torch.utils.data
import torch.nn
import transformers

import ChatTTS
import ChatTTS.model.gpt
import ChatTTS.model.dvae
from utils.dataset import XzListFolder, AudioFolder, AudioCollator
from utils.model import encode


class TrainModule(StrEnum):
    GPT_SPEAKER = 'gpt_speaker'
    GPT = 'gpt'
    SPEAKER = 'speaker'

    AUTOENCODER = 'autoencoder'
    ENCODER = 'encoder'
    DECODER = 'decoder'


def train_autoencoder(chat: ChatTTS.Chat, dataset: AudioFolder, train_module: TrainModule = TrainModule.AUTOENCODER):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']
    # encoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']   # TODO: placeholder
    decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']

    train_params = list(decoder.parameters())   # TODO: placeholder
    # match train_module:   # TODO: remove comments
    #     case TrainModule.AUTOENCODER:
    #         encoder.train().requires_grad_()
    #         decoder.train().requires_grad_()
    #         train_params = list(encoder.parameters()) + list(decoder.parameters())
    #     case TrainModule.ENCODER:
    #         encoder.train().requires_grad_()
    #         decoder.eval().requires_grad_(False)
    #         train_params = list(encoder.parameters())
    #     case TrainModule.DECODER:
    #         encoder.eval().requires_grad_(False)
    #         decoder.train().requires_grad_()
    #         train_params = list(decoder.parameters())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(train_params, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-6)

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id))
    for epoch in range(10):
        for batch in tqdm(loader):
            audio_mel_specs: torch.Tensor = batch['audio_mel_specs']  # (batch_size, audio_len*2, 100)
            # TODO: do we need to care about the padded parts?
            # audio_quantized_latents shape (batch_size, audio_len, audio_dim)
            audio_quantized_latents, _ = encode(chat, audio_mel_specs)

            # (batch_size, audio_len*2, audio_dim)
            gen_mel_specs = decoder(audio_quantized_latents.transpose(1, 2)).transpose(1, 2)
            loss: torch.Tensor = loss_fn(gen_mel_specs, audio_mel_specs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    optimizer.zero_grad()


def train_gpt(chat: ChatTTS.Chat, dataset: AudioFolder, train_module: TrainModule = TrainModule.GPT_SPEAKER):
    tokenizer: transformers.PreTrainedTokenizer = chat.pretrain_models['tokenizer']
    gpt: ChatTTS.model.gpt.GPT_warpper = chat.pretrain_models['gpt']
    # encoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']   # TODO: placeholder
    decoder: ChatTTS.model.dvae.DVAE = chat.pretrain_models['decoder']

    # encoder.eval().requires_grad_(False)   # TODO: remove comment
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
    SPEAKER_TOKEN: int = tokenizer.convert_tokens_to_ids('[spk_emb]')

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

            text_len = text_attention_mask.size(1)

            # audio_quantized_latents shape (batch_size, audio_len, audio_dim)
            # audio_input_ids shape (batch_size, audio_len, num_vq)
            audio_quantized_latents, audio_input_ids = encode(chat, audio_mel_specs)

            input_ids = torch.cat(   # (batch_size, text_len + audio_len, num_vq)
                [
                    text_input_ids.unsqueeze(-1).repeat(1, 1, gpt.num_vq),   # (batch_size, text_len, num_vq)
                    audio_input_ids,   # (batch_size, audio_len, num_vq)
                ],
                dim=1,
            )
            attention_mask = torch.cat(   # (batch_size, text_len + audio_len)
                [text_attention_mask, audio_attention_mask],
                dim=1,
            )
            text_mask = torch.cat(   # (batch_size, text_len + audio_len)
                [
                    torch.ones_like(text_attention_mask, dtype=bool),
                    torch.zeros_like(audio_attention_mask, dtype=bool),
                ],
                dim=1,
            )

            # (batch_size, text_len + audio_len, 768)
            inputs_embeds = gpt.get_emb(input_ids=input_ids, text_mask=text_mask)

            # (batch_size, text_len + audio_len)
            indices = torch.all(input_ids == SPEAKER_TOKEN, dim=-1)
            for i, speaker in enumerate(speakers):
                inputs_embeds[i, indices[i]] = torch.nn.functional.normalize(
                    speaker_embeds[speaker].to(dtype=inputs_embeds.dtype).unsqueeze(0),
                    p=2.0,
                    dim=1,
                    eps=1e-12,
                )
            outputs = gpt.gpt.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch_size, text_len + audio_len, 768)
            text_hidden_states = hidden_states[:, :text_len-1]  # (batch_size, text_len-1, 768)
            audio_hidden_states = hidden_states[:, text_len-1:-1]  # (batch_size, audio_len, 768)

            text_logits: torch.Tensor = gpt.head_text(text_hidden_states)  # (batch_size, text_len-1, num_class_text)
            audio_logits = torch.stack(
                [gpt.head_code[i](audio_hidden_states) for i in range(gpt.num_vq)],
                dim=3,
            )  # (batch_size, audio_len, num_class_audio)

            text_loss: torch.Tensor = loss_fn(text_logits.flatten(0, 1), text_input_ids[:, 1:].flatten(0, 1))
            audio_loss: torch.Tensor = loss_fn(audio_logits.flatten(0, 1), audio_input_ids.flatten(0, 1))
            loss = text_loss + audio_loss
            if False:   # TODO: A possible loss for "decoder" case ("dvae" case doesn't have this loss term)
                loss += torch.nn.functional.mse_loss(
                    audio_hidden_states,
                    audio_quantized_latents,
                )
                # TODO: an alternative is to measure mel_specs instead of quantized_latents
                # But the question is to measure which 2 mel_specs since there are 3.

                # audio_mel_specs
                # encoder_gen_mel_specs = decoder(audio_quantized_latents.transpose(1, 2)).transpose(1, 2)
                # gpt_gen_mel_specs = decoder(audio_quantized_latents.transpose(1, 2)).transpose(1, 2)

                # loss += torch.nn.functional.mse_loss(
                #     audio_mel_specs,
                #     gpt_gen_mel_specs,
                # )

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
    args = parser.parse_args()
    local_path: str | None = args.local_path
    data_path: str = args.data_path
    train_module: TrainModule = args.train_module

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
    train = train_gpt if train_module in [TrainModule.SPEAKER, TrainModule.GPT, TrainModule.GPT_SPEAKER] else train_autoencoder
    train(chat=chat, dataset=dataset, train_module=train_module)


if __name__ == '__main__':
    main()
