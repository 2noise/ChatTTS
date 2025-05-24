from enum import StrEnum

import torch
import torch.nn.functional
import torch.utils.data

import ChatTTS
from ChatTTS.utils.ansi import ansi, get_ansi_len, output_iter
from ChatTTS.utils.log import MetricLogger

from .dataset import AudioFolder, AudioCollator
from .utils import (
    get_mel_specs,
    get_mel_attention_mask,
    get_dvae_mel_specs,
    get_hidden_states_and_labels,
)


class TrainModule(StrEnum):
    GPT_ALL = "gpt_all"  # GPT + SPEAKER + DECODER
    GPT_SPEAKER = "gpt_speaker"  # GPT + SPEAKER

    GPT = "gpt"
    DECODER = "decoder"
    SPEAKER = "speaker"

    DVAE = "dvae"
    DVAE_ENCODER = "dvae_encoder"
    DVAE_DECODER = "dvae_decoder"


def train_autoencoder(
    chat: ChatTTS.Chat,
    dataset: AudioFolder,
    train_module: TrainModule = TrainModule.DVAE,
    batch_size: int = 10,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_norm_clip: float = 1.0,
    validate: bool = False,
):
    chat.dvae.eval().requires_grad_(False)
    if not validate:
        match train_module:
            case TrainModule.DVAE:
                train_params = list(chat.dvae.parameters())
            case TrainModule.DVAE_ENCODER:
                train_params = []
                train_params += list(chat.dvae.downsample_conv.parameters())
                train_params += list(chat.dvae.encoder.parameters())
                train_params += list(chat.dvae.vq_layer.parameters())
            case TrainModule.DVAE_DECODER:
                train_params = []
                train_params += list(chat.dvae.decoder.parameters())
                train_params += list(chat.dvae.out_conv.parameters())
        optimizer = torch.optim.AdamW(train_params, lr=lr, betas=[0.8, 0.99], eps=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, 1e-7
        )
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)

        def activate_params():
            match train_module:
                case TrainModule.DVAE:
                    chat.dvae.train().requires_grad_()
                case TrainModule.DVAE_ENCODER:
                    chat.dvae.downsample_conv.train().requires_grad_()
                    chat.dvae.encoder.train().requires_grad_()
                    chat.dvae.vq_layer.train().requires_grad_()
                case TrainModule.DVAE_DECODER:
                    chat.dvae.decoder.train().requires_grad_()
                    chat.dvae.out_conv.train().requires_grad_()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioCollator(),
        # num_workers=4,
    )
    logger = MetricLogger()
    logger.create_meters(loss=None)
    if not validate:
        train_autoencoder(chat=chat, dataset=dataset, validate=True)
    for _epoch in range(1 if validate else epochs):
        if not validate:
            activate_params()
        _epoch += 1
        logger.reset()
        if validate:
            header: str = "{blue_light}{0}{reset}".format("AutoEncoder", **ansi)
            header = header.ljust(max(len("AutoEncoder"), 30) + get_ansi_len(header))
        else:
            header: str = "{blue_light}{0}: {1}{reset}".format(
                "Epoch", output_iter(_epoch, epochs), **ansi
            )
            header = header.ljust(max(len("Epoch"), 30) + get_ansi_len(header))
        iterator = logger.log_every(loader, header=header, tqdm_header="Batch")
        for batch in iterator:
            waveforms: torch.Tensor = batch["waveforms"]  # (batch_size, time)
            waveform_attention_mask: torch.Tensor = batch[
                "waveform_attention_mask"
            ]  # (batch_size, time)

            waveforms = waveforms.to(chat.device, non_blocking=True)
            waveform_attention_mask = waveform_attention_mask.to(
                chat.device, non_blocking=True
            )

            mel_specs = get_mel_specs(chat, waveforms)  # (batch_size, 100, mel_len)
            mel_attention_mask = get_mel_attention_mask(
                waveform_attention_mask, mel_len=mel_specs.size(2)
            )  # (batch_size, mel_len)
            mel_specs = mel_specs * mel_attention_mask.unsqueeze(1)  # clip

            dvae_mel_specs = get_dvae_mel_specs(
                chat, mel_specs, mel_attention_mask
            )  # (batch_size, 100, mel_len)
            dvae_mel_specs = dvae_mel_specs * mel_attention_mask.unsqueeze(1)  # clip

            loss = torch.nn.functional.mse_loss(dvae_mel_specs, mel_specs)
            logger.meters["loss"].update(loss.item(), n=len(waveform_attention_mask))
            if not validate:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, grad_norm_clip)
                optimizer.step()
        if not validate:
            lr_scheduler.step()
            train_autoencoder(chat=chat, dataset=dataset, validate=True)
    if not validate:
        optimizer.zero_grad()


def train_gpt(
    chat: ChatTTS.Chat,
    dataset: AudioFolder,
    train_module: TrainModule = TrainModule.GPT_ALL,
    batch_size: int = 10,
    epochs: int = 10,
    grad_norm_clip: float = 1.0,
    speaker_embeds: dict[str, torch.Tensor] = {},
    train_text: bool = False,
    validate: bool = False,
) -> dict[str, torch.Tensor]:
    for speaker in dataset.speakers:
        if speaker not in speaker_embeds:
            speaker_embeds[speaker] = chat.speaker._sample_random().to(
                device=chat.device
            )

    chat.dvae.eval().requires_grad_(False)
    chat.gpt.eval().requires_grad_(False)
    chat.decoder.eval().requires_grad_(False)

    if not validate:
        train_speaker = train_module in [
            TrainModule.GPT_ALL,
            TrainModule.GPT_SPEAKER,
            TrainModule.SPEAKER,
        ]
        match train_module:
            case TrainModule.GPT_ALL:
                train_params = []
                train_params += list(speaker_embeds.values())
                train_params += list(chat.gpt.parameters())
                train_params += list(chat.decoder.parameters())
                optimizer = torch.optim.Adam(
                    chat.gpt.parameters(), lr=1e-5, weight_decay=0, betas=[0.9, 0.95]
                )
                optimizer.add_param_group(
                    {
                        "params": chat.decoder.parameters(),
                        "lr": 1e-5,
                        "weight_decay": 0,
                        "betas": [0.9, 0.95],
                    }
                )
                optimizer.add_param_group(
                    {
                        "params": speaker_embeds.values(),
                        "lr": 1e-2,
                        "weight_decay": 0,
                        "betas": [0.9, 0.95],
                    }
                )
            case TrainModule.GPT_SPEAKER:
                train_params = []
                train_params += list(speaker_embeds.values())
                train_params += list(chat.gpt.parameters())
                optimizer = torch.optim.Adam(
                    chat.gpt.parameters(), lr=1e-5, weight_decay=0, betas=[0.9, 0.95]
                )
                optimizer.add_param_group(
                    {
                        "params": speaker_embeds.values(),
                        "lr": 1e-2,
                        "weight_decay": 0,
                        "betas": [0.9, 0.95],
                    }
                )
            case TrainModule.GPT:
                train_params = list(chat.gpt.parameters())
            case TrainModule.DECODER:
                train_params = list(chat.decoder.parameters())
            case TrainModule.SPEAKER:
                train_params = list(speaker_embeds.values())
                optimizer = torch.optim.Adam(
                    train_params, lr=1e-2, weight_decay=0, betas=[0.9, 0.95]
                )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, 1e-7
        )
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial())

        def activate_params():
            if train_speaker:
                for speaker_embed in speaker_embeds.values():
                    speaker_embed.requires_grad_(True)
            match train_module:
                case TrainModule.GPT_ALL:
                    chat.gpt.train().requires_grad_()
                    chat.decoder.train().requires_grad_()
                case TrainModule.GPT_SPEAKER:
                    chat.gpt.train().requires_grad_()
                case TrainModule.GPT:
                    chat.gpt.train().requires_grad_()
                case TrainModule.DECODER:
                    chat.decoder.train().requires_grad_()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioCollator(),
        # num_workers=4,
    )
    logger = MetricLogger()
    logger.create_meters(audio_loss=None, mse_loss=None)
    if validate or train_text:
        logger.create_meters(text_loss=None)
    if not validate:
        train_gpt(
            chat=chat, dataset=dataset, speaker_embeds=speaker_embeds, validate=True
        )
    for _epoch in range(1 if validate else epochs):
        if not validate:
            activate_params()
        _epoch += 1
        logger.reset()
        if validate:
            header: str = "{blue_light}{0}{reset}".format("GPT", **ansi)
            header = header.ljust(max(len("GPT"), 30) + get_ansi_len(header))
        else:
            header: str = "{blue_light}{0}: {1}{reset}".format(
                "Epoch", output_iter(_epoch, epochs), **ansi
            )
            header = header.ljust(max(len("Epoch"), 30) + get_ansi_len(header))
        iterator = logger.log_every(loader, header=header, tqdm_header="Batch")
        for batch in iterator:
            speakers: list[str] = batch["speaker"]  # (batch_size,)
            text_input_ids: torch.Tensor = batch[
                "text_input_ids"
            ]  # (batch_size, text_len)
            text_attention_mask: torch.Tensor = batch[
                "text_attention_mask"
            ]  # (batch_size, text_len)
            waveforms: torch.Tensor = batch["waveforms"]  # (batch_size, time)
            waveform_attention_mask: torch.Tensor = batch[
                "waveform_attention_mask"
            ]  # (batch_size, time)

            text_input_ids = text_input_ids.to(chat.device, non_blocking=True)
            text_attention_mask = text_attention_mask.to(chat.device, non_blocking=True)
            waveforms = waveforms.to(chat.device, non_blocking=True)
            waveform_attention_mask = waveform_attention_mask.to(
                chat.device, non_blocking=True
            )

            mel_specs = get_mel_specs(chat, waveforms)  # (batch_size, 100, mel_len)
            mel_attention_mask = get_mel_attention_mask(
                waveform_attention_mask, mel_len=mel_specs.size(2)
            )  # (batch_size, mel_len)
            mel_specs = mel_specs * mel_attention_mask.unsqueeze(1)  # clip

            results = get_hidden_states_and_labels(
                chat=chat,
                mel_specs=mel_specs,
                mel_attention_mask=mel_attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                speakers=speakers,
                speaker_embeds=speaker_embeds,
            )
            hidden_states = results["hidden_states"]
            labels = results["labels"]

            text_len = text_input_ids.size(1)
            audio_hidden_states = hidden_states[
                :, text_len - 1 : -1
            ]  # (batch_size, mel_len+1, 768)
            audio_labels = labels[:, text_len:]  # (batch_size, mel_len+1)

            audio_logits = torch.stack(
                [
                    chat.gpt.head_code[i](audio_hidden_states)
                    for i in range(chat.gpt.num_vq)
                ],
                dim=2,
            )  # (batch_size, mel_len+1, num_vq, num_class_audio)
            audio_loss: torch.Tensor = torch.nn.functional.cross_entropy(
                audio_logits.flatten(0, 2), audio_labels.flatten(0, 2)
            )
            loss: torch.Tensor = audio_loss
            if validate or train_text:
                text_hidden_states = hidden_states[
                    :, : text_len - 1
                ]  # (batch_size, text_len-1, 768)
                text_labels = labels[:, 1:text_len, 0]  # (batch_size, text_len-1)

                text_logits: torch.Tensor = chat.gpt.head_text(
                    text_hidden_states
                )  # (batch_size, text_len-1, num_class_text)
                text_loss: torch.Tensor = torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1), text_labels.flatten(0, 1)
                )
                loss = loss + text_loss
                logger.meters["text_loss"].update(text_loss.item(), n=batch_size)

            decoder_mel_specs = chat.decoder(
                audio_hidden_states[:, :-1].transpose(1, 2)
            )
            decoder_mel_specs = decoder_mel_specs * mel_attention_mask.unsqueeze(
                1
            )  # clip
            mse_loss = torch.nn.functional.mse_loss(
                decoder_mel_specs,
                mel_specs,
            )
            loss = loss + 10 * mse_loss
            logger.meters["mse_loss"].update(mse_loss.item(), n=batch_size)
            logger.meters["audio_loss"].update(audio_loss.item(), n=batch_size)

            if not validate:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, grad_norm_clip)
                optimizer.step()
        if not validate:
            lr_scheduler.step()
            train_gpt(
                chat=chat, dataset=dataset, speaker_embeds=speaker_embeds, validate=True
            )
    if not validate:
        optimizer.zero_grad()
    return speaker_embeds
