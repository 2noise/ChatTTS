import os
import functools
import json
import logging
import abc
import typing

import torch.utils.data
import torchaudio
import transformers
import vocos

from ChatTTS.utils.infer_utils import count_invalid_characters, detect_language, apply_character_map, apply_half2full_map


class LazyDataType(typing.TypedDict):
    filepath: str
    speaker: str
    lang: str
    text: str


class DataType(LazyDataType):
    text_input_ids: torch.Tensor    # (batch_size, text_len)
    text_attention_mask: torch.Tensor   # (batch_size, text_len)
    audio_mel_specs: torch.Tensor    # (batch_size, audio_len*2, 100)
    audio_attention_mask: torch.Tensor  # (batch_size, audio_len)


class AudioFolder(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        root: str,
        tokenizer: transformers.PreTrainedTokenizer,
        vocos_model: vocos.Vocos,
        device: str | torch.device | None = None,
        speakers: typing.Iterable[str] | None = None,
        sample_rate: int = 24_000,
        default_speaker: str = None,
        default_lang: str = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.default_speaker = default_speaker
        self.default_lang = default_lang

        self.logger = logging.getLogger(__name__)
        self.normalizer = {}

        self.tokenizer = tokenizer
        self.vocos = vocos_model
        self.device = device or next(self.vocos.parameters()).device

        self.lazy_data, self.speakers = self.get_lazy_data(root, speakers)

        self.text_input_ids = [self.preprocess_text(item['text'], item['lang']) for item in self.lazy_data]
        self.audio_mel_specs = [self.preprocess_audio(item['filepath']) for item in self.lazy_data]

    @abc.abstractmethod
    def get_raw_data(self, root: str) -> list[dict[str, str]]:
        ...

    def __len__(self):
        return len(self.lazy_data)

    def __getitem__(self, n: int) -> DataType:
        text_input_ids = self.text_input_ids[n]
        audio_mel_specs = self.audio_mel_specs[n]
        text_attention_mask = torch.ones(len(text_input_ids), device=text_input_ids.device)
        audio_attention_mask = torch.ones(len(audio_mel_specs) // 2, device=audio_mel_specs.device)
        return {
            'filepath': self.lazy_data[n]['filepath'],
            'speaker': self.lazy_data[n]['speaker'],
            'lang': self.lazy_data[n]['lang'],
            'text': self.lazy_data[n]['text'],
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'audio_mel_specs': audio_mel_specs,
            'audio_attention_mask': audio_attention_mask,
        }

    def get_lazy_data(
        self,
        root: str,
        speakers: typing.Iterable[str] | None = None,
    ) -> tuple[list[LazyDataType], set[str]]:
        if speakers is not None:
            new_speakers = set(speakers)
        else:
            new_speakers = set()
        lazy_data = []

        raw_data = self.get_raw_data(root)
        folder_path = os.path.dirname(root)
        for item in raw_data:
            if 'speaker' not in item:
                item['speaker'] = self.default_speaker
            if 'lang' not in item:
                item['lang'] = self.default_lang

            if speakers is not None and item['speaker'] not in speakers:
                continue
            if speakers is None and item['speaker'] not in new_speakers:
                new_speakers.add(item['speaker'])
            lazy_data.append({
                'filepath': os.path.join(folder_path, item['filepath']),
                'speaker': item['speaker'],
                'lang': item['lang'].lower(),
                'text': item['text'],
            })
        return lazy_data, new_speakers

    def preprocess_text(
        self,
        text: str,
        lang: str,
        do_text_normalization: bool = True,
    ) -> torch.Tensor:
        if do_text_normalization:
            _lang = lang or detect_language(text)
            self.init_normalizer(_lang)
            text = self.normalizer[_lang](text)
            if _lang == 'zh':
                text = apply_half2full_map(text)

        invalid_characters = count_invalid_characters(text)
        if len(invalid_characters):
            self.logger.log(logging.WARNING, f'Invalid characters found! : {invalid_characters}')
            text = apply_character_map(text)

        # if not skip_refine_text:
        #     text_tokens = refine_text(self.pretrain_models, text, **params_refine_text)['ids']
        #     text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
        #     text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
        #     if refine_text_only:
        #         return text

        text = f'[Stts][spk_emb]{text}[Ptts]'
        # text = f'[Stts][empty_spk]{text}[Ptts]'

        text_token = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).to(device=self.device)
        return text_token['input_ids'].squeeze(0)

    def preprocess_audio(self, filepath: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        mel_spec: torch.Tensor = self.vocos.feature_extractor(waveform.to(device=self.device))
        return mel_spec.squeeze(0).transpose(0, 1)  # (audio_len*2, 100)

    def init_normalizer(self, lang: str):
        if lang not in self.normalizer:
            if lang == 'zh':
                from tn.chinese.normalizer import Normalizer
                self.normalizer[lang] = Normalizer().normalize
            else:
                from nemo_text_processing.text_normalization.normalize import Normalizer
                self.normalizer[lang] = functools.partial(
                    Normalizer(input_case='cased', lang=lang).normalize,
                    verbose=False,
                    punct_post_process=True,
                )


class JsonFolder(AudioFolder):
    """
    In json file, each item is formatted as following example:
    `{"filepath": "path/to/file.wav", "speaker": "John", "lang": "ZH", "text": "Hello"}`.

    filepath is relative to the dirname of root json file.
    """

    def get_raw_data(self, root: str) -> list[dict[str, str]]:
        with open(root, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return raw_data


class ListFolder(AudioFolder):
    """
    In list file, each row is formatted as `filepath|speaker|lang|text` with `|` as separator.
    `path/to/file.wav|John|ZH|Hello`.

    filepath is relative to the dirname of root list file.
    """

    def get_raw_data(self, root: str) -> list[dict[str, str]]:
        raw_data = []
        with open(os.path.join(root), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                filepath, speaker, lang, text = line.split(sep='|', maxsplit=3)
                text = text.strip().removesuffix('\n')
                raw_data.append({
                    'text': text,
                    'filepath': filepath,
                    'speaker': speaker,
                    'lang': lang,
                })
        return raw_data


class XzListFolder(ListFolder):
    """
    [Xz乔希](https://space.bilibili.com/5859321)

    Only look at the basename of filepath in list file. Previous folder paths are ignored.
    Files are organized as `[list basename]/[file basename]`

    Example tree structure:

    [folder]
    ├── speaker_A
    │   ├── 1.wav
    │   └── 2.wav
    ├── speaker_A.list
    ├── speaker_B
    │   ├── 1.wav
    │   └── 2.wav
    └── speaker_B.list
    """

    def get_raw_data(self, root: str) -> list[dict[str, str]]:
        raw_data = super().get_raw_data(root)
        for item in raw_data:
            item['filepath'] = os.path.join(
                os.path.basename(root).removesuffix('.list'),
                os.path.basename(item['filepath']),
            )
        return raw_data


class AudioCollator:
    def __init__(self, text_pad: int = 0, audio_pad: int = 0):
        self.text_pad = text_pad
        self.audio_pad = audio_pad

    def __call__(self, batch: list[DataType]):
        batch = [x for x in batch if x is not None]

        audio_maxlen = (max(len(item['audio_attention_mask']) for item in batch) + 1) // 2
        text_maxlen = max(len(item['text_attention_mask']) for item in batch)

        file_path = []
        speaker = []
        lang = []
        text = []
        text_input_ids = []
        text_attention_mask = []
        audio_mel_specs = []
        audio_attention_mask = []

        for x in batch:
            file_path.append(x['filepath'])
            speaker.append(x['speaker'])
            lang.append(x['lang'])
            text.append(x['text'])
            text_input_ids.append(
                torch.nn.functional.pad(
                    x['text_input_ids'],
                    (text_maxlen - len(x['text_input_ids']), 0),
                    value=self.text_pad,
                )
            )
            text_attention_mask.append(
                torch.nn.functional.pad(
                    x['text_attention_mask'],
                    (text_maxlen - len(x['text_attention_mask']), 0),
                    value=0,
                )
            )
            audio_mel_specs.append(
                torch.nn.functional.pad(
                    x['audio_mel_specs'],
                    (0, 0, 0, audio_maxlen*2 - len(x['audio_mel_specs'])),
                    value=self.audio_pad,
                )
            )
            audio_attention_mask.append(
                torch.nn.functional.pad(
                    x['audio_attention_mask'],
                    (0, audio_maxlen - len(x['audio_attention_mask'])),
                    value=0,
                )
            )
        return {
            'filepath': file_path,
            'speaker': speaker,
            'lang': lang,
            'text': text,
            'text_input_ids': torch.stack(text_input_ids),
            'text_attention_mask': torch.stack(text_attention_mask),
            'audio_mel_specs': torch.stack(audio_mel_specs),
            'audio_attention_mask': torch.stack(audio_attention_mask),
        }
