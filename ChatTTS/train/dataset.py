import os
import json
import tarfile
import io
import logging
import tqdm
import abc
import typing

import torch.utils.data
import torchaudio
from torchvision.datasets.utils import download_url
import transformers

from ChatTTS.norm import Normalizer


class LazyDataType(typing.TypedDict):
    filepath: str
    speaker: str
    lang: str
    text: str


class DataType(LazyDataType):
    text_input_ids: torch.Tensor  # (batch_size, text_len)
    text_attention_mask: torch.Tensor  # (batch_size, text_len)
    waveforms: torch.Tensor  # (batch_size, time)
    waveform_attention_mask: torch.Tensor  # (batch_size, time)


class XzListTarKwargsType(typing.TypedDict):
    tokenizer: typing.NotRequired[transformers.PreTrainedTokenizer | None]
    normalizer: typing.NotRequired[Normalizer | None]
    speakers: typing.NotRequired[typing.Iterable[str] | None]
    sample_rate: typing.NotRequired[int]
    default_speaker: typing.NotRequired[str | None]
    default_lang: typing.NotRequired[str | None]
    tar_in_memory: typing.NotRequired[bool]
    process_ahead: typing.NotRequired[bool]


class AudioFolder(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        root: str | io.TextIOWrapper,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        normalizer: Normalizer | None = None,
        speakers: typing.Iterable[str] | None = None,
        sample_rate: int = 24_000,
        default_speaker: str | None = None,
        default_lang: str | None = None,
        tar_path: str | None = None,
        tar_in_memory: bool = False,
        process_ahead: bool = False,
    ) -> None:
        self.root = root
        self.sample_rate = sample_rate
        self.default_speaker = default_speaker
        self.default_lang = default_lang

        self.logger = logging.getLogger(__name__)
        self.normalizer = normalizer
        self.tokenizer = tokenizer

        # tar -cvf ../Xz.tar *
        # tar -xf Xz.tar -C ./Xz
        self.tar_path = tar_path
        self.tar_file = None
        self.tar_io = None
        if tar_path is not None:
            if tar_in_memory:
                with open(tar_path, "rb") as f:
                    self.tar_io = io.BytesIO(f.read())
                self.tar_file = tarfile.open(fileobj=self.tar_io)
            else:
                self.tar_file = tarfile.open(tar_path)

        self.lazy_data, self.speakers = self.get_lazy_data(root, speakers)

        self.text_input_ids: dict[int, torch.Tensor] = {}
        self.waveforms: dict[int, torch.Tensor] = {}
        if process_ahead:
            print("Processing data ...")
            for n, item in enumerate(tqdm.tqdm(self.lazy_data)):
                self.waveforms[n] = self.preprocess_audio(item["filepath"])
                self.text_input_ids[n] = self.preprocess_text(
                    item["text"], item["lang"]
                )
            if self.tar_file is not None:
                self.tar_file.close()
            if self.tar_io is not None:
                self.tar_io.close()

    @abc.abstractmethod
    def get_raw_data(self, root: str | io.TextIOWrapper) -> list[dict[str, str]]: ...

    @staticmethod
    @abc.abstractmethod
    def save_config(
        save_path: str, lazy_data: list[LazyDataType], rel_path: str = "./"
    ) -> None: ...

    def __len__(self):
        return len(self.lazy_data)

    def __getitem__(self, n: int) -> DataType:
        lazy_data = self.lazy_data[n]
        if n in self.waveforms:
            waveforms = self.waveforms[n]
            text_input_ids = self.text_input_ids[n]
        else:
            waveforms = self.preprocess_audio(lazy_data["filepath"])
            text_input_ids = self.preprocess_text(lazy_data["text"], lazy_data["lang"])
            self.waveforms[n] = waveforms
            self.text_input_ids[n] = text_input_ids
            if len(self.waveforms) == len(self.lazy_data):
                if self.tar_file is not None:
                    self.tar_file.close()
                if self.tar_io is not None:
                    self.tar_io.close()
        text_attention_mask = torch.ones(
            len(text_input_ids), device=text_input_ids.device
        )
        waveform_attention_mask = torch.ones(len(waveforms), device=waveforms.device)
        return {
            "filepath": lazy_data["filepath"],
            "speaker": lazy_data["speaker"],
            "lang": lazy_data["lang"],
            "text": lazy_data["text"],
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "waveforms": waveforms,
            "waveform_attention_mask": waveform_attention_mask,
        }

    def get_lazy_data(
        self,
        root: str | io.TextIOWrapper,
        speakers: typing.Iterable[str] | None = None,
    ) -> tuple[list[LazyDataType], set[str]]:
        if speakers is not None:
            new_speakers = set(speakers)
        else:
            new_speakers = set()
        lazy_data = []

        raw_data = self.get_raw_data(root)
        folder_path = os.path.dirname(root) if isinstance(root, str) else ""
        for item in raw_data:
            if "speaker" not in item:
                item["speaker"] = self.default_speaker
            if "lang" not in item:
                item["lang"] = self.default_lang

            if speakers is not None and item["speaker"] not in speakers:
                continue
            if speakers is None and item["speaker"] not in new_speakers:
                new_speakers.add(item["speaker"])
            if self.tar_file is None and isinstance(root, str):
                filepath = os.path.join(folder_path, item["filepath"])
            else:
                filepath = item["filepath"]
            lazy_data.append(
                {
                    "filepath": filepath,
                    "speaker": item["speaker"],
                    "lang": item["lang"].lower(),
                    "text": item["text"],
                }
            )
        return lazy_data, new_speakers

    def preprocess_text(
        self,
        text: str,
        lang: str,
        do_text_normalization: bool = True,
        do_homophone_replacement: bool = True,
    ) -> torch.Tensor:

        text = self.normalizer(
            text,
            do_text_normalization,
            do_homophone_replacement,
            lang,
        )

        text = f"[Stts][spk_emb]{text}[Ptts]"
        # text = f'[Stts][empty_spk]{text}[Ptts]'

        text_token = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return text_token["input_ids"].squeeze(0)

    def preprocess_audio(self, filepath: str) -> torch.Tensor:
        if self.tar_file is not None:
            file = self.tar_file.extractfile(filepath)
            waveforms, sample_rate = torchaudio.load(file)
        else:
            waveforms, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.sample_rate:
            waveforms = torchaudio.functional.resample(
                waveforms,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
        # (channel, time)
        return waveforms.mean(0)  # (time,)


class JsonFolder(AudioFolder):
    """
    In json file, each item is formatted as following example:
    `{"filepath": "path/to/file.wav", "speaker": "John", "lang": "ZH", "text": "Hello"}`.

    filepath is relative to the dirname of root json file.
    """

    def get_raw_data(self, root: str | io.TextIOWrapper) -> list[dict[str, str]]:
        root = open(root, "r", encoding="utf-8") if isinstance(root, str) else root
        raw_data = json.load(root)
        root.close()
        return raw_data

    @staticmethod
    def save_config(
        save_path: str, lazy_data: list[LazyDataType], rel_path: str = "./"
    ) -> None:
        save_data = [item.copy() for item in lazy_data]
        for item in save_data:
            item["filepath"] = os.path.relpath(item["filepath"], rel_path)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)


class ListFolder(AudioFolder):
    """
    In list file, each row is formatted as `filepath|speaker|lang|text` with `|` as separator.
    `path/to/file.wav|John|ZH|Hello`.

    filepath is relative to the dirname of root list file.
    """

    def get_raw_data(self, root: str | io.TextIOWrapper) -> list[dict[str, str]]:
        raw_data = []
        root = open(root, "r", encoding="utf-8") if isinstance(root, str) else root
        for line in root.readlines():
            line = line.strip().removesuffix("\n")
            if len(line) == 0:
                continue
            filepath, speaker, lang, text = line.split(sep="|", maxsplit=3)
            raw_data.append(
                {
                    "text": text,
                    "filepath": filepath,
                    "speaker": speaker,
                    "lang": lang,
                }
            )
        root.close()
        return raw_data

    @staticmethod
    def save_config(
        save_path: str, lazy_data: list[LazyDataType], rel_path: str = "./"
    ) -> None:
        save_data = [item.copy() for item in lazy_data]
        for item in save_data:
            item["filepath"] = os.path.relpath(item["filepath"], rel_path)
        with open(save_path, "w", encoding="utf-8") as f:
            for item in save_data:
                f.write(
                    f"{item['filepath']}|{item['speaker']}|{item['lang']}|{item['text']}\n"
                )


class XzListTar(ListFolder):
    """
    from torchvision.datasets.utils import download_url
    download_url('https://drive.google.com/file/d/1vv73kAHiKb4KiL_oIH4DOWzUoaTeKzt_', './', 'Xz.tar', md5='47683c253d10250d9c32c964118c2b7c')
    """  # noqa: E501

    url = "https://drive.google.com/file/d/1vv73kAHiKb4KiL_oIH4DOWzUoaTeKzt_"
    md5 = "47683c253d10250d9c32c964118c2b7c"

    def __init__(
        self,
        *args,
        root: str | io.TextIOWrapper,
        tar_path: str | None = None,
        **kwargs: typing.Unpack[XzListTarKwargsType],
    ):
        if isinstance(root, str):
            # make sure root is a list file
            if not root.endswith(".list"):  # folder case
                if os.path.isfile(root):
                    raise FileExistsError(f"{root} is a file!")
                elif not os.path.exists(root):
                    os.makedirs(root)
                root = os.path.join(root, "all.list")
            # make sure tar_path is a tar file
            if tar_path is None:
                dirname = os.path.dirname(root)
                assert dirname
                tar_path = os.path.join(dirname, "Xz.tar")
            elif not tar_path.endswith(".tar"):  # folder case
                if os.path.isfile(tar_path):
                    raise FileExistsError(f"{tar_path} is a file!")
                elif not os.path.exists(tar_path):
                    os.makedirs(tar_path)
                tar_path = os.path.join(tar_path, "Xz.tar")
        else:
            assert tar_path is not None
        # download tar file if not exists
        if not os.path.isfile(tar_path):
            dirname, basename = os.path.split(tar_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            download_url(self.url, dirname, basename, md5=self.md5)
            self.prepare_all_list()  # prepare all.list
        if isinstance(root, str) and not os.path.isfile(root):
            # if root is all.list, make sure it is prepared
            if not root.endswith("all.list"):
                with tarfile.open(tar_path) as tar_file:
                    root_str = tar_file.extractfile(root).read().decode("utf-8")
                    root = io.StringIO(root_str)
            else:
                self.prepare_all_list(tar_path=tar_path)  # prepare all.list

        super().__init__(root, *args, tar_path=tar_path, **kwargs)

    @staticmethod
    def prepare_all_list(
        tar_path: str,
        save_folder: str | None = None,
        langs: list[str] = ["zh", "en"],
    ) -> None:
        if save_folder is None:
            save_folder = os.path.dirname(tar_path)
        if os.path.isfile(save_folder):
            raise FileExistsError(f"{save_folder} already exists as a file!")
        elif not os.path.exists(save_folder):
            os.makedirs(save_folder)
        lazy_data = []

        with tarfile.open(tar_path) as tar_file:
            for member in tar_file.getmembers():
                if not member.isfile():
                    continue
                if member.name.endswith(".list"):
                    print(member.name)
                    root_io = io.TextIOWrapper(tar_file.extractfile(member))
                    lazy_data += ListFolder(root_io).lazy_data
                if member.name.endswith(".json"):
                    print(member.name)
                    root_io = io.TextIOWrapper(tar_file.extractfile(member))
                    lazy_data += JsonFolder(root_io).lazy_data
        if langs is not None:
            lazy_data = [item for item in lazy_data if item["lang"] in langs]
        ListFolder.save_config(os.path.join(save_folder, "all.list"), lazy_data)
        JsonFolder.save_config(os.path.join(save_folder, "all.json"), lazy_data)
        print(f"all.list and all.json are saved to {save_folder}")


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
            item["filepath"] = os.path.join(
                os.path.basename(root).removesuffix(".list"),
                os.path.basename(item["filepath"]),
            )
        return raw_data


class AudioCollator:
    def __init__(self, text_pad: int = 0, audio_pad: int = 0):
        self.text_pad = text_pad
        self.audio_pad = audio_pad

    def __call__(self, batch: list[DataType]):
        batch = [x for x in batch if x is not None]

        audio_maxlen = max(len(item["waveforms"]) for item in batch)
        text_maxlen = max(len(item["text_input_ids"]) for item in batch)

        filepath = []
        speaker = []
        lang = []
        text = []
        text_input_ids = []
        text_attention_mask = []
        waveforms = []
        waveform_attention_mask = []

        for x in batch:
            filepath.append(x["filepath"])
            speaker.append(x["speaker"])
            lang.append(x["lang"])
            text.append(x["text"])
            text_input_ids.append(
                torch.nn.functional.pad(
                    x["text_input_ids"],
                    (text_maxlen - len(x["text_attention_mask"]), 0),
                    value=self.text_pad,
                )
            )
            text_attention_mask.append(
                torch.nn.functional.pad(
                    x["text_attention_mask"],
                    (text_maxlen - len(x["text_attention_mask"]), 0),
                    value=0,
                )
            )
            waveforms.append(
                torch.nn.functional.pad(
                    x["waveforms"],
                    (0, audio_maxlen - len(x["waveform_attention_mask"])),
                    value=self.audio_pad,
                )
            )
            waveform_attention_mask.append(
                torch.nn.functional.pad(
                    x["waveform_attention_mask"],
                    (0, audio_maxlen - len(x["waveform_attention_mask"])),
                    value=0,
                )
            )
        return {
            "filepath": filepath,
            "speaker": speaker,
            "lang": lang,
            "text": text,
            "text_input_ids": torch.stack(text_input_ids),
            "text_attention_mask": torch.stack(text_attention_mask),
            "waveforms": torch.stack(waveforms),
            "waveform_attention_mask": torch.stack(waveform_attention_mask),
        }


def formalize_xz_list(src_folder: str):
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".list"):
                filepath = os.path.join(root, file)
                print(filepath)
                lazy_data = XzListFolder(filepath).lazy_data
                XzListFolder.save_config(filepath, lazy_data, rel_path=src_folder)


def prepare_all_list(
    src_folder: str, save_folder: str | None = None, langs: list[str] = ["zh", "en"]
) -> None:
    if save_folder is None:
        save_folder = src_folder
    if os.path.isfile(save_folder):
        raise FileExistsError(f"{save_folder} already exists as a file!")
    elif not os.path.exists(save_folder):
        os.makedirs(save_folder)
    lazy_data = []
    same_folder = os.path.samefile(src_folder, save_folder)
    for root, _, files in os.walk(src_folder):
        for file in files:
            filepath = os.path.join(root, file)
            if same_folder and file in ("all.list", "all.json"):
                continue
            if file.endswith(".list"):
                print(filepath)
                lazy_data += ListFolder(filepath).lazy_data
            elif file.endswith(".json"):
                print(filepath)
                lazy_data += JsonFolder(filepath).lazy_data
    if langs is not None:
        lazy_data = [item for item in lazy_data if item["lang"] in langs]
    ListFolder.save_config(
        os.path.join(save_folder, "all.list"), lazy_data, rel_path=save_folder
    )
    JsonFolder.save_config(
        os.path.join(save_folder, "all.json"), lazy_data, rel_path=save_folder
    )
    print(f"all.list and all.json are saved to {save_folder}")
