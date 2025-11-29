import json
import logging
import re
from typing import Dict, Tuple, List, Literal, Callable, Optional
import sys

from numba import jit
import numpy as np

from .utils import del_all


@jit(nopython=True)
def _find_index(table: np.ndarray, val: np.uint16):
    for i in range(table.size):
        if table[i] == val:
            return i
    return -1


@jit(nopython=True)
def _fast_replace(
    table: np.ndarray, text: bytes
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    result = np.frombuffer(text, dtype=np.uint16).copy()
    replaced_words = []
    for i in range(result.size):
        ch = result[i]
        p = _find_index(table[0], ch)
        if p >= 0:
            repl_char = table[1][p]
            result[i] = repl_char
            replaced_words.append((chr(ch), chr(repl_char)))
    return result, replaced_words


@jit(nopython=True)
def _split_tags(text: str) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    tags: List[str] = []
    current_text = ""
    current_tag = ""
    for c in text:
        if c == "[":
            texts.append(current_text)
            current_text = ""
            current_tag = c
        elif current_tag != "":
            current_tag += c
        else:
            current_text += c
        if c == "]":
            tags.append(current_tag)
            current_tag = ""
    if current_text != "":
        texts.append(current_text)
    return texts, tags


@jit(nopython=True)
def _combine_tags(texts: List[str], tags: List[str]) -> str:
    text = ""
    for t in texts:
        tg = ""
        if len(tags) > 0:
            tg = tags.pop(0)
        text += t + tg
    return text


class Normalizer:
    def __init__(self, map_file_path: str, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.normalizers: Dict[str, Callable[[str], str]] = {}
        self.homophones_map = self._load_homophones_map(map_file_path)
        """
        homophones_map

        Replace the mispronounced characters with correctly pronounced ones.

        Creation process of homophones_map.json:

        1. Establish a word corpus using the [Tencent AI Lab Embedding Corpora v0.2.0 large] with 12 million entries. After cleaning, approximately 1.8 million entries remain. Use ChatTTS to infer the text.
        2. Record discrepancies between the inferred and input text, identifying about 180,000 misread words.
        3. Create a pinyin to common characters mapping using correctly read characters by ChatTTS.
        4. For each discrepancy, extract the correct pinyin using [python-pinyin] and find homophones with the correct pronunciation from the mapping.

        Thanks to:
        [Tencent AI Lab Embedding Corpora for Chinese and English Words and Phrases](https://ai.tencent.com/ailab/nlp/en/embedding.html)
        [python-pinyin](https://github.com/mozillazg/python-pinyin)

        """
        self.coding = "utf-16-le" if sys.byteorder == "little" else "utf-16-be"
        self.reject_pattern = re.compile(r"[^\u4e00-\u9fffA-Za-z，。、,\. ]")
        self.sub_pattern = re.compile(r"\[[\w_]+\]")
        self.chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
        self.english_word_pattern = re.compile(r"\b[A-Za-z]+\b")
        self.character_simplifier = str.maketrans(
            {
                "：": "，",
                "；": "，",
                "！": "。",
                "（": "，",
                "）": "，",
                "【": "，",
                "】": "，",
                "『": "，",
                "』": "，",
                "「": "，",
                "」": "，",
                "《": "，",
                "》": "，",
                "－": "，",
                ":": ",",
                ";": ",",
                "!": ".",
                "(": ",",
                ")": ",",
                # "[": ",",
                # "]": ",",
                ">": ",",
                "<": ",",
                "-": ",",
            }
        )
        self.halfwidth_2_fullwidth = str.maketrans(
            {
                "!": "！",
                '"': "“",
                "'": "‘",
                "#": "＃",
                "$": "＄",
                "%": "％",
                "&": "＆",
                "(": "（",
                ")": "）",
                ",": "，",
                "-": "－",
                "*": "＊",
                "+": "＋",
                ".": "。",
                "/": "／",
                ":": "：",
                ";": "；",
                "<": "＜",
                "=": "＝",
                ">": "＞",
                "?": "？",
                "@": "＠",
                # '[': '［',
                "\\": "＼",
                # ']': '］',
                "^": "＾",
                # '_': '＿',
                "`": "｀",
                "{": "｛",
                "|": "｜",
                "}": "｝",
                "~": "～",
            }
        )

    def __call__(
        self,
        text: str,
        do_text_normalization=True,
        do_homophone_replacement=True,
        lang: Optional[Literal["zh", "en"]] = None,
    ) -> str:
        if do_text_normalization:
            _lang = self._detect_language(text) if lang is None else lang
            if _lang in self.normalizers:
                texts, tags = _split_tags(text)
                self.logger.debug("split texts %s, tags %s", str(texts), str(tags))
                texts = [self.normalizers[_lang](t) for t in texts]
                self.logger.debug("normed texts %s", str(texts))
                text = _combine_tags(texts, tags) if len(tags) > 0 else texts[0]
                self.logger.debug("combined text %s", text)
            if _lang == "zh":
                text = self._apply_half2full_map(text)
        invalid_characters = self._count_invalid_characters(text)
        if len(invalid_characters):
            self.logger.warning(f"found invalid characters: {invalid_characters}")
            text = self._apply_character_map(text)
        if do_homophone_replacement:
            arr, replaced_words = _fast_replace(
                self.homophones_map,
                text.encode(self.coding),
            )
            if replaced_words:
                text = arr.tobytes().decode(self.coding)
                repl_res = ", ".join([f"{_[0]}->{_[1]}" for _ in replaced_words])
                self.logger.info(f"replace homophones: {repl_res}")
        if len(invalid_characters):
            texts, tags = _split_tags(text)
            self.logger.debug("split texts %s, tags %s", str(texts), str(tags))
            texts = [self.reject_pattern.sub("", t) for t in texts]
            self.logger.debug("normed texts %s", str(texts))
            text = _combine_tags(texts, tags) if len(tags) > 0 else texts[0]
            self.logger.debug("combined text %s", text)
        return text

    def register(self, name: str, normalizer: Callable[[str], str]) -> bool:
        if name in self.normalizers:
            self.logger.warning(f"name {name} has been registered")
            return False
        try:
            val = normalizer("test string 测试字符串")
            if not isinstance(val, str):
                self.logger.warning("normalizer must have caller type (str) -> str")
                return False
        except Exception as e:
            self.logger.warning(e)
            return False
        self.normalizers[name] = normalizer
        return True

    def unregister(self, name: str):
        if name in self.normalizers:
            del self.normalizers[name]

    def destroy(self):
        del_all(self.normalizers)
        del self.homophones_map

    def _load_homophones_map(self, map_file_path: str) -> np.ndarray:
        with open(map_file_path, "r", encoding="utf-8") as f:
            homophones_map: Dict[str, str] = json.load(f)
        map = np.empty((2, len(homophones_map)), dtype=np.uint32)
        for i, k in enumerate(homophones_map.keys()):
            map[:, i] = (ord(k), ord(homophones_map[k]))
        del homophones_map
        return map

    def _count_invalid_characters(self, s: str):
        s = self.sub_pattern.sub("", s)
        non_alphabetic_chinese_chars = self.reject_pattern.findall(s)
        return set(non_alphabetic_chinese_chars)

    def _apply_half2full_map(self, text: str) -> str:
        return text.translate(self.halfwidth_2_fullwidth)

    def _apply_character_map(self, text: str) -> str:
        return text.translate(self.character_simplifier)

    def _detect_language(self, sentence: str) -> Literal["zh", "en"]:
        chinese_chars = self.chinese_char_pattern.findall(sentence)
        english_words = self.english_word_pattern.findall(sentence)

        if len(chinese_chars) > len(english_words):
            return "zh"
        else:
            return "en"
