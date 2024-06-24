import json
import re
from typing import Dict, Tuple, List
import sys

from numba import jit
import numpy as np


@jit
def _find_index(table: np.ndarray, val: np.uint16):
    for i in range(table.size):
        if table[i] == val:
            return i
    return -1

@jit
def _fast_replace(table: np.ndarray, text: bytes) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
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

class HomophonesReplacer:
    """
    Homophones Replacer

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
    def __init__(self, map_file_path: str):
        self.homophones_map = self._load_homophones_map(map_file_path)
        self.coding = "utf-16-le" if sys.byteorder == "little" else "utf-16-be"

    def _load_homophones_map(self, map_file_path: str) -> np.ndarray:
        with open(map_file_path, 'r', encoding='utf-8') as f:
            homophones_map: Dict[str, str] = json.load(f)
        map = np.empty((2, len(homophones_map)), dtype=np.uint32)
        for i, k in enumerate(homophones_map.keys()):
            map[:, i] = (ord(k), ord(homophones_map[k]))
        del homophones_map
        return map

    def replace(self, text: str):
        arr, lst = _fast_replace(
            self.homophones_map,
            text.encode(self.coding),
        )
        return arr.tobytes().decode(self.coding), lst

accept_pattern = re.compile(r'[^\u4e00-\u9fffA-Za-z，。、,\. ]')
sub_pattern = re.compile(r'\[uv_break\]|\[laugh\]|\[lbreak\]')

def count_invalid_characters(s: str):
    global accept_pattern, sub_pattern
    s = sub_pattern.sub('', s)
    non_alphabetic_chinese_chars = accept_pattern.findall(s)
    return set(non_alphabetic_chinese_chars)

chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
english_word_pattern = re.compile(r'\b[A-Za-z]+\b')

def detect_language(sentence):
    global chinese_char_pattern, english_word_pattern

    chinese_chars = chinese_char_pattern.findall(sentence)
    english_words = english_word_pattern.findall(sentence)

    if len(chinese_chars) > len(english_words):
        return "zh"
    else:
        return "en"


character_simplifier = str.maketrans({
    '：': '，',
    '；': '，',
    '！': '。',
    '（': '，',
    '）': '，',
    '【': '，',
    '】': '，',
    '『': '，',
    '』': '，',
    '「': '，',
    '」': '，',
    '《': '，',
    '》': '，',
    '－': '，',
    '‘': '',
    '“': '',
    '’': '',
    '”': '',
    ':': ',',
    ';': ',',
    '!': '.',
    '(': ',',
    ')': ',',
    '[': ',',
    ']': ',',
    '>': ',',
    '<': ',',
    '-': ',',
})

halfwidth_2_fullwidth = str.maketrans({
        '!': '！',
        '"': '“',
        "'": '‘',
        '#': '＃',
        '$': '＄',
        '%': '％',
        '&': '＆',
        '(': '（',
        ')': '）',
        ',': '，',
        '-': '－',
        '*': '＊',
        '+': '＋',
        '.': '。',
        '/': '／',
        ':': '：',
        ';': '；',
        '<': '＜',
        '=': '＝',
        '>': '＞',
        '?': '？',
        '@': '＠',
        # '[': '［',
        '\\': '＼',
        # ']': '］',
        '^': '＾',
        # '_': '＿',
        '`': '｀',
        '{': '｛',
        '|': '｜',
        '}': '｝',
        '~': '～'
    })

def apply_half2full_map(text: str) -> str:
    return text.translate(halfwidth_2_fullwidth)

def apply_character_map(text: str) -> str:
    return text.translate(character_simplifier)
