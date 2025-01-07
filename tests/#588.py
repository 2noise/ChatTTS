import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging
import re

import ChatTTS

from tools.logger import get_logger

logger = get_logger("Test", lv=logging.WARN)

chat = ChatTTS.Chat(logger)
chat.load(compile=False, source="huggingface")  # Set to True for better performance

texts = [
    "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。",
    "你真是太聪明啦。",
]

fail = False

refined = chat.infer(
    texts,
    refine_text_only=True,
    stream=False,
    split_text=False,
    params_refine_text=ChatTTS.Chat.RefineTextParams(show_tqdm=False),
)

trimre = re.compile("\\[[\w_]+\\]")


def trim_tags(txt: str) -> str:
    global trimre
    return trimre.sub("", txt)


for i, t in enumerate(refined):
    if len(trim_tags(t)) > 4 * len(texts[i]):
        fail = True
        logger.warning("in: %s, out: %s", texts[i], t)

if fail:
    import sys

    sys.exit(1)
