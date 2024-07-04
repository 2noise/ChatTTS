import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging

import ChatTTS

from tools.logger import get_logger

logger = get_logger("Test #521", lv=logging.WARN)

chat = ChatTTS.Chat(logger)
chat.load(compile=False, source="huggingface")  # Set to True for better performance

texts = [
    "这段代码在流式输出的情况下，和非流式情况下，计算是否一致？我在流式输出的情况下，会产生噪音，怀疑这部分有问题，哪位大佬可以指教一下？",
    "我也发现流式输出有时候有问题，流式输出是一个ndarray list，正常情况下会ndarray输出是12032维，但是会随机在中间偶发输出256维，开始输出256维后就会一直保持，256维的部分都是噪声。",
]

gen_result = chat.infer(
    texts,
    stream=True,
    params_refine_text=ChatTTS.Chat.RefineTextParams(
        show_tqdm=False,
    ),
    params_infer_code=ChatTTS.Chat.InferCodeParams(
        show_tqdm=False,
    ),
)

has_finished = [False for _ in range(len(texts))]

fail = False

for i, result in enumerate(gen_result):
    for j, wav in enumerate(result):
        if wav is None:
            continue
        logger.info("iter %d index %d len %d", i, j, len(wav))
        if len(wav) == 12032:
            continue
        if not has_finished[j]:
            has_finished[j] = True
            logger.warning(
                "iter %d index %d finished with non-12032 len %d", i, j, len(wav)
            )
        else:
            logger.warning(
                "stream iter %d index %d returned non-zero wav after finished", i, j
            )
            fail = True

if fail:
    import sys

    sys.exit(1)
