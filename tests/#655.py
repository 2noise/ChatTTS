import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging

import torch

import ChatTTS

from tools.logger import get_logger
from tools.normalizer import normalizer_en_nemo_text

logger = get_logger("Test", lv=logging.WARN)

chat = ChatTTS.Chat(logger)
chat.load(compile=False, source="huggingface")  # Set to True for better performance
try:
    chat.normalizer.register("en", normalizer_en_nemo_text())
except:
    logger.warning("Package nemo_text_processing not found!")

rand_spk = chat.sample_random_speaker()


text = ["What is [uv_break]your favorite english food?[laugh][lbreak]"]

fail = False

refined_text = chat.infer(
    text,
    refine_text_only=True,
    params_refine_text=ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_6]",
        manual_seed=12345,
    ),
    split_text=False,
)
if (
    refined_text[0]
    != "what is [uv_break] your favorite english [uv_break] food [laugh] like [lbreak]"
):
    fail = True
    logger.warning("refined text is '%s'", refined_text[0])

params = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.3,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)
input_ids, attention_mask, text_mask = chat.tokenizer.encode(
    chat.speaker.decorate_code_prompts(
        text,
        params.prompt,
        params.txt_smp,
        params.spk_emb,
    ),
    chat.config.gpt.num_vq,
    prompt=(
        chat.speaker.decode_prompt(params.spk_smp)
        if params.spk_smp is not None
        else None
    ),
    device=chat.device_gpt,
)
with torch.inference_mode():
    start_idx, end_idx = 0, torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.long
    ).fill_(input_ids.shape[1])

    recoded_text = chat.tokenizer.decode(
        chat.gpt._prepare_generation_outputs(
            input_ids,
            start_idx,
            end_idx,
            [],
            [],
            True,
        ).ids
    )

if (
    recoded_text[0]
    != "[Stts] [spk_emb] [speed_5] what is [uv_break] your favorite english food? [laugh] [lbreak] [Ptts]"
):
    fail = True
    logger.warning("recoded text is '%s'", refined_text)

if fail:
    import sys

    sys.exit(1)
