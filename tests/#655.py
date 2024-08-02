import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging

import torch

import ChatTTS

from tools.logger import get_logger

logger = get_logger("Test #655", lv=logging.WARN)

chat = ChatTTS.Chat(logger)
chat.load(compile=False, source="huggingface")  # Set to True for better performance

rand_spk = chat.sample_random_speaker()

params = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

text = ['What is [uv_break]your favorite english food?[laugh][lbreak]']

fail = False

input_ids, attention_mask, text_mask = chat.tokenizer.encode(
    chat.tokenizer.decorate_code_prompts(
        text, params.prompt, params.txt_smp, params.spk_emb,
    ),
    chat.config.gpt.num_vq,
    prompt_str=params.spk_smp,
    device=chat.device_gpt,
)
with torch.inference_mode():
    start_idx, end_idx = 0, torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.long
    ).fill_(input_ids.shape[1])

    recoded_text = chat.tokenizer.decode(chat.gpt._prepare_generation_outputs(
        input_ids, start_idx, end_idx, [], [], True,
    ).ids)

fail = recoded_text[0] != '[Stts] [spk_emb] [speed_5] what is [uv_break] your favorite english food? [laugh] [lbreak] [Ptts]'

if fail:

    logging.warning("got recoded_text '%s'", recoded_text[0])

    import sys

    sys.exit(1)
