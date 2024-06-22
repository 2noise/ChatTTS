import random

import torch
import gradio as gr
import numpy as np

from tools.audio import unsafe_float_to_int16
from tools.logger import get_logger
logger = get_logger(" WebUI ")

import ChatTTS
chat = ChatTTS.Chat(get_logger("ChatTTS"))

# 音色选项：用于预置合适的音色
voices = {
    "默认": {"seed": 2},
    "音色1": {"seed": 1111},
    "音色2": {"seed": 2222},
    "音色3": {"seed": 3333},
    "音色4": {"seed": 4444},
    "音色5": {"seed": 5555},
    "音色6": {"seed": 6666},
    "音色7": {"seed": 7777},
    "音色8": {"seed": 8888},
    "音色9": {"seed": 9999},
    "音色10": {"seed": 11111},
}

def generate_seed():
    return gr.update(value=random.randint(1, 100000000))

# 返回选择音色对应的seed
def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)['seed']

def refine_text(text, audio_seed_input, text_seed_input, refine_text_flag):
    if not refine_text_flag:
        return text

    global chat

    torch.manual_seed(audio_seed_input)
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    torch.manual_seed(text_seed_input)

    text = chat.infer(text,
                        skip_refine_text=False,
                        refine_text_only=True,
                        params_refine_text=params_refine_text,
                        )
    return text[0] if isinstance(text, list) else text

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, stream):
    if not text: return None

    global chat

    torch.manual_seed(audio_seed_input)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk,
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    torch.manual_seed(text_seed_input)

    wav = chat.infer(
        text,
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=stream,
    )

    if stream:
        for gen in wav:
            yield 24000, unsafe_float_to_int16(gen[0][0])
        return

    yield 24000, unsafe_float_to_int16(np.array(wav[0]).flatten())
