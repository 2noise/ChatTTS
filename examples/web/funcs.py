import random
from typing import Optional

import gradio as gr
import numpy as np

from tools.audio import unsafe_float_to_int16
from tools.logger import get_logger
logger = get_logger(" WebUI ")

from tools.seeder import TorchSeedContext

import ChatTTS
chat = ChatTTS.Chat(get_logger("ChatTTS"))

custom_path: Optional[str] = None

# 音色选项：用于预置合适的音色
voices = {
    "Default": {"seed": 2},
    "Timbre1": {"seed": 1111},
    "Timbre2": {"seed": 2222},
    "Timbre3": {"seed": 3333},
    "Timbre4": {"seed": 4444},
    "Timbre5": {"seed": 5555},
    "Timbre6": {"seed": 6666},
    "Timbre7": {"seed": 7777},
    "Timbre8": {"seed": 8888},
    "Timbre9": {"seed": 9999},
}

def generate_seed():
    return gr.update(value=random.randint(1, 100000000))

# 返回选择音色对应的seed
def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)['seed']

def reload_chat(coef: Optional[str]) -> str:
    global custom_path
    chat.unload()
    gr.Info("Model unloaded.")
    try:
        if len(coef) != 230:
            gr.Warning("Ingore invalid DVAE coefficient.")
            coef = None
        if custom_path == None:
            ret = chat.load_models(coef=coef)
        else:
            logger.info('local model path: %s', custom_path)
            ret = chat.load_models('custom', custom_path=custom_path, coef=coef)
        if not ret:
            raise gr.Error("Unable to load model.")
        gr.Info("Reload succeess.")
        return chat.coef
    except Exception as e:
        raise gr.Error(str(e))

def refine_text(text, text_seed_input, refine_text_flag):
    if not refine_text_flag:
        return text

    global chat

    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    with TorchSeedContext(text_seed_input):
        text = chat.infer(text,
                            skip_refine_text=False,
                            refine_text_only=True,
                            params_refine_text=params_refine_text,
                            )
    return text[0] if isinstance(text, list) else text

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, stream):
    if not text: return None

    global chat

    with TorchSeedContext(audio_seed_input):
        rand_spk = chat.sample_random_speaker()
        params_infer_code = {
            'spk_emb': rand_spk,
            'temperature': temperature,
            'top_P': top_P,
            'top_K': top_K,
        }

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
