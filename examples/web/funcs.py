import sys
import random
from typing import Optional

import gradio as gr
import numpy as np

from tools.audio import unsafe_float_to_int16
from tools.logger import get_logger
logger = get_logger(" WebUI ")

from tools.seeder import TorchSeedContext
from tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn

import ChatTTS
chat = ChatTTS.Chat(get_logger("ChatTTS"))

custom_path: Optional[str] = None

has_interrupted = False

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

def load_chat(cust_path: Optional[str], coef: Optional[str]) -> bool:
    if cust_path == None:
        ret = chat.load(coef=coef, compile=sys.platform != 'win32')
    else:
        logger.info('local model path: %s', cust_path)
        ret = chat.load('custom', custom_path=cust_path, coef=coef, compile=sys.platform != 'win32')
        global custom_path
        custom_path = cust_path
    if ret:
        try:
            chat.normalizer.register("en", normalizer_en_nemo_text())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning('Package nemo_text_processing not found!')
            logger.warning(
                'Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing',
            )
        try:
            chat.normalizer.register("zh", normalizer_zh_tn())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning('Package WeTextProcessing not found!')
            logger.warning(
                'Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing',
            )
    return ret

def reload_chat(coef: Optional[str]) -> str:
    chat.unload()
    gr.Info("Model unloaded.")
    if len(coef) != 230:
        gr.Warning("Ingore invalid DVAE coefficient.")
        coef = None
    try:
        global custom_path
        ret = load_chat(custom_path, coef)
    except Exception as e:
        raise gr.Error(str(e))
    if not ret:
        raise gr.Error("Unable to load model.")
    gr.Info("Reload succeess.")
    return chat.coef

def set_generate_buttons(generate_button, interrupt_button, is_reset=False):
    return gr.update(value=generate_button, visible=is_reset, interactive=is_reset), gr.update(value=interrupt_button, visible=not is_reset, interactive=not is_reset)

def refine_text(text, text_seed_input, refine_text_flag, generate_button, interrupt_button):
    global chat, has_interrupted
    has_interrupted = False

    if not refine_text_flag:
        return text, *set_generate_buttons(generate_button, interrupt_button, is_reset=True)

    with TorchSeedContext(text_seed_input):
        text = chat.infer(
            text,
            skip_refine_text=False,
            refine_text_only=True,
        )
    return text[0] if isinstance(text, list) else text, *set_generate_buttons(generate_button, interrupt_button, is_reset=True)

def text_output_listener(generate_button, interrupt_button):
    return set_generate_buttons(generate_button, interrupt_button)

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, stream):
    global chat, has_interrupted

    if not text or text == "𝕃𝕠𝕒𝕕𝕚𝕟𝕘..." or has_interrupted: return None

    with TorchSeedContext(audio_seed_input):
        rand_spk = chat.sample_random_speaker()

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
    )

    with TorchSeedContext(audio_seed_input):
        wav = chat.infer(
            text,
            skip_refine_text=True,
            params_infer_code=params_infer_code,
            stream=stream,
        )
        if stream:
            for gen in wav:
                audio = gen[0]
                if audio is not None and len(audio) > 0:
                    yield 24000, unsafe_float_to_int16(audio[0])
                    del audio
            return

    yield 24000, unsafe_float_to_int16(np.array(wav[0]).flatten())

def interrupt_generate():
    global chat, has_interrupted

    has_interrupted = True
    chat.interrupt()

def set_buttons_after_generate(generate_button, interrupt_button, audio_output):
    global has_interrupted

    return set_generate_buttons(
        generate_button, interrupt_button,
        audio_output is not None or has_interrupted,
    )
