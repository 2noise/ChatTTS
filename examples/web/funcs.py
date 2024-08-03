import sys
import random
from typing import Optional
from time import sleep

import gradio as gr

from tools.audio import float_to_int16, has_ffmpeg_installed, load_audio
from tools.logger import get_logger

logger = get_logger(" WebUI ")

from tools.seeder import TorchSeedContext
from tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn

import ChatTTS

chat = ChatTTS.Chat(get_logger("ChatTTS"))

custom_path: Optional[str] = None

has_interrupted = False
is_in_generate = False

seed_min = 1
seed_max = 4294967295

use_mp3 = has_ffmpeg_installed()
if not use_mp3:
    logger.warning("no ffmpeg installed, use wav file output")

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
    return gr.update(value=random.randint(seed_min, seed_max))


# 返回选择音色对应的seed
def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)["seed"]


def on_audio_seed_change(audio_seed_input):
    with TorchSeedContext(audio_seed_input):
        rand_spk = chat.sample_random_speaker()
    return rand_spk


def load_chat(cust_path: Optional[str], coef: Optional[str]) -> bool:
    if cust_path == None:
        ret = chat.load(coef=coef, compile=sys.platform != "win32")
    else:
        logger.info("local model path: %s", cust_path)
        ret = chat.load(
            "custom", custom_path=cust_path, coef=coef, compile=sys.platform != "win32"
        )
        global custom_path
        custom_path = cust_path
    if ret:
        try:
            chat.normalizer.register("en", normalizer_en_nemo_text())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package nemo_text_processing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
            )
        try:
            chat.normalizer.register("zh", normalizer_zh_tn())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package WeTextProcessing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
            )
    return ret


def reload_chat(coef: Optional[str]) -> str:
    global is_in_generate

    if is_in_generate:
        gr.Warning("Cannot reload when generating!")
        return coef

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


def on_upload_sample_audio(sample_audio_input: Optional[str]) -> str:
    if sample_audio_input is None:
        return ""
    sample_audio = load_audio(sample_audio_input, 24000)
    spk_smp = chat.sample_audio_speaker(sample_audio)
    del sample_audio
    return spk_smp


def _set_generate_buttons(generate_button, interrupt_button, is_reset=False):
    return gr.update(
        value=generate_button, visible=is_reset, interactive=is_reset
    ), gr.update(value=interrupt_button, visible=not is_reset, interactive=not is_reset)


def refine_text(
    text,
    text_seed_input,
    refine_text_flag,
):
    global chat

    if not refine_text_flag:
        sleep(1)  # to skip fast answer of loading mark
        return text

    with TorchSeedContext(text_seed_input):
        text = chat.infer(
            text,
            skip_refine_text=False,
            refine_text_only=True,
        )

    return text[0] if isinstance(text, list) else text


def generate_audio(
    text,
    temperature,
    top_P,
    top_K,
    spk_emb_text: str,
    stream,
    audio_seed_input,
    sample_text_input,
    sample_audio_code_input,
):
    global chat, has_interrupted

    if not text or has_interrupted or not spk_emb_text.startswith("蘁淰"):
        return None

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb_text,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
    )

    if sample_text_input and sample_audio_code_input:
        params_infer_code.txt_smp = sample_text_input
        params_infer_code.spk_smp = sample_audio_code_input
        params_infer_code.spk_emb = None

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
                    yield 24000, float_to_int16(audio).T
                del audio
        else:
            yield 24000, float_to_int16(wav[0]).T


def interrupt_generate():
    global chat, has_interrupted

    has_interrupted = True
    chat.interrupt()


def set_buttons_before_generate(generate_button, interrupt_button):
    global has_interrupted, is_in_generate

    has_interrupted = False
    is_in_generate = True

    return _set_generate_buttons(
        generate_button,
        interrupt_button,
    )


def set_buttons_after_generate(generate_button, interrupt_button, audio_output):
    global has_interrupted, is_in_generate

    is_in_generate = False

    return _set_generate_buttons(
        generate_button,
        interrupt_button,
        audio_output is not None or has_interrupted,
    )
