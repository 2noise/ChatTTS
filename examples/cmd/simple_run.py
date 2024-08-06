r"""
mkdir -p records && python -m examples.cmd.simple_run
mkdir -p records && STEAM=1 python -m examples.cmd.simple_run
mkdir -p records && CUSTOM_PATH=../../models/2Noise/ChatTTS \
    python -m examples.cmd.simple_run
mkdir -p records && STEAM=1 CUSTOM_PATH=../../models/2Noise/ChatTTS \
    python -m examples.cmd.simple_run
"""
import logging
import os

import torchaudio
import numpy as np

import ChatTTS
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn
from tools.audio import ndarray_to_tensor


def load_normalizer(chat: ChatTTS.Chat):
    # try to load normalizer
    try:
        chat.normalizer.register("en", normalizer_en_nemo_text())
    except ValueError as e:
        logging.error(e)
    except BaseException:
        logging.warning("Package nemo_text_processing not found!")
        logging.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
        )
    try:
        chat.normalizer.register("zh", normalizer_zh_tn())
    except ValueError as e:
        logging.error(e)
    except BaseException:
        logging.warning("Package WeTextProcessing not found!")
        logging.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
        )


if __name__ == "__main__":
    is_stream = bool(os.getenv("STREAM", ""))
    custom_path = os.getenv("CUSTOM_PATH", "")

    chat = ChatTTS.Chat()
    load_normalizer(chat)

    if os.path.isdir(custom_path):
        chat.load(compile=True,
                  source="custom",
                  custom_path=custom_path)
    else:
        chat.load(compile=True)

    texts = ["你好，我是机器人", "我是机器人"]
    ###################################
    # Sample a speaker from Gaussian.
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt="[speed_5]",
        spk_emb=rand_spk,  # add sampled speaker
        top_P=0.7,  # top P decode
        top_K=20,  # top K decode
        temperature=0.3,  # using custom temperature
    )

    ###################################
    # For sentence level manual control.

    # use oral_(0-9), laugh_(0-2), break_(0-7)
    # to generate special token in text to synthesize.
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )

    if is_stream is False:
        wavs = chat.infer(
            texts,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=True,
        )
        print("wavs[0] type", type(wavs[0]), "wavs[0] shape", wavs[0].shape, "shape", wavs.shape)
        for i in range(wavs.shape[0]):
            wav_tensor = ndarray_to_tensor(wavs[i])
            print("wav tensor shape", wav_tensor.shape, "type", wav_tensor.dtype)
            torchaudio.save(f"./records/chat_tts_sentence{i}.wav", wav_tensor, 24000)

        ###################################
        # For word level manual control.
        text = 'What is [uv_break]your favorite english food[laugh][lbreak]'
        wav = chat.infer(
            text, skip_refine_text=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=True,
        )
        print("wav[0] type", type(wav[0]), "shape", wav.shape)

        torchaudio.save("./records/chat_tts_word.wav", ndarray_to_tensor(wav[0]), 24000)

    else:
        ############ stream infer ############
        texts = ["你好，我是机器人", "我是机器人一号", "我是机器人二号"]
        wavs_iter = chat.infer(
            texts,
            stream=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=False,
        )
        res = None
        for wavs in wavs_iter:
            if res is None:
                res = wavs
            else:
                res = np.concatenate([res, wavs], axis=1)

            print("wavs type", type(wavs), "shape", wavs.shape)

        print("res type", type(res), "shape", res.shape)

        for i in range(res.shape[0]):
            wav_tensor = ndarray_to_tensor(res[i])
            print("wav tensor shape", wav_tensor.shape, "type", wav_tensor.dtype)
            torchaudio.save(f"./records/chat_tts_stream{i}.wav", wav_tensor, 24000)
