import ChatTTS
from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view

import os
import sys


def init():
    logger = get_logger("Test")

    now_dir = os.getcwd()
    sys.path.append(now_dir)

    chat = ChatTTS.Chat()
    chat.load(source="local", compile=False)

    # Sample a speaker from Gaussian.
    rand_spk = chat.sample_random_speaker()
    logger.info(f"rand_spk: {rand_spk}")
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )

    """
    RefineTextParams: 
    For sentence level manual control.
    use oral_(0-9), laugh_(0-2), break_(0-7) to generate special token in text to synthesize.
    """
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_1][break_6]",
    )

    return logger, chat, params_infer_code, params_refine_text


def save_mp3_file(wav, tag: str):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{tag}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")


if __name__ == "__main__":
    logger, chat, params_infer_code, params_refine_text = init()
    logger.info("Initializing ChatTTS ...")

    # test for sentence level manual control
    texts = ["朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。"]
    logger.info("Text input: %s", str(texts))
    wavs1 = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )
    save_mp3_file(wavs1[0], "sentence_level_test")

    # test for word level manual control
    text = "朝辞白帝[uv_break]彩云间[uv_break]，千里江陵[uv_break]一日还[uv_break]。两岸猿声[uv_break]啼不住[laugh]，轻舟[uv_break]已过[uv_break]万重山[lbreak]。"
    wavs2 = chat.infer(
        text,
        skip_refine_text=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )
    save_mp3_file(wavs2[0], "words_level_test")
