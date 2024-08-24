import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional, List
import argparse

import numpy as np

import ChatTTS

from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

logger = get_logger("Command")


def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")


def load_normalizer(chat: ChatTTS.Chat):
    # try to load normalizer
    try:
        chat.normalizer.register("en", normalizer_en_nemo_text())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package nemo_text_processing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
        )
    try:
        chat.normalizer.register("zh", normalizer_zh_tn())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package WeTextProcessing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
        )


def main(
    texts: List[str],
    spk: Optional[str] = None,
    stream: bool = False,
    source: str = "local",
    custom_path: str = "",
):
    logger.info("Text input: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    load_normalizer(chat)

    is_load = False
    if os.path.isdir(custom_path) and source == "custom":
        is_load = chat.load(source="custom", custom_path=custom_path)
    else:
        is_load = chat.load(source=source)

    if is_load:
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    if spk is None:
        spk = chat.sample_random_speaker()
    logger.info("Use speaker:")
    print(spk)

    logger.info("Start inference.")
    wavs = chat.infer(
        texts,
        stream,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
        ),
    )
    logger.info("Inference completed.")
    # Save each generated wav file to a local file
    if stream:
        wavs_list = []
    for index, wav in enumerate(wavs):
        if stream:
            for i, w in enumerate(wav):
                save_mp3_file(w, (i + 1) * 1000 + index)
            wavs_list.append(wav)
        else:
            save_mp3_file(wav, index)
    if stream:
        for index, wav in enumerate(np.concatenate(wavs_list, axis=1)):
            save_mp3_file(wav, index)
    logger.info("Audio generation successful.")


if __name__ == "__main__":
    r"""
    python -m examples.cmd.run \
        --source custom --custom_path ../../models/2Noise/ChatTTS 你好喲 ":)"
    """
    logger.info("Starting ChatTTS commandline demo...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command",
        usage='[--spk xxx] [--stream] [--source ***] [--custom_path XXX] "Your text 1." " Your text 2."',
    )
    parser.add_argument(
        "--spk",
        help="Speaker (empty to sample a random one)",
        type=Optional[str],
        default=None,
    )
    parser.add_argument(
        "--stream",
        help="Use stream mode",
        action="store_true",
    )
    parser.add_argument(
        "--source",
        help="source form [ huggingface(hf download), local(ckpt save to asset dir), custom(define) ]",
        type=str,
        default="local",
    )
    parser.add_argument(
        "--custom_path",
        help="custom defined model path(include asset ckpt dir)",
        type=str,
        default="",
    )
    parser.add_argument(
        "texts",
        help="Original text",
        default=["YOUR TEXT HERE"],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger.info(args)
    main(args.texts, args.spk, args.stream, args.source, args.custom_path)
    logger.info("ChatTTS process finished.")
