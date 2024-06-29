import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import argparse
from typing import Optional, List

import ChatTTS

from tools.audio import wav_arr_to_mp3_view
from tools.logger import get_logger

logger = get_logger("Command")


def save_mp3_file(wav, index):
    data = wav_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")


def main(texts: List[str], spk: Optional[str] = None):
    logger.info("Text input: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
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
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
        ),
    )
    logger.info("Inference completed.")
    # Save each generated wav file to a local file
    for index, wav in enumerate(wavs):
        save_mp3_file(wav, index)
    logger.info("Audio generation successful.")


if __name__ == "__main__":
    logger.info("Starting ChatTTS commandline demo...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command",
        usage='[--spk xxx] "Your text 1." " Your text 2."',
    )
    parser.add_argument(
        "--spk",
        help="Speaker (empty to sample a random one)",
        type=Optional[str],
        default=None,
    )
    parser.add_argument(
        "texts", help="Original text", default="YOUR TEXT HERE", nargs="*"
    )
    args = parser.parse_args()
    main(args.texts, args.spk)
    logger.info("ChatTTS process finished.")
