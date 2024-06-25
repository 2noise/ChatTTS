import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import wave
import argparse
from io import BytesIO

import ChatTTS

from tools.audio import unsafe_float_to_int16, wav2
from tools.logger import get_logger

logger = get_logger("Command")


def save_mp3_file(wav, index):
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(unsafe_float_to_int16(wav))
    buf.seek(0, 0)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        wav2(buf, f, "mp3")
    logger.info(f"Audio saved to {mp3_filename}")


def main(texts: list[str]):
    logger.info("Text input: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    wavs = chat.infer(texts, use_decoder=True)
    logger.info("Inference completed. Audio generation successful.")
    # Save each generated wav file to a local file
    for index, wav in enumerate(wavs):
        save_mp3_file(wav, index)


if __name__ == "__main__":
    logger.info("Starting the TTS application...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command", usage="--stream hello, my name is bob."
    )
    parser.add_argument(
        "text", help="Original text", default="YOUR TEXT HERE", nargs="*"
    )
    args = parser.parse_args()
    main(args.text)
    logger.info("TTS application finished.")
