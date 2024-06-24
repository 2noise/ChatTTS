import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import wave
import argparse

from dotenv import load_dotenv
load_dotenv("sha256.env")

import ChatTTS

from tools.audio import unsafe_float_to_int16
from tools.logger import get_logger

logger = get_logger("Command")

def save_wav_file(wav, index):
    wav_filename = f"output_audio_{index}.wav"
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(unsafe_float_to_int16(wav))
    logger.info(f"Audio saved to {wav_filename}")

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
        save_wav_file(wav, index)

if __name__ == "__main__":
    logger.info("Starting the TTS application...")
    parser = argparse.ArgumentParser(description='ChatTTS Command', usage="--stream hello, my name is bob.")
    parser.add_argument("text", help="Original text", default='YOUR TEXT HERE', nargs='*')
    args = parser.parse_args()
    main(args.text)
    logger.info("TTS application finished.")
