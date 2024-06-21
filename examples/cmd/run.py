import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from dotenv import load_dotenv
load_dotenv("sha256.env")

import wave
import ChatTTS
from IPython.display import Audio

from tools.logger import get_logger

logger = get_logger("Command")

def save_wav_file(wav, index):
    wav_filename = f"output_audio_{index}.wav"
    # Convert numpy array to bytes and write to WAV file
    wav_bytes = (wav * 32768).astype('int16').tobytes()
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(wav_bytes)
    logger.info(f"Audio saved to {wav_filename}")

def main():
    # Retrieve text from command line argument
    text_input = sys.argv[1] if len(sys.argv) > 1 else "<YOUR TEXT HERE>"
    logger.info("Received text input: %s", text_input)

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load_models():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    texts = [text_input]
    logger.info("Text prepared for inference: %s", texts)

    wavs = chat.infer(texts, use_decoder=True)
    logger.info("Inference completed. Audio generation successful.")
    # Save each generated wav file to a local file
    for index, wav in enumerate(wavs):
        save_wav_file(wav, index)

    return Audio(wavs[0], rate=24_000, autoplay=True)

if __name__ == "__main__":
    logger.info("Starting the TTS application...")
    main()
    logger.info("TTS application finished.")
