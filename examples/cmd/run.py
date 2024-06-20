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
import numpy as np
import argparse


def save_wav_file(wav, index):
    wav_filename = f"output_audio_{index}.wav"
    # Convert numpy array to bytes and write to WAV file
    wav_bytes = (wav * 32768).astype('int16').tobytes()
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(wav_bytes)
    print(f"Audio saved to {wav_filename}")

def main(text="<YOUR TEXT HERE>", stream=False):
    print(f"{stream=} Received text input: {text}")

    chat = ChatTTS.Chat()
    print("Initializing ChatTTS...")
    # if using macbook(M1), I suggest you set `device='cpu', compile=False`
    chat.load_models()
    print("Models loaded successfully.")

    texts = [text]
    print("Text prepared for inference:", texts)

    wavs_gen = chat.infer(texts, use_decoder=True, stream=stream)
    print("Inference completed. Audio generation successful.")
    # Save each generated wav file to a local file

    if stream:
        print('generate with stream mode ..')
        wavs = [np.array([[]])]
        for gen in wavs_gen:
            print('got new chunk', gen)
            # play chunk or combine into one complete audio;
            wavs[0] = np.hstack([wavs[0], np.array(gen[0])])
    else:
        print('generate without stream mode ..')
        wavs = wavs_gen

    for index, wav in enumerate(wavs):
        save_wav_file(wav, index)

    return Audio(wavs[0], rate=24_000, autoplay=True)

if __name__ == "__main__":
    print("Starting the TTS application...")

    parser = argparse.ArgumentParser(description='ChatTTS cmd demo', usage="--stream hello, my name is bob.")
    parser.add_argument("--stream", action="store_true", default=False, help="Use stream infer")
    parser.add_argument("text", help="Original text", default='YOUR TEXT HERE', nargs='*')
    args = parser.parse_args()
    main(text=' '.join(args.text), stream=args.stream)
    print("TTS application finished.")
