import sys
import torch
import wave
import ChatTTS
from IPython.display import Audio

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

def main():
    # Retrieve text from command line argument
    text_input = sys.argv[1] if len(sys.argv) > 1 else "<YOUR TEXT HERE>"
    print("Received text input:", text_input)

    chat = ChatTTS.Chat()
    print("Initializing ChatTTS...")
    chat.load_models()
    print("Models loaded successfully.")

    texts = [text_input]
    print("Text prepared for inference:", texts)

    wavs = chat.infer(texts, use_decoder=True)
    print("Inference completed. Audio generation successful.")
    # Save each generated wav file to a local file
    for index, wav in enumerate(wavs):
        save_wav_file(wav, index)

    return Audio(wavs[0], rate=24_000, autoplay=True)

if __name__ == "__main__":
    print("Starting the TTS application...")
    main()
    print("TTS application finished.")
