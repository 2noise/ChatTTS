import argparse
import os
import sys
import wave

import numpy as np
import torch
import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ChatTTS

chat = ChatTTS.Chat()
print("loading ChatTTS model...")
# Windows not yet supported for torch.compile
chat.load_models(compile=False)


def generate_speaker(num=10):
    return [(_, chat.sample_random_speaker()) for _ in range(num)]


def save_speakers(speakers, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for idx, speaker in speakers:
        speaker_dir = os.path.join(directory, f"{idx:04d}")
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        speaker_file = os.path.join(speaker_dir, "speaker.pt")
        torch.save(speaker, speaker_file)


def load_dataset(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_audio(audio):
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def save_audio(file_name, audio, rate=24000):
    audio = normalize_audio(audio)
    audio = (audio * 32767).astype(np.int16)
    with wave.open(file_name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    return file_name


def generate_audio_files(speakers, dataset_name, dateset_txts, output_dir):
    for speaker_idx, speaker in tqdm.tqdm(speakers):
        speaker_dir = os.path.join(output_dir, f"{speaker_idx:04d}", dataset_name)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)

        params_infer_code = {
            'spk_emb': speaker,
            'temperature': 0.001,
            'top_P': 0.7,
            'top_K': 20,
        }
        wavs = chat.infer(dateset_txts,
                          skip_refine_text=True,
                          params_infer_code=params_infer_code
                          )
        for idx, w in enumerate(wavs):
            save_audio(os.path.join(speaker_dir, f"{idx:04d}.wav"), w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random speakers and audio samples.')
    parser.add_argument('--dir', type=str, default='./test_audio', help='Directory to store test audio files.')
    parser.add_argument('--num', type=int, default=10, help='Number of random speakers to generate.')
    parser.add_argument('--ds', type=str, default='test_data.yaml', help='Path to the dataset YAML file.')

    args = parser.parse_args()
    if os.path.exists(args.dir) and os.listdir(args.dir):
        user_input = input(
            f"The directory '{args.dir}' is not empty. Do you want to overwrite its contents? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Operation cancelled.")
            exit()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    speakers = generate_speaker(args.num)

    save_speakers(speakers, args.dir)

    dataset = load_dataset(args.ds)

    for name, txts in dataset.items():
        generate_audio_files(speakers, name, txts, args.dir)
