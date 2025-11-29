#!/usr/bin/env python3
"""
ChatTTS Prompt to MP3 Demo

This script demonstrates how to:
1. Accept text input from user prompts
2. Generate speech using ChatTTS
3. Save the output as MP3 files
4. Support advanced features like speaker control and prosody

Usage:
    python prompt_to_mp3.py
"""

import os
import sys
import ChatTTS
import torch
import torchaudio
from pydub import AudioSegment
import numpy as np

def save_as_mp3(wav_array, filename, sample_rate=24000):
    """Convert numpy array to MP3 file using pydub"""
    # Normalize audio to prevent clipping
    wav_array = wav_array / np.max(np.abs(wav_array)) * 0.9

    # Convert to 16-bit PCM for MP3 encoding
    wav_array_int = (wav_array * 32767).astype(np.int16)

    # Create AudioSegment from numpy array
    audio_segment = AudioSegment(
        wav_array_int.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )

    # Export as MP3
    audio_segment.export(filename, format="mp3", bitrate="192k")
    print(f"✓ Saved as MP3: {filename}")

def get_user_input():
    """Get text input from user with options for advanced features"""
    print("\n" + "="*50)
    print("ChatTTS Prompt to MP3 Demo")
    print("="*50)

    # Get main text input
    print("\n📝 Enter the text you want to convert to speech:")
    print("   (You can use special tokens like [laugh], [break_4], [uv_break])")
    text = input(">>> ").strip()

    if not text:
        print("❌ No text provided. Exiting.")
        return None

    # Ask for optional speaker control
    print("\n🎤 Do you want to use a specific speaker? (y/n)")
    use_speaker = input(">>> ").strip().lower() == 'y'

    speaker_emb = None
    if use_speaker:
        print("\nChoose speaker option:")
        print("1. Random speaker")
        print("2. Use existing speaker embedding")
        print("3. Extract from audio file")
        choice = input(">>> ").strip()

        if choice == "1":
            speaker_emb = chat.sample_random_speaker()
            print(f"🎭 Random speaker embedding: {speaker_emb[:50]}...")
        elif choice == "2":
            print("Enter speaker embedding string:")
            speaker_emb = input(">>> ").strip()
        elif choice == "3":
            print("Enter path to audio file for speaker extraction:")
            audio_path = input(">>> ").strip()
            if os.path.exists(audio_path):
                try:
                    # Load audio and extract speaker
                    audio, sr = torchaudio.load(audio_path)
                    if sr != 24000:
                        # Resample if needed
                        resampler = torchaudio.transforms.Resample(sr, 24000)
                        audio = resampler(audio)
                    speaker_emb = chat.sample_audio_speaker(audio.numpy())
                    print(f"🎭 Extracted speaker embedding: {speaker_emb[:50]}...")
                except Exception as e:
                    print(f"❌ Error loading audio: {e}")
                    speaker_emb = None

    # Ask for output filename
    print("\n💾 Enter output filename (without .mp3 extension):")
    filename = input(">>> ").strip()
    if not filename:
        filename = "output"

    return {
        'text': text,
        'speaker_emb': speaker_emb,
        'filename': filename
    }

def main():
    """Main function to run the demo"""
    global chat

    print("🚀 Initializing ChatTTS...")

    try:
        # Initialize ChatTTS
        chat = ChatTTS.Chat()

        # Load models (use compile=True for better performance if you have CUDA)
        print("📥 Loading models...")
        success = chat.load(compile=False)

        if not success:
            print("❌ Failed to load ChatTTS models")
            return

        print("✅ ChatTTS loaded successfully!")

        while True:
            # Get user input
            user_input = get_user_input()
            if user_input is None:
                break

            # Prepare inference parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                temperature=0.3,
                top_P=0.7,
                top_K=20
            )

            # Apply speaker embedding if provided
            if user_input['speaker_emb']:
                params_infer_code.spk_emb = user_input['speaker_emb']

            # Generate speech
            print("\n🔊 Generating speech...")
            try:
                wavs = chat.infer(
                    user_input['text'],
                    params_infer_code=params_infer_code
                )

                if wavs and len(wavs) > 0:
                    # Save each generated audio as MP3
                    for i, wav in enumerate(wavs):
                        if len(wavs) > 1:
                            output_filename = f"{user_input['filename']}_{i+1}.mp3"
                        else:
                            output_filename = f"{user_input['filename']}.mp3"

                        save_as_mp3(wav, output_filename)

                        # Also save as WAV for reference
                        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
                        torchaudio.save(output_filename.replace('.mp3', '.wav'), wav_tensor, 24000)
                        print(f"✓ Saved as WAV: {output_filename.replace('.mp3', '.wav')}")

                    print(f"\n🎉 Successfully generated {len(wavs)} audio file(s)!")

                else:
                    print("❌ No audio generated")

            except Exception as e:
                print(f"❌ Error during inference: {e}")

            # Ask if user wants to continue
            print("\n🔄 Do you want to generate another audio? (y/n)")
            if input(">>> ").strip().lower() != 'y':
                break

        print("\n👋 Demo completed. Thank you for using ChatTTS!")

    except Exception as e:
        print(f"❌ Error initializing ChatTTS: {e}")
        print("\n💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()