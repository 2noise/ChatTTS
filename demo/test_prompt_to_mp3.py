#!/usr/bin/env python3
"""
Test version of Prompt to MP3 Demo

This script tests the basic functionality without requiring interactive input.
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
        channels=1,
    )

    # Export as MP3
    audio_segment.export(filename, format="mp3", bitrate="192k")
    print(f"✓ Saved as MP3: {filename}")


def test_basic_functionality():
    """Test basic ChatTTS functionality with MP3 output"""
    print("🚀 Testing ChatTTS with MP3 output...")

    try:
        # Initialize ChatTTS
        chat = ChatTTS.Chat()

        # Load models
        print("📥 Loading models...")
        success = chat.load(compile=False)

        if not success:
            print("❌ Failed to load ChatTTS models")
            return False

        print("✅ ChatTTS loaded successfully!")

        # Test text with special tokens
        test_texts = [
            "深夜翻看电报群，满屏皆是红绿跳动的数字，恍惚间竟分不清是K线还是心电图。忽闻有人叩门，原是楼下张二，他两眼发赤，手里攥着手机，口中喃喃：“牛市！牛市要来了！”我问他：“何谓牛市？”他咧嘴一笑：“便是人人持币，夜夜暴富，狗儿撒尿也能浇出个元宇宙！”说罢踉跄而去，鞋底沾着前日暴跌时撕碎的合约单。窗外的月亮冷得很，像极了一枚被庄家抛售的比特币。"
        ]

        # Generate speech
        print("🔊 Generating speech...")
        wavs = chat.infer(test_texts)

        if wavs and len(wavs) > 0:
            # Save as MP3
            for i, wav in enumerate(wavs):
                output_filename = f"test_output_{i+1}.mp3"
                save_as_mp3(wav, output_filename)

                # Also save as WAV for reference
                wav_tensor = torch.from_numpy(wav).unsqueeze(0)
                torchaudio.save(
                    output_filename.replace(".mp3", ".wav"), wav_tensor, 24000
                )
                print(f"✓ Saved as WAV: {output_filename.replace('.mp3', '.wav')}")

            print(f"\n🎉 Successfully generated {len(wavs)} audio file(s)!")
            return True
        else:
            print("❌ No audio generated")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ Test completed successfully!")
        print("📁 Output files:")
        for file in os.listdir("."):
            if file.startswith("test_output_") and (
                file.endswith(".mp3") or file.endswith(".wav")
            ):
                print(f"   - {file}")
    else:
        print("\n❌ Test failed!")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
