# ChatTTS Demo Scripts

This directory contains demo scripts for ChatTTS text-to-speech functionality.

## Available Demos

### 1. Basic Demo
- **File**: `demo.py`
- **Description**: Simple text-to-speech conversion with Chinese text
- **Usage**:
  ```bash
  python demo.py
  ```
- **Output**: WAV files (`output0.wav`, etc.)

### 2. Advanced Demo
- **File**: `advanced.py`
- **Description**: Advanced features including speaker control and English text
- **Usage**:
  ```bash
  python advanced.py
  ```
- **Output**: WAV files with random speaker embeddings

### 3. Interactive Prompt to MP3 Demo
- **File**: `prompt_to_mp3.py`
- **Description**: Interactive script that accepts user input and saves as MP3
- **Features**:
  - Interactive text input
  - Speaker control (random, custom, or from audio)
  - MP3 output format
  - Support for special tokens like `[laugh]`, `[break_4]`, `[uv_break]`
- **Usage**:
  ```bash
  python prompt_to_mp3.py
  ```
- **Output**: MP3 files with user-specified names

### 4. Test Script
- **File**: `test_prompt_to_mp3.py`
- **Description**: Non-interactive test of the MP3 functionality
- **Usage**:
  ```bash
  python test_prompt_to_mp3.py
  ```
- **Output**: Test MP3 and WAV files

## Special Tokens for Prosody Control

You can use these special tokens in your text for fine-grained control:

- **Laughter**: `[laugh]`, `[laugh_0]`, `[laugh_1]`, `[laugh_2]`
- **Pauses**: `[break_0]` to `[break_7]` (different durations)
- **Voice breaks**: `[uv_break]`, `[lbreak]`
- **Oral quality**: `[oral_0]` to `[oral_9]`

## Example Usage

```python
# Text with special tokens
text = "Hello! [laugh] This is a test. [break_4] How are you today?"

# Generate speech
wavs = chat.infer([text])

# Save as MP3
save_as_mp3(wavs[0], "output.mp3")
```

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r ../requirements.txt
```

## Notes

- The first run will download models (about 400MB total)
- Models are cached locally for future use
- MP3 conversion requires `pydub` library
- For better performance, set `compile=True` when loading models (requires CUDA)