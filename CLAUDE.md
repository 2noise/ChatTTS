# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChatTTS is a generative speech model designed specifically for dialogue scenarios such as LLM assistant conversations. It supports English and Chinese languages with fine-grained control over prosodic features including laughter, pauses, and interjections.

## Development Commands

### Installation
```bash
# Install from local directory in development mode
pip install -e .

# Install dependencies
pip install --upgrade -r requirements.txt

# Optional: Install vLLM for acceleration (Linux only)
pip install safetensors vllm==0.2.7 torchaudio
```

### Running Examples
```bash
# Launch WebUI
python examples/web/webui.py

# Command line inference
python examples/cmd/run.py "Your text 1." "Your text 2."

# Streaming inference
python examples/cmd/stream.py
```

### Testing
```bash
# Run demo script
python demo.py

# Test advanced features
python advanced.py
```

## Code Architecture

### Core Components
- **`ChatTTS.Chat`** (`ChatTTS/core.py`): Main orchestrator class for TTS pipeline
- **`ChatTTS.model.GPT`** (`ChatTTS/model/gpt.py`): Language model backbone using Llama architecture
- **`ChatTTS.model.DVAE`** (`ChatTTS/model/dvae.py`): Discrete Variational Autoencoder for audio encoding
- **`ChatTTS.model.Tokenizer`** (`ChatTTS/model/tokenizer.py`): Text tokenization with special TTS tokens
- **`ChatTTS.model.Speaker`** (`ChatTTS/model/speaker.py`): Speaker embedding and voice control
- **`ChatTTS.norm.Normalizer`** (`ChatTTS/norm.py`): Text normalization and homophone replacement

### TTS Pipeline
1. Text input → Normalization → Tokenization → GPT generation → DVAE decoding → Vocos synthesis → Audio output
2. Supports streaming generation and text refinement
3. Multi-speaker capabilities with fine-grained prosody control

### Configuration System
- Configuration managed via `ChatTTS.config.Config` dataclass
- Model paths, GPT parameters, DVAE settings, and vocoder config
- Automatic model downloading from HuggingFace or local cache

## Key Development Patterns

### Model Loading
```python
chat = ChatTTS.Chat()
# Local models
chat.load(source="local", compile=True)
# HuggingFace models
chat.load(source="huggingface")
# Custom path
chat.load(source="custom", custom_path="/path/to/models")
```

### Inference with Controls
```python
# Basic inference
wavs = chat.infer(["Your text here"])

# Advanced controls
params = ChatTTS.Chat.InferCodeParams(
    spk_emb=speaker_embedding,
    temperature=0.3,
    top_P=0.7
)
wavs = chat.infer(texts, params_infer_code=params)

# Streaming generation
for wav_chunk in chat.infer(text, stream=True):
    # Process streaming audio
```

### Speaker Management
```python
# Sample random speaker
speaker_emb = chat.sample_random_speaker()

# Extract speaker from audio
speaker_emb = chat.sample_audio_speaker(audio_array)
```

## Important Notes

### Model Licensing
- Code: AGPLv3+ license
- Models: CC BY-NC 4.0 license (academic/research use only)

### Hardware Requirements
- Minimum 4GB GPU memory for 30-second audio
- Supports CPU, CUDA, MPS (Apple Silicon), and NPU devices
- vLLM acceleration requires Linux + CUDA

### Performance Optimizations
- Set `compile=True` for better performance (requires CUDA)
- Use `use_vllm=True` for vLLM acceleration
- Enable `enable_cache=True` for faster repeated inference

### Special Tokens for Prosody Control
- `[laugh]`, `[laugh_0]`, `[laugh_1]`, `[laugh_2]` - Laughter control
- `[break_0]` to `[break_7]` - Pause duration control
- `[uv_break]`, `[lbreak]` - Voice break controls
- `[oral_0]` to `[oral_9]` - Oral quality control

## File Structure
```
ChatTTS/
├── core.py              # Main Chat class and inference pipeline
├── model/
│   ├── gpt.py           # GPT language model
│   ├── dvae.py          # Discrete VAE for audio
│   ├── tokenizer.py     # Text tokenization
│   ├── speaker.py       # Speaker embedding management
│   └── velocity/        # vLLM integration
├── config/
│   └── config.py        # Configuration dataclasses
├── utils/               # Utility functions
├── examples/            # Usage examples
└── res/                 # Resource files (homophones, checksums)
```

## Troubleshooting

- **Memory issues**: Use streaming generation for long texts
- **Audio quality**: Try multiple samples for autoregressive model stability
- **Device compatibility**: Vocos runs on CPU for MPS/NPU devices
- **Model loading**: Check network connection for HuggingFace downloads