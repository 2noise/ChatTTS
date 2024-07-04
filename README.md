<div align="center">

<a href="https://trendshift.io/repositories/10489" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10489" alt="2noise%2FChatTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# ChatTTS
A generative speech model for daily dialogue.

[![Licence](https://img.shields.io/badge/LICENSE-CC%20BY--NC%204.0-green.svg?style=for-the-badge)](https://github.com/2noise/ChatTTS/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ChatTTS.svg?style=for-the-badge)](https://pypi.org/project/ChatTTS)

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/2noise/ChatTTS/blob/main/examples/ipynb/colab.ipynb)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Ud5Jxgx5yD)

**English** | [**简体中文**](docs/cn/README.md) | [**日本語**](docs/jp/README.md) | [**Русский**](docs/ru/README.md) | [**Español**](docs/es/README.md)

</div>

## Introduction
> [!Note]
> This repo contains the algorithm infrastructure and some simple examples.

> [!Tip]
> For the extended end-user products, please refer to the index repo [Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS/tree/en) maintained by the community.

ChatTTS is a text-to-speech model designed specifically for dialogue scenarios such as LLM assistant.

### Supported Languages
- [x] English
- [x] Chinese
- [ ] Coming Soon...

### Highlights
> You can refer to **[this video on Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)** for the detailed description.

1. **Conversational TTS**: ChatTTS is optimized for dialogue-based tasks, enabling natural and expressive speech synthesis. It supports multiple speakers, facilitating interactive conversations.
2. **Fine-grained Control**: The model could predict and control fine-grained prosodic features, including laughter, pauses, and interjections. 
3. **Better Prosody**: ChatTTS surpasses most of open-source TTS models in terms of prosody. We provide pretrained models to support further research and development.

### Dataset & Model
- The main model is trained with Chinese and English audio data of 100,000+ hours.
- The open-source version on **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** is a 40,000 hours pre-trained model without SFT.

### Roadmap
- [x] Open-source the 40k hour base model and spk_stats file.
- [x] Streaming audio generation.
- [ ] Open-source the 40k hour version with multi-emotion control.
- [ ] ChatTTS.cpp (new repo in `2noise` org is welcomed)

### Disclaimer
> [!Important]
> This repo is for academic purposes only.

It is intended for educational and research use, and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this repo, are for academic and research purposes only. The data obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

ChatTTS is a powerful text-to-speech system. However, it is very important to utilize this technology responsibly and ethically. To limit the use of ChatTTS, we added a small amount of high-frequency noise during the training of the 40,000-hour model, and compressed the audio quality as much as possible using MP3 format, to prevent malicious actors from potentially using it for criminal purposes. At the same time, we have internally trained a detection model and plan to open-source it in the future.

### Contact
> GitHub issues/PRs are always welcomed.

#### Formal Inquiries
For formal inquiries about the model and roadmap, please contact us at **open-source@2noise.com**.

#### Online Chat
##### 1. QQ Group (Chinese Social APP)
- **Group 1**, 808364215 (Full)
- **Group 2**, 230696694 (Full)
- **Group 3**, 933639842 (Full)
- **Group 4**, 608667975

##### 2. Discord Server
Join by clicking [here](https://discord.gg/Ud5Jxgx5yD).

## Get Started
### Clone Repo
```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

### Install requirements
#### 1. Install Directly
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Install from conda
```bash
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
```

#### Optional: Install TransformerEngine if using NVIDIA GPU (Linux only)
> [!Note]
> The installation process is very slow.

> [!Warning]
> The adaptation of TransformerEngine is currently under development and CANNOT run properly now. 
> Only install it on developing purpose.

```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### Optional: Install FlashAttention-2 (mainly NVIDIA GPU)
> [!Note]
> See supported devices at the [Hugging Face Doc](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2).

> [!Warning]
> Currently the FlashAttention-2 will slow down the generating speed according to [this issue](https://github.com/huggingface/transformers/issues/26990). 
> Only install it on developing purpose.

```bash
pip install flash-attn --no-build-isolation
```

### Quick Start
> Make sure you are under the project root directory when you execute these commands below.

#### 1. Launch WebUI
```bash
python examples/web/webui.py
```

#### 2. Infer by Command Line
> It will save audio to `./output_audio_n.mp3`

```bash
python examples/cmd/run.py "Your text 1." "Your text 2."
```

## Installation

1. Install the stable version from PyPI
```bash
pip install ChatTTS
```

2. Install the latest version from GitHub
```bash
pip install git+https://github.com/2noise/ChatTTS
```

3. Install from local directory in dev mode
```bash
pip install -e .
```

### Basic Usage

```python
import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts)

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

### Advanced Usage

```python
###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

###################################
# For word level manual control.

text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>Example: self introduction</h4></summary>

```python
inputs_en = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements like 
[uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '') # English is still experimental.

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
)

audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
torchaudio.save("output3.wav", torch.from_numpy(audio_array_en[0]), 24000)
```

<table>
<tr>
<td align="center">

**male speaker**

</td>
<td align="center">

**female speaker**

</td>
</tr>
<tr>
<td align="center">

[male speaker](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

</td>
<td align="center">

[female speaker](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)

</td>
</tr>
</table>


</details>

## FAQ

#### 1. How much VRAM do I need? How about infer speed?
For a 30-second audio clip, at least 4GB of GPU memory is required. For the 4090 GPU, it can generate audio corresponding to approximately 7 semantic tokens per second. The Real-Time Factor (RTF) is around 0.3.

#### 2. Model stability is not good enough, with issues such as multi speakers or poor audio quality.

This is a problem that typically occurs with autoregressive models (for bark and valle). It's generally difficult to avoid. One can try multiple samples to find a suitable result.

#### 3. Besides laughter, can we control anything else? Can we control other emotions?

In the current released model, the only token-level control units are `[laugh]`, `[uv_break]`, and `[lbreak]`. In future versions, we may open-source models with additional emotional control capabilities.

## Acknowledgements
- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS) and [valle](https://arxiv.org/abs/2301.02111) demostrate a remarkable TTS result by an autoregressive-style system.
- [fish-speech](https://github.com/fishaudio/fish-speech) reveals capability of GVQ as audio tokenizer for LLM modeling.
- [vocos](https://github.com/gemelo-ai/vocos) which is used as a pretrained vocoder.

## Special Appreciation
- [wlu-audio lab](https://audio.westlake.edu.cn/) for early algorithm experiments.

## Thanks to all contributors for their efforts
[![contributors](https://contrib.rocks/image?repo=2noise/ChatTTS)](https://github.com/2noise/ChatTTS/graphs/contributors)

<div align="center">

  ![counter](https://counter.seku.su/cmoe?name=chattts&theme=mbs)

</div>
