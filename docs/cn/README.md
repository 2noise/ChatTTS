<div align="center">

<a href="https://trendshift.io/repositories/10489" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10489" alt="2noise%2FChatTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# ChatTTS
一款适用于日常对话的生成式语音模型。

[![Licence](https://img.shields.io/badge/LICENSE-CC%20BY--NC%204.0-green.svg?style=for-the-badge)](https://github.com/2noise/ChatTTS/blob/main/LICENSE)

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/2noise/ChatTTS/blob/main/examples/ipynb/colab.ipynb)

[**English**](../../README.md) | **简体中文** | [**日本語**](../jp/README.md) | [**Русский**](../ru/README.md)

</div>

> [!NOTE]
> 注意此版本可能不是最新版，所有内容请以英文版为准。

## 简介

ChatTTS 是一款专门为对话场景（例如 LLM 助手）设计的文本转语音模型。

### 支持的语种

- [x] 英语
- [x] 中文
- [ ] 敬请期待...

### 亮点

> 你可以参考 **[Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)** 上的这个视频，了解本项目的详细情况。

1. **对话式 TTS**: ChatTTS 针对对话式任务进行了优化，能够实现自然且富有表现力的合成语音。它支持多个说话者，便于生成互动式对话。
2. **精细的控制**: 该模型可以预测和控制精细的韵律特征，包括笑声、停顿和插入语。
3. **更好的韵律**: ChatTTS 在韵律方面超越了大多数开源 TTS 模型。我们提供预训练模型以支持进一步的研究和开发。

### 数据集和模型

- 主模型使用了 100,000+ 小时的中文和英文音频数据进行训练。
- **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** 上的开源版本是一个在 40,000 小时数据上进行无监督微调的预训练模型。

### 路线图

- [x] 开源 4 万小时基础模型和 spk_stats 文件
- [ ] 开源 VQ 编码器和 Lora 训练代码
- [ ] 无需细化文本即可进行流式音频生成
- [ ] 开源具有多情感控制功能的 4 万小时版本
- [ ] 也许会有 ChatTTS.cpp ？(欢迎 PR 或新建仓库)

### 免责声明

> [!Important]
> 此仓库仅供学术用途。

本项目旨在用于教育和研究目的，不适用于任何商业或法律目的。作者不保证信息的准确性、完整性和可靠性。此仓库中使用的信息和数据仅供学术和研究目的。数据来自公开来源，作者不声称对数据拥有任何所有权或版权。

ChatTTS 是一款强大的文本转语音系统。但是，负责任和道德地使用这项技术非常重要。为了限制 ChatTTS 的使用，我们在 40,000 小时模型的训练过程中添加了少量高频噪声，并使用 MP3 格式尽可能压缩音频质量，以防止恶意行为者将其用于犯罪目的。同时，我们内部训练了一个检测模型，并计划在未来开源它。

### 联系方式

> 欢迎随时提交 GitHub issues/PRs。

#### 合作洽谈

如需就模型和路线图进行合作洽谈，请发送邮件至 **open-source@2noise.com**。

#### 线上讨论

##### 1. 官方 QQ 群

- **群 1**, 808364215 (已满)
- **群 2**, 230696694 (已满)
- **群 3**, 933639842

## 安装教程 (丰富中)

> 将在近期上传至 pypi，详情请查看 https://github.com/2noise/ChatTTS/issues/269 上的讨论。

#### 1. 使用源代码安装

```bash
pip install git+https://github.com/2noise/ChatTTS
```

#### 2. 使用 conda 安装

```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
```

## 使用教程

### 安装依赖

```bash
pip install --upgrade -r requirements.txt
```

### 快速开始

#### 1. 启动 WebUI

```bash
python examples/web/webui.py
```

#### 2. 使用命令行

> 生成的音频将保存至 `./output_audio_xxx.wav`

```bash
python examples/cmd/run.py "Please input your text."
```

### 基础用法

```python
import ChatTTS
from IPython.display import Audio
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR TEXT HERE",]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

### 进阶用法

```python
###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()

params_infer_code = {
  'spk_emb': rand_spk, # add sampled speaker 
  'temperature': .3, # using custom temperature
  'top_P': 0.7, # top P decode
  'top_K': 20, # top K decode
}

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wavs = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

###################################
# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>示例: 自我介绍</h4></summary>

```python
inputs_en = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements [laugh]like like 
[uv_break]laughter[laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '') # English is still experimental.

params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_4]'
} 
# audio_array_cn = chat.infer(inputs_cn, params_refine_text=params_refine_text)
audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
torchaudio.save("output3.wav", torch.from_numpy(audio_array_en[0]), 24000)
```

[男性音色](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

[女性音色](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)

</details>

## 常见问题

#### 1. 我需要多少 VRAM？ 推理速度如何？

对于 30 秒的音频片段，至少需要 4GB 的 GPU 内存。 对于 4090 GPU，它可以每秒生成大约 7 个语义 token 对应的音频。实时因子 (RTF) 约为 0.3。

#### 2. 模型稳定性不够好，存在多个说话者或音频质量差等问题。

这是一个通常发生在自回归模型（例如 bark 和 valle）中的问题，通常很难避免。可以尝试多个样本以找到合适的结果。

#### 3. 除了笑声，我们还能控制其他东西吗？我们能控制其他情绪吗？

在当前发布的模型中，可用的 token 级控制单元是 `[laugh]`, `[uv_break]` 和 `[lbreak]`。未来的版本中，我们可能会开源具有更多情绪控制功能的模型。

## 致谢

- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS) 和 [valle](https://arxiv.org/abs/2301.02111) 通过自回归式系统展示了非凡的 TTS 效果。
- [fish-speech](https://github.com/fishaudio/fish-speech) 揭示了 GVQ 作为 LLM 建模的音频分词器的能力。
- [vocos](https://github.com/gemelo-ai/vocos) vocos 被用作预训练声码器。

## 特别鸣谢

- [wlu-audio lab](https://audio.westlake.edu.cn/) 对于早期算法实验的支持。

## 相关资源

- [Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS) 一个 ChatTTS 的资源汇总列表。

## 贡献者列表

[![contributors](https://contrib.rocks/image?repo=2noise/ChatTTS)](https://github.com/2noise/ChatTTS/graphs/contributors)

## 项目浏览量

<div align="center">

![counter](https://counter.seku.su/cmoe?name=chattts&theme=mbs)

</div>
