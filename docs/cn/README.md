# ChatTTS
> [!NOTE]
> 以下内容可能不是最新，一切请以英文版为准。

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)

[**English**](../../README.md) | **简体中文** | [**日本語**](../jp/README.md) | [**Русский**](../ru/README.md)

ChatTTS是专门为对话场景设计的文本转语音模型，例如LLM助手对话任务。它支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。在HuggingFace中开源的版本为4万小时训练且未SFT的版本.

如需就模型进行正式商业咨询，请发送邮件至 **open-source@2noise.com**。对于中文用户，您可以加入我们的QQ群：~~808364215 (已满)~~ ~~230696694 (二群)~~ 933639842 (三群) 进行讨论。同时欢迎在GitHub上提出问题。如果遇到无法使用 **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** 的情况,可以在 [modelscope](https://www.modelscope.cn/models/pzc163/chatTTS) 上进行下载. 

---
## 亮点
1. **对话式 TTS**: ChatTTS针对对话式任务进行了优化，实现了自然流畅的语音合成，同时支持多说话人。
2. **细粒度控制**: 该模型能够预测和控制细粒度的韵律特征，包括笑声、停顿和插入词等。
3. **更好的韵律**: ChatTTS在韵律方面超越了大部分开源TTS模型。同时提供预训练模型，支持进一步的研究。

对于模型的具体介绍, 可以参考B站的 **[宣传视频](https://www.bilibili.com/video/BV1zn4y1o7iV)**

---

## 免责声明
本文件中的信息仅供学术交流使用。其目的是用于教育和研究，不得用于任何商业或法律目的。作者不保证信息的准确性、完整性或可靠性。本文件中使用的信息和数据，仅用于学术研究目的。这些数据来自公开可用的来源，作者不对数据的所有权或版权提出任何主张。

ChatTTS是一个强大的文本转语音系统。然而，负责任地和符合伦理地利用这项技术是非常重要的。为了限制ChatTTS的使用，我们在4w小时模型的训练过程中添加了少量额外的高频噪音，并用mp3格式尽可能压低了音质，以防不法分子用于潜在的犯罪可能。同时我们在内部训练了检测模型，并计划在未来开放。

---
## 安装

```
pip install git+https://github.com/2noise/ChatTTS
```
## 用法

<h4>基本用法</h4>

```python
import ChatTTS
from IPython.display import Audio
import torchaudio

chat = ChatTTS.Chat()
chat.load_models(compile=False) # 设置为True以获得更快速度

texts = ["在这里输入你的文本",]

wavs = chat.infer(texts, use_decoder=True)

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

<h4>进阶用法</h4>

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
# use_decoder=False to infer faster with a bit worse quality
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_infer_code=params_infer_code, use_decoder=False)

torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>自我介绍样例</h4></summary>

```python
inputs_cn = """
chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。
chat T T S 不仅能够生成自然流畅的语音，还能控制[laugh]笑声啊[laugh]，
停顿啊[uv_break]语气词啊等副语言现象[uv_break]。这个韵律超越了许多开源模型[uv_break]。
请注意，chat T T S 的使用应遵守法律和伦理准则，避免滥用的安全风险。[uv_break]'
""".replace('\n', '')

params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_4]'
} 
audio_array_cn = chat.infer(inputs_cn, params_refine_text=params_refine_text)
# audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)

torchaudio.save("output3.wav", torch.from_numpy(audio_array_cn[0]), 24000)
```
[男说话人](https://github.com/2noise/ChatTTS/assets/130631963/bbfa3b83-2b67-4bb6-9315-64c992b63788)

[女说话人](https://github.com/2noise/ChatTTS/assets/130631963/e061f230-0e05-45e6-8e4e-0189f2d260c4)
</details>


---
## 计划路线
- [x] 开源4w小时基础模型和spk_stats文件
- [ ] 开源VQ encoder和Lora 训练代码
- [ ] 在非refine text情况下, 流式生成音频*
- [ ] 开源多情感可控的4w小时版本
- [ ] ChatTTS.cpp maybe? (欢迎社区PR或独立的新repo)

---
## 常见问题

##### 连不上HuggingFace
请使用[modelscope](https://www.modelscope.cn/models/pzc163/chatTTS)的版本. 并设置cache的位置:
```python
chat.load_models(source='local', local_path='你的下载位置')
```

##### 我要多少显存? Infer的速度是怎么样的?
对于30s的音频, 至少需要4G的显存. 对于4090, 1s生成约7个字所对应的音频. RTF约0.3.

##### 模型稳定性似乎不够好, 会出现其他说话人或音质很差的现象.
这是自回归模型通常都会出现的问题. 说话人可能会在中间变化, 可能会采样到音质非常差的结果, 这通常难以避免. 可以多采样几次来找到合适的结果.

##### 除了笑声还能控制什么吗? 还能控制其他情感吗?
在现在放出的模型版本中, 只有[laugh]和[uv_break], [lbreak]作为字级别的控制单元. 在未来的版本中我们可能会开源其他情感控制的版本.

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=2noise/ChatTTS&type=Date)](https://star-history.com/#2noise/ChatTTS&Date)

---
## 致谢
- [bark](https://github.com/suno-ai/bark),[XTTSv2](https://github.com/coqui-ai/TTS)和[valle](https://arxiv.org/abs/2301.02111)展示了自回归任务用于TTS任务的可能性.
- [fish-speech](https://github.com/fishaudio/fish-speech)一个优秀的自回归TTS模型, 揭示了GVQ用于LLM任务的可能性.
- [vocos](https://github.com/gemelo-ai/vocos)作为模型中的vocoder.

---
## 特别致谢
- [wlu-audio lab](https://audio.westlake.edu.cn/)为我们提供了早期算法试验的支持.
