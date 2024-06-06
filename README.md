# ChatTTS-api-ui-docker 一键启动!

- 一行命令启动一个带有 Web 界面的 ChatTTS API 服务器
- 前提条件：支持 CUDA 的 GPU，Docker, 4GB GPU 内存

## Usage 使用方法

启动服务 (ps: 镜像压缩后 5.16 GB)
```bash
docker run --name chat_tts \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8080:8080 -p 8501:8501 \
  jackiexiao/chat_tts_api_ui
```

删除服务
```
docker rm -f chat_tts
```

- 等待服务启动完成后，访问 http://localhost:8501 即可查看 Web 界面。
- Web UI
  - URL 网页: http://localhost:8501
  - build with streamlit
- Api
  - API 地址: http://localhost:8080
  - API 文档: http://localhost:8080/docs
  - 客户端示例 client: `python client.py`
  - 采样率为 24000, 默认返回 mp3
  - build with fastapi

## 备注
- ChatTTS 的简略流程
    - Text -> GPT -> 离散 Token -> VAE Decoder（实际上用的 Hidden latents 不是离散 Token） -> Mel -> Vocos -> Audio
- 优点
    - 自回归模型：更高的自然度，可以达到类似 openai 的水平
    - 更大的模型更大的数据：通常使用 1-10万小时训练，模型在 0.3B-1B 之间
    - 少样本克隆：模型不需要通过微调就可以克隆新的音色（当然也可以微调获得更好的效果）但训练代码还没开源
    - 更高的可玩性/可控性：可以通过文本或者音频 prompt 控制发音风格，合成多样性的语音
    - 基于 token 而不基于拼音输入的方式可以非常容易扩展到多语种，不需要每个语种开发文本转拼音的工具包
- 缺点
    - 训练成本较高（开源版本4万小时, 275M参数量左右，据说有10万小时训练的更大参数量的版本）
    - 推理成本较高，推理速度较慢（4090单卡 RTF 0.3），需要后续优化
    - 目前合成不够稳定，容易出现多字，漏字，或者出现杂音。但后续迭代可解决
    - 目前 Badcase 较多，比如数字，英文字母，标点符号，多音字，部分英文单词常常念错，停顿上错误较多
    - 基于 token 的输入：缺点是如果训练数据量不够大，或者缺乏高质量数据，很多生僻字和特殊的人名地名将无法正确合成，多音字也容易出错。
    - 目前开源的版本基于4万小时训练的小参数量模型，音质和稳定性上比较一般
    - 无法生成超过 30 秒的音频，需要手动分句合成
    - 虽然是随机音色，但年龄都在20-45岁之间。没有更年轻或者更老的音色


## 类似项目
> 也提供了 UI 和 Api
- https://github.com/jianchang512/ChatTTS-ui

# ChatTTS
[**English**](./README.md) | [**中文简体**](./README_CN.md)

ChatTTS is a text-to-speech model designed specifically for dialogue scenario such as LLM assistant. It supports both English and Chinese languages. Our model is trained with 100,000+ hours composed of chinese and english. The open-source version on **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** is a 40,000 hours pre trained model without SFT.

For formal inquiries about model and roadmap, please contact us at **open-source@2noise.com**. You could join our QQ group: ~~808364215 (Full)~~ 230696694 (Group 2) for discussion. Adding github issues is always welcomed.

---
## Highlights
1. **Conversational TTS**: ChatTTS is optimized for dialogue-based tasks, enabling natural and expressive speech synthesis. It supports multiple speakers, facilitating interactive conversations.
2. **Fine-grained Control**: The model could predict and control fine-grained prosodic features, including laughter, pauses, and interjections. 
3. **Better Prosody**: ChatTTS surpasses most of open-source TTS models in terms of prosody. We provide pretrained models to support further research and development.

For the detailed description of the model, you can refer to **[video on Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)**

---

## Disclaimer

This repo is for academic purposes only. It is intended for educational and research use, and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this repo, are for academic and research purposes only. The data obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

ChatTTS is a powerful text-to-speech system. However, it is very important to utilize this technology responsibly and ethically. To limit the use of ChatTTS, we added a small amount of high-frequency noise during the training of the 40,000-hour model, and compressed the audio quality as much as possible using MP3 format, to prevent malicious actors from potentially using it for criminal purposes. At the same time, we have internally trained a detection model and plan to open-source it in the future.


---
## Usage

<h4>Basic usage</h4>

```python
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["PUT YOUR TEXT HERE",]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

<h4>Advanced usage</h4>

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

wav = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

###################################
# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wav = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>Example: self introduction</h4></summary>

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
[male speaker](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

[female speaker](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)
</details>

---
## Roadmap
- [x] Open-source the 40k hour base model and spk_stats file
- [ ] Open-source VQ encoder and Lora training code
- [ ] Streaming audio generation without refining the text*
- [ ] Open-source the 40k hour version with multi-emotion control
- [ ] ChatTTS.cpp maybe? (PR or new repo are welcomed.)
 
----
## FAQ

##### How much VRAM do I need? How about infer speed?
For a 30-second audio clip, at least 4GB of GPU memory is required. For the 4090 GPU, it can generate audio corresponding to approximately 7 semantic tokens per second. The Real-Time Factor (RTF) is around 0.3.

##### model stability is not good enough, with issues such as multi speakers or poor audio quality.

This is a problem that typically occurs with autoregressive models(for bark and valle). It's generally difficult to avoid. One can try multiple samples to find a suitable result.

##### Besides laughter, can we control anything else? Can we control other emotions?

In the current released model, the only token-level control units are [laugh], [uv_break], and [lbreak]. In future versions, we may open-source models with additional emotional control capabilities.

---
## Acknowledgements
- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS) and [valle](https://arxiv.org/abs/2301.02111) demostrate a remarkable TTS result by a autoregressive-style system.
- [fish-speech](https://github.com/fishaudio/fish-speech) reveals capability of GVQ as audio tokenizer for LLM modeling.
- [vocos](https://github.com/gemelo-ai/vocos) which is used as a pretrained vocoder.

---
## Special Appreciation
- [wlu-audio lab](https://audio.westlake.edu.cn/) for early algorithm experiments.
