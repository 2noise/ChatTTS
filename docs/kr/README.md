<div align="center">

<a href="https://trendshift.io/repositories/10489" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10489" alt="2noise%2FChatTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# ChatTTS
일상 대화를 위한 생성형 음성 모델입니다.

[![Licence](https://img.shields.io/github/license/2noise/ChatTTS?style=for-the-badge)](https://github.com/2noise/ChatTTS/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ChatTTS.svg?style=for-the-badge&color=green)](https://pypi.org/project/ChatTTS)

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/2noise/ChatTTS/blob/main/examples/ipynb/colab.ipynb)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Ud5Jxgx5yD)

[**English**](../../README.md) | [**简体中文**](../cn/README.md) | [**日本語**](../jp/README.md) | [**Русский**](../ru/README.md) | [**Español**](../es/README.md) | [**Français**](../fr/README.md) | **한국어**

</div>

> [!NOTE]
> 이 문서는 최신 버전이 아닐 수 있습니다. [영어 문서](../../README.md)를 기준으로 작업하는 것을 권장합니다.

## 프로젝트 소개

> [!Note]
> 이 저장소에는 알고리즘 구조와 간단한 예시들이 포함되어 있습니다.

> [!Tip]
> 이 프로젝트에서 파생된 프로젝트는 커뮤니티가 유지 관리하는 커뮤니티[Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS)를 참조하시길 바랍니다.

ChatTTS는 대화 기반 작업(예: LLM 어시스턴트)을 위해 설계된 텍스트-음성 변환(TTS) 모델입니다.

### 지원 언어

- [x] 영어
- [x] 중국어
- [ ] 계속 추가 예정...

### 프로젝트 특징

> 이 프로젝트의 내용은 **[Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)**에서 제공되는 비디오를 참조하시길 바랍니다.

1. **대화형 TTS**: ChatTTS는 대화 기반 작업에 최적화되어 자연스럽고 표현력 있는 음성 합성을 구현합니다. 다중 화자를 지원하여 상호작용적인 대화를 가능하게 합니다.
2. **세밀한 제어**: 이 모델은 웃음, 일시 정지, 삽입어 등 세밀한 운율적 특징을 예측하고 제어할 수 있습니다.
3. **향상된 운율**: ChatTTS는 운율 측면에서 대부분의 오픈 소스 TTS 모델을 능가하며, 추가 연구와 개발을 지원하기 위해 사전 훈련된 모델을 제공합니다.

### 데이터셋 및 모델
> [!Important]
> 공개된 모델은 학술 목적으로만 사용 가능합니다.

- 주요 모델은 100,000+ 시간의 중국어 및 영어 오디오 데이터를 사용하여 훈련되었습니다.
- **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)**에서 제공되는 오픈 소스 버전은 40,000시간의 사전 훈련된 모델로, SFT가 적용되지 않았습니다.

### 로드맵
- [x] 40,000시간 기반 모델과 spk_stats 파일 오픈 소스화.
- [x] 스트리밍 오디오 생성.
- [x] DVAE 인코더와 제로 샷 추론 코드 오픈 소스화.
- [ ] 다중 감정 제어 기능.
- [ ] ChatTTS.cpp (`2noise` 조직 내의 새로운 저장소를 환영합니다.)

### 라이선스

#### 코드
코드는 `AGPLv3+` 라이선스를 따릅니다.

#### 모델
모델은 `CC BY-NC 4.0` 라이선스로 공개되었습니다. 이 모델은 교육 및 연구 목적으로만 사용되며, 상업적 또는 불법적 목적으로 사용되어서는 안 됩니다. 저자들은 정보의 정확성, 완전성, 신뢰성을 보장하지 않습니다. 이 저장소에서 사용된 정보와 데이터는 학술 및 연구 목적으로만 사용되며, 공개적으로 이용 가능한 출처에서 얻은 데이터입니다. 저자들은 데이터에 대한 소유권 또는 저작권을 주장하지 않습니다.

### 면책 조항

ChatTTS는 강력한 텍스트-음성 변환 시스템입니다. 그렇기에 기술을 책임감 있고 윤리적으로 사용하는 것은 아주 중요합니다. ChatTTS의 악용을 방지하기 위해 40,000시간 모델의 훈련 중 소량의 고주파 노이즈를 추가하고, 오디오 품질을 최대한 압축하여 MP3 형식으로 제공했습니다. 또한, 우리는 내부적으로 탐지 모델을 훈련했으며, 추후 이를 오픈 소스화할 계획입니다.

### 연락처
> GitHub 이슈/PR은 언제든지 환영합니다.

#### 공식 문의
모델 및 로드맵에 대한 공식적인 문의는 **open-source@2noise.com**으로 연락해 주십시오.

#### 온라인 채팅
##### 1. QQ Group (Chinese Social APP)
- **Group 1**, 808364215
- **Group 2**, 230696694
- **Group 3**, 933639842
- **Group 4**, 608667975

##### 2. Discord 서버
[이곳](https://discord.gg/Ud5Jxgx5yD)를 클릭하여 참여하십시오.

## 시작하기
### 레포지토리 클론
```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

### 의존성 설치
#### 1. 직접 설치
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Conda에서 설치
```bash
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
```

#### 선택사항: vLLM 설치 (Linux 전용)
```bash
pip install safetensors vllm==0.2.7 torchaudio
```

#### 권장되지 않는 선택사항: NVIDIA GPU 사용 시 TransformerEngine 설치 (Linux 전용)
> [!Warning]
> 설치하지 마십시오!
> TransformerEngine의 적응 작업은 현재 개발 중이며, 아직 제대로 작동하지 않습니다.
> 개발 목적으로만 설치하십시오. 자세한 내용은 #672 및 #676에서 확인할 수 있습니다.

> [!Note]
> 설치 과정은 매우 느립니다.

```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### 권장되지 않는 선택사항: FlashAttention-2 설치 (주로 NVIDIA GPU)
> [!Warning]
> 설치하지 마십시오!
> 현재 FlashAttention-2는 [이 이슈](https://github.com/huggingface/transformers/issues/26990)에 따르면 생성 속도를 저하시킵니다.
> 개발 목적으로만 설치하십시오.

> [!Note]
> 지원되는 장치는 [Hugging Face 문서](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)에서 확인할 수 있습니다.

```bash
pip install flash-attn --no-build-isolation
```

### 빠른 시작
> 아래 명령어를 실행할 때 반드시 프로젝트 루트 디렉토리에서 실행하십시오.

#### 1. WebUI 실행
```bash
python examples/web/webui.py
```

#### 2. 커맨드 라인에서 추론
> 오디오는 `./output_audio_n.mp3`에 저장됩니다.

```bash
python examples/cmd/run.py "Your text 1." "Your text 2."
```

## 설치 방법

1. PyPI에서 안정 버전 설치
```bash
pip install ChatTTS
```

2. GitHub에서 최신 버전 설치
```bash
pip install git+https://github.com/2noise/ChatTTS
```

3. 로컬 디렉토리에서 개발 모드로 설치
```bash
pip install -e .
```

### 기본 사용법

```python
import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # 성능 향상을 위해 True로 설정 가능

texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    """
    torchaudio의 버전에 따라 첫 번째 줄이 작동할 수 있고, 다른 버전에서는 두 번째 줄이 작동할 수 있습니다.
    """
    try:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)
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
"""
In some versions of torchaudio, the first line works but in other versions, so does the second line.
"""
try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)
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
torchaudio.save("self_introduction_output.wav", torch.from_numpy(audio_array_en[0]), 24000)
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

#### 1. VRAM이 얼마나 필요한가요? 추론 속도는 어느 정도인가요?
30초 길이의 오디오 클립을 생성하려면 최소 4GB의 GPU 메모리가 필요합니다. 4090 GPU의 경우 초당 약 7개의 의미 토큰에 해당하는 오디오를 생성할 수 있습니다. 실시간 인자(RTF)는 약 0.3입니다.

#### 2. 모델의 안정성은 불안정하며, 화자가 많은 경우 및 오디오 품질이 저하되는 이슈 존재.

이는 일반적으로 autoregressive 모델(bark 및 valle 등)에서 발생하는 불가피한 문제입니다. 현재로선 여러 번 샘플링하여 적절한 결과를 찾는 것이 최선입니다.

#### 3. 웃음 뿐 아니라 다른 감정도 표현할 수 있나요?

현재 공개된 모델에서는 제어 가능한 토큰은 `[laugh]`, `[uv_break]`, `[lbreak]`입니다. 향후 버전의 모델에서는 추가적인 감정 제어 기능 포함하여 오픈 소스로 제공할 계획입니다.

## 감사의 인사
- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS), [valle](https://arxiv.org/abs/2301.02111)는 autoregressive 방식의 시스템으로 뛰어난 TTS 성능을 보여주었습니다.
- [fish-speech](https://github.com/fishaudio/fish-speech)는 LLM 모델링을 위한 오디오 토크나이저로서 GVQ의 능력을 보여주었습니다.
- [vocos](https://github.com/gemelo-ai/vocos)는 사전 훈련된 vocoder로 사용되었습니다.

## 특별 감사
- 초기 알고리즘 실험을 위한 [wlu-audio lab](https://audio.westlake.edu.cn/)에 감사의 말씀을 전합니다.

## 모든 기여자들의 노고에 감사드립니다
[![contributors](https://contrib.rocks/image?repo=2noise/ChatTTS)](https://github.com/2noise/ChatTTS/graphs/contributors)

<div align="center">

  ![counter](https://counter.seku.su/cmoe?name=chattts&theme=mbs)

</div>
