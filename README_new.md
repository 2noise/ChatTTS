# 揭秘最像真人的TTS：从方法看ChatTTS如何工作

## 前言
作为一只模型移植小工，我经常在 github 冲浪🏄‍♀️看近期有什么爆款项目可以移植一波冲业绩。 两周前开源的 ChatTTS 是近期最吸睛的，文本转语音效果超像真人（简直惊悚），大量技术博主发布了实测视频、使用教程、整合包和功能参数介绍的相关内容。

但是！！竟然没有从技术角度分析how it works的！！进仓库一看，只有推理代码和模型链接，没有 paper 没有 technical report💔，readme 只有基本的使用例子，然而收割了 23k+ star。破大防啦😭，平时不出意外（虽然老出意外）本小工都是当模型是黑盒流水线作业的= = 这...这波要直接看代码看它到底有些什么妖术生成效果这么像人了呀。。。更震惊我的是：

**它把 Llama 用于TTS**

**同一个 Llama 模型既用于文本的口语化（文生文，这很常见）又用于语音 code 生成（文生音频？？？**

看了半天 readme，只在 acknowledgement看到一句“fish-speech reveals capability of ** GVQ as audio tokenizer for LLM modeling **.”，所以到底是怎么回事捏？这个 fish-speech 也没有找到对应的 report（头大

整理一波自己的分析， 抛砖引玉，希望路过的大佬多多交流赐教！

本文内容包括：

- ChatTTS介绍、功能使用、参数解读
- ChatTTS推理过程
- 对ChatTTS训练过程的一点推测（将LLM用于TTS，如何打通语音和文本）

## ChatTTS介绍和功能使用
### 简介
ChatTTS 是一个为对话场景设计的文本转语音（Text-To-Speech）模型。之所以火爆，是因为它在语音合成领域带来了一些创新和亮点：

- 对话式 TTS 优化，很像真人：ChatTTS 针对对话式任务进行了特别优化，能够实现韵律流畅自然的声音合成效果。
- 细粒度控制：模型能够预测口头表达特征，也可以手动设定笑声、暂停、语速等，使生成的语音更加贴近人类的自然表达。ChatTTS 在韵律处理上的卓越性能，超越了许多其他开源 TTS 模型，提供了更加自然和富有表现力的语音输出。
- 高斯采样说话人：不满意可以无限抽新音色
- 支持中英文
- 使用了大规模数据训练：最大的模型使用了超过10万小时的中英文数据进行训练，当前开源的模型为4 万小时训练版。

实测视频见 项目作者的 b 站投稿和各博主的推文。

### 功能使用
直接上代码.
```python
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["PUT YOUR TEXT HERE",] # 文本内容

# 说话人采样：从高斯分布随机采样一个[1, 768]的 speaker embedding，与生成语音的音色有关
rand_spk = chat.sample_random_speaker() 

"""
refine_text的参数设置，影响文本的口语化
"""
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
  'top_P': 0.7, 
  'top_K': 20, 
  'temperature': 0.7, 
  'repetition_penalty': 1.0,
  'max_new_token': 384
} 

# infer_code的参数设置，控制语音的生成
params_infer_code = {
  'spk_emb': rand_spk, # 设置说话人
  'temperature': .3,
  'top_P': 0.7, 
  'top_K': 20, 
}

# skip_refine_text控制是否添加口语化表达。若设置为 false，不走refine_text，params_refine_text不生效
wav = chat.infer(texts, skip_refine_text=True, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
```

### refine_text（输入文本 -> 加了口头词的文本）参数解读
prompt：特殊 token 构成的字符串，控制文本内容的口头化的程度。具体参考下面的说明：

- [oral_i]：控制文本口语化程度，i范围为 0-9，数字越大，添加的“啊”、“就是”、“那”之类的口头词越多；
- [laugh_i] : 控制文本中添加笑声的程度，i范围为 0-9，值越大，笑声越多；
- [break_i] : 控制文本中添加停顿的程度，i范围为 0-9，值越大，停顿越多；
可能还有其他的特殊 token，可以print 下 tokenizer 看看。

其他参数是LLM 的常见设置，直接让 Kimi 解释一下：
- top_P: 这个参数控制着模型在生成下一个词时考虑的词汇范围。top_P是一个概率阈值，意味着模型将只考虑累积概率达到这个阈值的词汇。例如，如果设置为0.7，那么模型将只考虑前30%的词汇。调大这个值有助于生成更多样化的文本。
- top_K: 这个参数与top_P类似，但它是直接限制生成词汇的数量。top_K表示模型在生成下一个词时只考虑概率最高的K个词汇。
- temperature: 这个参数控制着生成文本时的随机性。temperature的值越低，生成的文本就越倾向于选择概率最高的词汇，从而更加确定性；值越高，生成的文本就越随机。
- repetition_penalty: 这个参数用于控制文本中重复词汇的出现频率。repetition_penalty为1.0时，表示没有对重复词汇进行惩罚，即词汇可以自由重复。如果这个值大于1，那么重复词汇的概率会降低，从而减少文本中的重复。
- max_new_token: 这个参数设置了模型生成文本时最多可以生成的新词汇数量。

这些参数可以根据不同的需求进行调整，以达到更好的文本生成效果。

### infer_code（文本 ->语音 code）参数解读
- spk_emb: 可选。高斯采样得到的一个[1, 768]的向量，控制说话人的音色。

上一小节中refine_text的参数，在这一步均可以再设置，默认值有差异。

## 推理过程分析
推理代码都在下图的chat.infer()函数中，其中依次调用了

- refine_text（）：从输入文本到加了口头词的文本，添加语气词连词。主要用了 Tokenizer、emb_text、LLama model、head_text。
- infer_code（）：从文本到语音 code（code怎么翻译= =）。主要用了Tokenizer、emb_text、LLama model、head_code。与refine_text（）的差异在于使用了不同的 LM head。
- 调用decoder 或 DVAE ：从语音 code 到声学特征
- 调用vocos： 由声学特征重构音频

### 推理过程 overview
Tokenizer（文本 tokenizer）额外加了描述语音的特殊token。比如：[oral_1]、[laugh_5]、[break_3]，数字表示停顿、语气的程度。[empty_spk]、[spk_emb]表示说话人。
<img width="893" alt="Screenshot 2024-06-13 at 22 26 43" src="https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/f0d1cc66-49b7-478a-a1c8-7e545b8eba6e">
有一个我比较疑惑的地方。refine_text（）中，Llama输出的token序列可能带了特殊token。原方案会在此处移除掉模型生成的语气token，在 infer_code前文本中只会或多或少加了“嗯、啊、那么...”之类的语气词。强行解释一下就是，params_refine_text中的 prompt 中的语气 token 可以影响模型的输出，让模型输出更多嗯啊之类的语气词，但最后的结果会移除了输出中的语气token ？？？

简单起见，下面的分析和图示中不展示刚刚给出说明的可配置参数，仅对必需的输入输出做说明。

### Refine text 文本口语化
输入: 用户输入文本
输出: 加了语气词 (嗯、吧、那...) 的文本。
这步可跳过，直接用原始输入/手动加语气韵律特殊 token 去执行 infer_code。
过程图上已经很清楚了。

<img width="480" alt="Screenshot 2024-06-13 at 20 19 57" src="https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/d0169bac-9a3c-4777-9c20-f3ebee01b063">

### Infer code 文本到语音 code
输入：文本、说话人 embedding（可选）
输出：语音code（包括 token id 和 hiddens 两种形式，分别用作下一步 decoder和 DVAE 的输入）
过程图上已经很清楚了。这一步很有意思，吃进去 text token，输出 audio token。注意这里的LM head 是 head_code。

<img width="631" alt="Screenshot 2024-06-13 at 23 08 26" src="https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/1d4896ef-ebc2-4910-a770-de0cfb202118">

### Refine_text和Infer_code 的差异
主要在 LM head。对比如下：
```
(Pdb) self.head_text
ParametrizedLinear(
  in_features=768, out_features=21178, bias=False // 21178是 text token 数量，包括特殊 token，对应的是 BertTokenizer
  (parametrizations): ModuleDict(
    (weight): ParametrizationList(
      (0): _WeightNorm()
    )
  )
)
(Pdb) self.head_code
ModuleList(
  (0-3): 4 x ParametrizedLinear(
    in_features=768, out_features=626, bias=False // 626 是 audio token 数量， 这个 Tokenizer 是 GVQ，推理中不可见
    (parametrizations): ModuleDict(
      (weight): ParametrizationList(
        (0): _WeightNorm()
      )
    )
  )
)
```
### 从语音 code 到音频
没细看，不多赘述。

<img width="860" alt="Screenshot 2024-06-13 at 23 41 48" src="https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/99d83a7f-05fa-45b4-8fad-a8b9fef00b28">

## 对训练的一点推测
查看 infer 代码可知，在推理过程中，下图 get_emb()的text_mask是全 1。那么 emb_code (audio token的 embedding 实际没有被使用过)。但这块一定是有用的。那猜一手是训练中用到的。具体说说我很不完整的瞎猜：

训练任务应该有点像 LLM 训练中的text completion 任务。给一组 token，mask 掉其中几个，让模型去预测。

从项目 readme 中的“reveals capability of GVQ as audio tokenizer for LLM modeling”推测，作者用了Group Vector Quantization（GVQ）把语音数据变成 audio token，从而利用了 LLM 的强大语言理解和生成能力去生成语音。 这个 audio tokenizer 在推理中是不可见的。训练时，mask为 1 对应的 token（文本token)是不需要预测的，mask 为0 的语音 token 是预测的对象。通过训练Llama model 在给定文本上下文的情况下预测空缺部分的 audio token，让 Llama model 连通了文本和语音。
<img width="813" alt="Screenshot 2024-06-13 at 22 42 57" src="https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/e0c7c013-d5d5-4d23-a416-f8d6e196d839">

从这里瞎猜一下训练的方式。。。

移植Overview
![ChatTTS inference](https://github.com/ZillaRU/ChatTTS-TPU/assets/25343084/8cb2e12c-0c63-4a44-a60e-be9ba29f64bb)

希望大家一起讨论～
