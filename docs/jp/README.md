# ChatTTS
> [!NOTE]
> 以下の内容は最新情報ではない可能性がありますのでご了承ください。全ての内容は英語版に基準することになります。

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)

[**English**](../../README.md) | [**简体中文**](../cn/README.md) | **日本語** | [**Русский**](../ru/README.md) | [**Español**](../es/README.md) | [**Français**](../fr/README.md) | [**한국어**](../kr/README.md)

ChatTTSは、LLMアシスタントなどの対話シナリオ用に特別に設計されたテキストから音声へのモデルです。英語と中国語の両方をサポートしています。私たちのモデルは、中国語と英語で構成される100,000時間以上でトレーニングされています。**[HuggingFace](https://huggingface.co/2Noise/ChatTTS)**でオープンソース化されているバージョンは、40,000時間の事前トレーニングモデルで、SFTは行われていません。

モデルやロードマップについての正式なお問い合わせは、**open-source@2noise.com**までご連絡ください。QQグループ：808364215に参加してディスカッションすることもできます。GitHubでの問題提起も歓迎します。

---
## ハイライト
1. **会話型TTS**: ChatTTSは対話ベースのタスクに最適化されており、自然で表現豊かな音声合成を実現します。複数の話者をサポートし、対話型の会話を容易にします。
2. **細かい制御**: このモデルは、笑い、一時停止、間投詞などの細かい韻律特徴を予測および制御することができます。
3. **より良い韻律**: ChatTTSは、韻律の面でほとんどのオープンソースTTSモデルを超えています。さらなる研究と開発をサポートするために、事前トレーニングされたモデルを提供しています。

モデルの詳細な説明については、**[Bilibiliのビデオ](https://www.bilibili.com/video/BV1zn4y1o7iV)**を参照してください。

---

## 免責事項

このリポジトリは学術目的のみのためです。教育および研究用途にのみ使用され、商業的または法的な目的には使用されません。著者は情報の正確性、完全性、または信頼性を保証しません。このリポジトリで使用される情報およびデータは、学術および研究目的のみのためのものです。データは公開されているソースから取得され、著者はデータに対する所有権または著作権を主張しません。

ChatTTSは強力なテキストから音声へのシステムです。しかし、この技術を責任を持って、倫理的に利用することが非常に重要です。ChatTTSの使用を制限するために、40,000時間のモデルのトレーニング中に少量の高周波ノイズを追加し、MP3形式を使用して音質を可能な限り圧縮しました。これは、悪意のあるアクターが潜在的に犯罪目的で使用することを防ぐためです。同時に、私たちは内部的に検出モデルをトレーニングしており、将来的にオープンソース化する予定です。

---
## 使用方法

<h4>基本的な使用方法</h4>

```python
import ChatTTS
from IPython.display import Audio
import torch

chat = ChatTTS.Chat()
chat.load(compile=False) # より良いパフォーマンスのためにTrueに設定

texts = ["ここにテキストを入力してください",]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

<h4>高度な使用方法</h4>

```python
###################################
# ガウス分布から話者をサンプリングします。

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = {
  'spk_emb': rand_spk, # サンプリングされた話者を追加
  'temperature': .3, # カスタム温度を使用
  'top_P': 0.7, # トップPデコード
  'top_K': 20, # トップKデコード
}

###################################
# 文レベルの手動制御のために。

# 特別なトークンを生成するためにテキストにoral_(0-9)、laugh_(0-2)、break_(0-7)を使用します。
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wav = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

###################################
# 単語レベルの手動制御のために。
text = 'あなたの好きな英語の食べ物は何ですか？[uv_break][laugh][lbreak]'
wav = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>例：自己紹介</h4></summary>

```python
inputs_jp = """
ChatTTSは、対話アプリケーション用に設計されたテキストから音声へのモデルです。
[uv_break]混合言語入力をサポートし[uv_break]、韻律要素[laugh]の正確な制御を提供します
[uv_break]笑い[laugh]、[uv_break]一時停止、[uv_break]およびイントネーション。[uv_break]自然で表現豊かな音声を提供します
[uv_break]したがって、自己責任でプロジェクトを責任を持って使用してください。[uv_break]
""".replace('\n', '') # 英語はまだ実験的です。

params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_4]'
} 
audio_array_jp = chat.infer(inputs_jp, params_refine_text=params_refine_text)
torchaudio.save("output3.wav", torch.from_numpy(audio_array_jp[0]), 24000)
```
[男性話者](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

[女性話者](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)
</details>

---
## ロードマップ
- [x] 40k時間のベースモデルとspk_statsファイルをオープンソース化
- [ ] VQエンコーダーとLoraトレーニングコードをオープンソース化
- [ ] テキストをリファインせずにストリーミングオーディオ生成*
- [ ] 複数の感情制御を備えた40k時間バージョンをオープンソース化
- [ ] ChatTTS.cppもしかしたら？（PRや新しいリポジトリが歓迎されます。）

----
## FAQ

##### VRAMはどれくらい必要ですか？推論速度はどうですか？
30秒のオーディオクリップには、少なくとも4GBのGPUメモリが必要です。4090 GPUの場合、約7つの意味トークンに対応するオーディオを1秒あたり生成できます。リアルタイムファクター（RTF）は約0.3です。

##### モデルの安定性が十分でなく、複数の話者や音質が悪いという問題があります。

これは、自己回帰モデル（barkおよびvalleの場合）で一般的に発生する問題です。一般的に避けるのは難しいです。複数のサンプルを試して、適切な結果を見つけることができます。

##### 笑い以外に何か制御できますか？他の感情を制御できますか？

現在リリースされているモデルでは、トークンレベルの制御ユニットは[laugh]、[uv_break]、および[lbreak]のみです。将来のバージョンでは、追加の感情制御機能を備えたモデルをオープンソース化する可能性があります。

---
## 謝辞
- [bark](https://github.com/suno-ai/bark)、[XTTSv2](https://github.com/coqui-ai/TTS)、および[valle](https://arxiv.org/abs/2301.02111)は、自己回帰型システムによる顕著なTTS結果を示しました。
- [fish-speech](https://github.com/fishaudio/fish-speech)は、LLMモデリングのためのオーディオトークナイザーとしてのGVQの能力を明らかにしました。
- 事前トレーニングされたボコーダーとして使用される[vocos](https://github.com/gemelo-ai/vocos)。

---
## 特別感謝
- 初期のアルゴリズム実験をサポートしてくれた[wlu-audio lab](https://audio.westlake.edu.cn/)。
