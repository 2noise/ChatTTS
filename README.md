# ChatTTS

## To be Finished

```python
import torch
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

texts = ["<YOUR TEXT HERE>",]

wavs = chat.infer(texts, use_decoder=True)

Audio(wavs[0], rate=24_000, autoplay=True)
```

## 
Disclaimer: For Academic Purposes Only

The information provided in this document is for academic purposes only. It is intended for educational and research use, and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this document, are for academic and research purposes only. The data have been obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

免责声明：仅供学术交流

本文件中的信息仅供学术交流使用。其目的是用于教育和研究，不得用于任何商业或法律目的。作者不保证信息的准确性、完整性或可靠性。本文件中使用的信息和数据，仅用于学术研究目的。这些数据来自公开可用的来源，作者不对数据的所有权或版权提出任何主张。
