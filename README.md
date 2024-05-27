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