import ChatTTS as ChatTTS
import torch
import torchaudio
import soundfile as sf
chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance
rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)
texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
    )

# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
sf.write("output1.wav", wavs[1], 24000)