import ChatTTS
import torch
import torchaudio


chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance
rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk,
    temperature = .3,
    top_P = 0.7,
    top_K = 20,
)
# 空 refine，避免干扰
params_refine_text = ChatTTS.Chat.RefineTextParams(prompt='')

text = 'What is [uv_break] your favorite english food? [laugh] [lbreak]'

wavs = chat.infer(
    text,
    skip_refine_text=True,
    params_infer_code=params_infer_code
)

for i, wav in enumerate(wavs):
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    torchaudio.save(f"output{i}.wav", wav_tensor, 24000)

print("done")