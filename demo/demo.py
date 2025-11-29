import ChatTTS
import torch
import torchaudio

# 设置后端为系统里唯一的 sox
torchaudio.set_audio_backend("sox")

chat = ChatTTS.Chat()
chat.load(compile=False)

texts = [
    """今晚的风比平时更温柔一些，像是在给整座城市按着舒缓的节奏。[breath] 我走在街边，耳机里放着轻轻的音乐，脚步也不自觉慢下来。就在这时，我忽然想起白天发生的一件小事，忍不住轻轻笑出声。[laugh][uv_break][uv_dur=1] 那种笑并不夸张，但足够把心情往上托一点点。回想起早上，我一边喝豆浆，一边看着路边的大叔苦恼地追报纸。纸一张张像小鸟一样飞起来，他却急得在原地跺脚。[uv_break] 我帮他捡起来时，他憨憨地笑了一声：“这风比我儿子还调皮。”他说完自己先笑得前仰后合。[laugh][lbreak] 我当时也没忍住，差点把豆浆喷出来。[laugh]午后的阳光透过树叶，像金色碎片一样落在长椅上。[breath] 我坐在那里发呆，忽然听到旁边的小孩在练习吹口哨，但吹得乱七八糟。
    [uv_break] 他越吹越认真，我越听越觉得好笑。[laugh] 到最后，我俩对视了一眼，他居然也笑了。[laugh][uv_dur=2]
傍晚的路口，人来人往，空气里混着烤串的味道，还有一点潮意。我看着街灯亮起，一盏、两盏，像有人在悄悄点亮冬天的按钮。[breath][uv_break] 我不知道为什么，心里突然松了一口气，像卸下了一整天的重量.
走到家门口时，我心里忽然冒出一个念头：日子其实并不需要太多特别的东西，它们是由这些微妙的瞬间、意外的小笑声、以及偶尔的轻快节奏拼合起来的。[uv_break] 想着想着，我又笑了一声。[laugh][uv_dur=1]"""
]
wavs = chat.infer(texts)

for i, wav in enumerate(wavs):
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    torchaudio.save(f"output{i}.wav", wav_tensor, 24000)

print("done")
