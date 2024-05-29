import numpy as np
import ChatTTS
import wave
from pathlib import Path

# 当前目录
current_path = Path(__file__).resolve().parent

# 下载模型到本地的models目录中，然后按下面目录指定模型路径
model_dir = current_path.joinpath("models").joinpath("2Noise").joinpath("ChatTTS").as_posix()

# 初始化模型
chat = ChatTTS.Chat()
chat.load_models(source="local", local_path=model_dir)

def infer_text_to_audio(text, *args, **kwargs):
    """
    text推理后保存到本地
    :param text: 需要推理的文本
    :param args:
    :param kwargs:
    :return:
    """

    # 推理文本到语音
    wavs = chat.infer(text, use_decoder=True)

    # 提取音频数据
    wavs_np = wavs[0]

    # 将音频数据缩放到[-1, 1]范围内，这是wav文件的标准范围
    audio_data = wavs_np / np.max(np.abs(wavs_np))

    # 将浮点数转为16位整数，这是wav文件常用格式
    audio_data_int16 = (audio_data * 32767).astype(np.int16)

    # 保存到本地，注意采样率为24000
    with wave.open("output.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data_int16)


if __name__ == '__main__':
    text = "你会掉下去[uv_break]造成严重外伤，大量内出血[uv_break]与多处复杂性骨折，也有机会在下方毒雾中[uv_break]受到电击或被分解。"
    # 执行后会在当前仓库根目录找到output.wav文件
    infer_text_to_audio(text)
