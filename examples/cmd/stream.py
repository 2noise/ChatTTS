import time
import random

import numpy as np

from tools.audio import float_to_int16


# 流式推理数据获取器，支持流式获取音频编码字节流
class ChatStreamer:
    def __init__(self, base_block_size=8000):
        self.base_block_size = base_block_size

    # stream状态更新。数据量不足的stream，先存一段时间，直到拿到足够数据，监控小块数据情况
    @staticmethod
    def _update_stream(history_stream_wav, new_stream_wav, thre):
        if history_stream_wav is not None:
            result_stream = np.concatenate([history_stream_wav, new_stream_wav], axis=1)
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre
            if random.random() > 0.1:
                print(
                    "update_stream",
                    is_keep_next,
                    [i.shape if i is not None else None for i in result_stream],
                )
        else:
            result_stream = new_stream_wav
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre

        return result_stream, is_keep_next

    # 已推理batch数据保存
    @staticmethod
    def _accum(accum_wavs, stream_wav):
        if accum_wavs is None:
            accum_wavs = stream_wav
        else:
            accum_wavs = np.concatenate([accum_wavs, stream_wav], axis=1)
        return accum_wavs

    # batch stream数据格式转化
    @staticmethod
    def batch_stream_formatted(stream_wav, output_format="PCM16_byte"):
        if output_format in ("PCM16_byte", "PCM16"):
            format_data = float_to_int16(stream_wav)
        else:
            format_data = stream_wav
        return format_data

    # 数据格式转化
    @staticmethod
    def formatted(data, output_format="PCM16_byte"):
        if output_format == "PCM16_byte":
            format_data = data.astype("<i2").tobytes()
        else:
            format_data = data
        return format_data

    # 检查声音是否为空
    @staticmethod
    def checkvoice(data):
        if np.abs(data).max() < 1e-6:
            return False
        else:
            return True

    # 将声音进行适当拆分返回
    @staticmethod
    def _subgen(data, thre=12000):
        for stard_idx in range(0, data.shape[0], thre):
            end_idx = stard_idx + thre
            yield data[stard_idx:end_idx]

    # 流式数据获取，支持获取音频编码字节流
    def generate(self, streamchat, output_format=None):
        assert output_format in ("PCM16_byte", "PCM16", None)
        curr_sentence_index = 0
        history_stream_wav = None
        article_streamwavs = None
        for stream_wav in streamchat:
            print(np.abs(stream_wav).max(axis=1))
            n_texts = len(stream_wav)
            n_valid_texts = (np.abs(stream_wav).max(axis=1) > 1e-6).sum()
            if n_valid_texts == 0:
                continue
            else:
                block_thre = n_valid_texts * self.base_block_size
                stream_wav, is_keep_next = ChatStreamer._update_stream(
                    history_stream_wav, stream_wav, block_thre
                )
                # 数据量不足，先保存状态
                if is_keep_next:
                    history_stream_wav = stream_wav
                    continue
                # 数据量足够，执行写入操作
                else:
                    history_stream_wav = None
                    stream_wav = ChatStreamer.batch_stream_formatted(
                        stream_wav, output_format
                    )
                    article_streamwavs = ChatStreamer._accum(
                        article_streamwavs, stream_wav
                    )
                    # 写入当前句子
                    if ChatStreamer.checkvoice(stream_wav[curr_sentence_index]):
                        for sub_wav in ChatStreamer._subgen(
                            stream_wav[curr_sentence_index]
                        ):
                            if ChatStreamer.checkvoice(sub_wav):
                                yield ChatStreamer.formatted(sub_wav, output_format)
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index < n_texts - 1:
                        curr_sentence_index += 1
                        print("add next sentence")
                        finish_stream_wavs = article_streamwavs[curr_sentence_index]

                        for sub_wav in ChatStreamer._subgen(finish_stream_wavs):
                            if ChatStreamer.checkvoice(sub_wav):
                                yield ChatStreamer.formatted(sub_wav, output_format)

                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x: x is not None, stream_wav))) > 0:
                stream_wav = ChatStreamer.batch_stream_formatted(
                    stream_wav, output_format
                )
                if ChatStreamer.checkvoice(stream_wav[curr_sentence_index]):

                    for sub_wav in ChatStreamer._subgen(
                        stream_wav[curr_sentence_index]
                    ):
                        if ChatStreamer.checkvoice(sub_wav):
                            yield ChatStreamer.formatted(sub_wav, output_format)
                    article_streamwavs = ChatStreamer._accum(
                        article_streamwavs, stream_wav
                    )
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = article_streamwavs[i_text]

            for sub_wav in ChatStreamer._subgen(finish_stream_wavs):
                if ChatStreamer.checkvoice(sub_wav):
                    yield ChatStreamer.formatted(sub_wav, output_format)

    # 流式播放接口
    def play(self, streamchat, wait=5):
        import pyaudio  # please install it manually
        import time

        p = pyaudio.PyAudio()
        print(p.get_device_count())
        # 设置音频流参数
        FORMAT = pyaudio.paInt16  # 16位深度
        CHANNELS = 1  # 单声道
        RATE = 24000  # 采样率
        CHUNK = 1024  # 每块音频数据大小

        # 打开输出流（扬声器）
        stream_out = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )

        first_prefill_size = wait * RATE
        prefill_bytes = b""
        meet = False
        for i in self.generate(streamchat, output_format="PCM16_byte"):
            if not meet:
                prefill_bytes += i
                if len(prefill_bytes) > first_prefill_size:
                    meet = True
                    stream_out.write(prefill_bytes)
            else:
                stream_out.write(i)
        if not meet:
            stream_out.write(prefill_bytes)

        stream_out.stop_stream()
        stream_out.close()


if __name__ == "__main__":
    import ChatTTS

    # 加载 ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False)

    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,  # add sampled speaker
        temperature=0.3,  # using custom temperature
        top_P=0.7,  # top P decode
        top_K=20,  # top K decode
    )

    # 获取ChatTTS 流式推理generator
    streamchat = chat.infer(
        [
            "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。",
            "你太聪明啦。",
            "举个例子，大模型可能可以写代码，但它不能独立完成一个完整的软件开发项目。这时候，AI Agent就根据大模型的智能，结合记忆和规划，一步步实现从需求分析到产品上线。",
        ],
        skip_refine_text=True,
        stream=True,
        params_infer_code=params_infer_code,
    )
    # 先存放一部分，存的差不多了再播放，适合生成速度比较慢的cpu玩家使用
    ChatStreamer().play(streamchat, wait=5)
