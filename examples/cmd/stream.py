import io
import threading
import time
import random

import pyaudio  # please install it manually
import numpy as np
import ChatTTS

from tools.audio import batch_unsafe_float_to_int16


# 流式声音处理器
class AudioStreamer:
    def __init__(self):
        self.bio = io.BytesIO()
        self.lock = threading.Lock()
        self.seek_index = 0

    # 流式写入
    def write(self, waveform):
        with self.lock:
            #             waveform=(new_wave*32767).astype(np.int16)
            #             waveform=unsafe_float_to_int16(new_wave)
            # 将整数列表转换为字节字符串
            write_binary = waveform.astype("<i2").tobytes()
            self.bio.write(write_binary)

    # 流式读取
    def read(self):
        with self.lock:
            self.bio.seek(self.seek_index)
            read_binary = self.bio.read()
            self.seek_index += len(read_binary)
        return read_binary


# ChatTTS流式处理
class ChatStreamer:
    def __init__(self, waittime_topause=50, base_block_size=8000):
        self.streamer = AudioStreamer()
        self.accum_streamwavs = []
        self.waittime_topause = waittime_topause
        self.base_block_size = base_block_size

    def write(self, chatstream):
        # 已推理batch数据保存
        def accum(accum_wavs, stream_wav):
            n_texts = len(stream_wav)
            if accum_wavs is None:
                accum_wavs = [[i] for i in stream_wav]
            else:
                for i_text in range(n_texts):
                    if stream_wav[i_text] is not None:
                        accum_wavs[i_text].append(stream_wav[i_text])
            return accum_wavs

        # stream状态更新。数据量不足的stream，先存一段时间，直到拿到足够数据，监控小块数据情况
        def update_stream(history_stream_wav, new_stream_wav, thre):
            result_stream = []
            randn = -1
            if history_stream_wav is not None:
                randn = random.random()
                if randn > 0.1:
                    print("update_stream")
                n_texts = len(new_stream_wav)
                for i in range(n_texts):
                    if new_stream_wav[i] is not None:
                        result_stream.append(
                            np.concatenate(
                                [history_stream_wav[i], new_stream_wav[i]], axis=1
                            )
                        )
                    else:
                        result_stream.append(history_stream_wav[i])
            else:
                result_stream = new_stream_wav

            is_keep_next = (
                sum([i.shape[1] for i in result_stream if i is not None]) < thre
            )
            if randn > 0.1:
                print(
                    "result_stream:",
                    is_keep_next,
                    [i.shape if i is not None else None for i in result_stream],
                )
            return result_stream, is_keep_next

        self.finish = False
        curr_sentence_index = 0
        base_block_size = self.base_block_size
        history_stream_wav = None
        article_streamwavs = None
        for stream_wav in chatstream:
            n_texts = len(stream_wav)
            n_valid_texts = len(list(filter(lambda x: x is not None, stream_wav)))
            if n_valid_texts == 0:
                continue
            else:
                block_thre = n_valid_texts * base_block_size
                stream_wav, is_keep_next = update_stream(
                    history_stream_wav, stream_wav, block_thre
                )
                # 数据量不足，先保存状态
                if is_keep_next:
                    history_stream_wav = stream_wav
                    continue
                # 数据量足够，执行写入操作
                else:
                    history_stream_wav = None
                    stream_wav = batch_unsafe_float_to_int16(stream_wav)
                    article_streamwavs = accum(article_streamwavs, stream_wav)
                    # 写入当前句子
                    if stream_wav[curr_sentence_index] is not None:
                        self.streamer.write(stream_wav[curr_sentence_index][0])
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index < n_texts - 1:
                        curr_sentence_index += 1
                        print("add next sentence")
                        finish_stream_wavs = np.concatenate(
                            article_streamwavs[curr_sentence_index], axis=1
                        )
                        self.streamer.write(finish_stream_wavs[0])
                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
            # 有一定概率遇到奇怪bug（一定概率遇到256维异常输出，正常是1w+维），输出全是噪声，写的快遇到的概率更高？
            time.sleep(0.02)
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x: x is not None, stream_wav))) > 0:
                stream_wav = batch_unsafe_float_to_int16(stream_wav)
                if stream_wav[curr_sentence_index] is not None:
                    self.streamer.write(stream_wav[curr_sentence_index][0])
                    article_streamwavs = accum(article_streamwavs, stream_wav)
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = np.concatenate(article_streamwavs[i_text], axis=1)
            self.streamer.write(finish_stream_wavs[0])

        self.accum_streamwavs.append(article_streamwavs)
        self.finish = True

    def play(self, waittime_tostart=5, auto_end=False):
        # 初始化PyAudio对象
        p = pyaudio.PyAudio()

        # 设置音频流参数
        FORMAT = pyaudio.paInt16  # 16位深度
        CHANNELS = 1  # 单声道
        RATE = 24000  # 采样率
        CHUNK = 1024  # 每块音频数据大小

        # 打开输出流（扬声器）
        stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

        print("开始流式音频播放...")
        import time

        time.sleep(waittime_tostart)

        wait_time = 0
        while (self.streamer.bio.tell() > self.streamer.seek_index) | (
            wait_time < self.waittime_topause
        ):

            if self.streamer.bio.tell() > self.streamer.seek_index:
                read_data = self.streamer.read()
                stream_out.write(read_data)
                wait_time = 0
            # 如果不设置自动结束，就等待一段时间，如果一直没有新写入，就自动结束。如果设置了自动结束，就在写操作结束时结束播放
            else:
                if auto_end & self.finish:
                    print("写操作完成，自动结束。")
                    break
                else:
                    time.sleep(self.waittime_topause / 10)
                    wait_time += self.waittime_topause / 10

        print("完成流式音频播放...")
        stream_out.stop_stream()
        stream_out.close()

    # 获取完整历史播放数据
    def get_complete_speech(self):
        complete_waveform = np.concatenate(
            sum([sum(i_infer, []) for i_infer in self.accum_streamwavs], []), axis=1
        )
        return complete_waveform

    # 开始音频写入。可支持多次音频写入
    def start_writing(self, streamchat):
        self.writer = threading.Thread(target=self.write, args=(streamchat,))
        self.writer.start()

    # 开始音频播放
    def start_playing(self, waittime_tostart=5):
        self.player = threading.Thread(target=self.play, args=(waittime_tostart,))
        self.player.start()

    # writer和player完成join，需复杂操作可自行调用self.writer.join()或self.player.join()实现
    def join(self):
        self.writer.join()
        self.player.join()

    # 一次完整的音频写入+播放
    def run(self, streamchat, waittime_tostart=5):
        self.writer = threading.Thread(target=self.write, args=(streamchat,))
        self.player = threading.Thread(target=self.play, args=(waittime_tostart, True))
        self.writer.start()
        self.player.start()
        self.writer.join()
        self.player.join()


if __name__ == "__main__":

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
            "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。它们共同协作，让AI不仅仅是理论上的智能，而是能够在现实世界中发挥作用的智能。",
            "你太聪明啦。",
            "举个例子，大模型可能可以写代码，但它不能独立完成一个完整的软件开发项目。这时候，AI Agent就根据大模型的智能，结合记忆和规划，使用合适的工具，一步步实现从需求分析到产品上线。",
            "牛的牛的",
        ],
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )

    # 分别开启一个写线程和读线程，进行流式播放
    streamer = ChatStreamer()

    # 一次性生成
    streamer.run(streamchat)

    # 复杂使用示例：在同一个play中进行多次流式写入
    streamchat1 = chat.infer(
        [
            "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。",
            "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。它们共同协作，让AI不仅仅是理论上的智能，而是能够在现实世界中发挥作用的智能。",
            "你太聪明啦。",
            "举个例子，大模型可能可以写代码，但它不能独立完成一个完整的软件开发项目。这时候，AI Agent就根据大模型的智能，结合记忆和规划，使用合适的工具，一步步实现从需求分析到产品上线。",
            "牛的牛的",
        ],
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )

    streamchat2 = chat.infer(
        [
            "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
            "注意此版本可能不是最新版，所有内容请以英文版为准。",
        ],
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )

    streamer.start_playing()

    streamer.start_writing(streamchat1)
    streamer.writer.join()
    print("finish streamchat1")
    streamer.start_writing(streamchat2)
    streamer.writer.join()
    print("finish streamchat2")
    streamer.player.join()
    print("finish play")
