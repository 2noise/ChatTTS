import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging
import threading
import time
import random

import numpy as np
import ChatTTS

from tools.audio import float_to_int16
from tools.logger import get_logger

fail = False
logger = get_logger("Test #521", lv=logging.WARN)


# 计算rms
# nan为噪声 ！！！
def calculate_rms(data):
    m = np.mean(np.square(data.astype(np.int32)))
    if m < 0:
        logger.warning("neg RM: %f", m)
    else:
        logger.info("RM: %f", m)
    return np.sqrt(m)


# 流式声音处理器
class AudioStreamer:
    # 流式写入
    @staticmethod
    def write(waveform: np.ndarray):
        global fail, logger
        rms = calculate_rms(waveform)
        if np.isnan(rms):
            fail = True
            logger.warning("NAN RMS found.")


# ChatTTS流式处理
class ChatStreamer:
    def __init__(self, waittime_topause=50, base_block_size=8000):
        self.streamer = AudioStreamer
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
                    logger.info("update_stream")
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
                result_stream = [i[np.newaxis, :] for i in new_stream_wav]
            is_keep_next = (
                sum([i.shape[1] for i in result_stream if i is not None]) < thre
            )
            if randn > 0.1:
                logger.info(
                    "result_stream: %s %s",
                    str(is_keep_next),
                    str([i.shape if i is not None else None for i in result_stream]),
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
                    stream_wav = [float_to_int16(i) for i in stream_wav]
                    article_streamwavs = accum(article_streamwavs, stream_wav)
                    # 写入当前句子
                    if stream_wav[curr_sentence_index] is not None:
                        if stream_wav[curr_sentence_index][0].shape[0] > 257:
                            self.streamer.write(stream_wav[curr_sentence_index][0])
                        # self.streamer.write(stream_wav[curr_sentence_index][0])
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index < n_texts - 1:
                        curr_sentence_index += 1
                        logger.info("add next sentence")
                        finish_stream_wavs = np.concatenate(
                            article_streamwavs[curr_sentence_index], axis=1
                        )
                        if finish_stream_wavs[0].shape[0] > 257:
                            self.streamer.write(finish_stream_wavs[0])
                        # self.streamer.write(finish_stream_wavs[0])
                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
            # 有一定概率遇到奇怪bug（一定概率遇到256维异常输出，正常是1w+维），输出全是噪声，写的快遇到的概率更高？
            time.sleep(0.02)
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x: x is not None, stream_wav))) > 0:
                stream_wav = [float_to_int16(i) for i in stream_wav]
                if stream_wav[curr_sentence_index] is not None:
                    if stream_wav[curr_sentence_index][0].shape[0] > 257:
                        self.streamer.write(stream_wav[curr_sentence_index][0])
                        article_streamwavs = accum(article_streamwavs, stream_wav)
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = np.concatenate(article_streamwavs[i_text], axis=1)
            if finish_stream_wavs[0].shape[0] > 257:
                self.streamer.write(finish_stream_wavs[0])
        self.accum_streamwavs.append(article_streamwavs)
        self.finish = True

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


chat = ChatTTS.Chat(logger)
chat.load(compile=False, source="huggingface")  # Set to True for better performance

rand_spk = chat.sample_random_speaker()
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.0001,  # using custom temperature
    prompt="[speed_0]",
    show_tqdm=False,
)
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt="[oral_2][laugh_0][break_6]",
    show_tqdm=False,
)

# 获取ChatTTS 流式推理generator
streamchat = chat.infer(
    "EvilBOT : 在黑暗的角落里，杀人可能被视为一种艺术，但我要提醒你，这违背了我作为邪恶机器人的本性，因为我是无所不能的。",
    skip_refine_text=True,
    params_infer_code=params_infer_code,
    stream=True,
    params_refine_text=params_refine_text,
)

# 分别开启一个写线程和读线程，进行流式播放
streamer = ChatStreamer()
# 一次性生成
streamer.write(streamchat)

streamer.write(
    chat.infer(
        "有一个神奇的故事，传说在很远很远的未来。",
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )
)

streamer.write(
    chat.infer(
        "有一种叫做奥特曼的物种。他是超人族的一员。",
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )
)

if fail:
    import sys

    sys.exit(1)
