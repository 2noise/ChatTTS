import sys
import torch
import numpy as np
import ChatTTS
from IPython.display import Audio



import io
import threading
import time
import pyaudio
import struct

# 流式声音处理器
class AudioStreamer:
    def __init__(self):
        self.bio=io.BytesIO()
        self.lock = threading.Lock()
        self.seek_index=0
        
    # 流式写入
    def write(self, new_wave):
        with self.lock:
            waveform=(new_wave*32767).astype(np.int16)
            # 将整数列表转换为字节字符串
            write_binary = struct.pack('<' + 'h' * len(waveform), *waveform)
            self.bio.write(write_binary)
            
    # 流式读取
    def read(self):
        with self.lock:
            self.bio.seek(self.seek_index)
            read_binary = self.bio.read()
            self.seek_index+=len(read_binary)
        return read_binary
    
# ChatTTS流式处理
class ChatTTSstreamer:
    def __init__(self,waittime_topause=20):
        self.streamer=AudioStreamer()
        self.accum_streamwavs=None
        self.waittime_topause=waittime_topause
    def write(self,chatstream):
        
        curr_article_index=0
        
        index=0
        for stream_wav in chatstream:
            # if index%100==0:
            #     print([i.shape if i is not None else None for i in stream_wav])
            n_texts=len(stream_wav)
            
            if self.accum_streamwavs is None:
                self.accum_streamwavs=stream_wav
            else:
                for i_text in range(n_texts):
                    if stream_wav[i_text] is not None:
                        self.accum_streamwavs[i_text]=np.concatenate([
                            self.accum_streamwavs[i_text].reshape(1,-1)
                            ,stream_wav[i_text].reshape(1,-1)
                        ],axis=1)
            
            if stream_wav[curr_article_index] is not None:
                self.streamer.write(stream_wav[curr_article_index][0])

            elif curr_article_index<n_texts-1:
                curr_article_index+=1
                print('add next sentence')
                self.streamer.write(self.accum_streamwavs[curr_article_index][0])
                
            else:
                break
            index+=1
        for i in range(curr_article_index+1,n_texts):
            self.streamer.write(self.accum_streamwavs[i][0])
            
    def play_article(self,waittime_tostart=30):
        # 初始化PyAudio对象
        p = pyaudio.PyAudio()

        # 设置音频流参数
        FORMAT = pyaudio.paInt16  # 16位深度
        CHANNELS = 1              # 单声道
        RATE = 24000              # 采样率
        CHUNK = 1024              # 每块音频数据大小

        # 打开输出流（扬声器）
        stream_out = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            output=True)

        print("开始流式音频播放...")
        import time
        time.sleep(waittime_tostart)

        wait_time=0
        while (self.streamer.bio.tell()>self.streamer.seek_index)|(wait_time<self.waittime_topause):
            if self.streamer.bio.tell()>self.streamer.seek_index:
                read_data=self.streamer.read()
                stream_out.write(read_data)
                wait_time=0
            else:
                time.sleep(self.waittime_topause/5)
                wait_time+=self.waittime_topause/5
                print('wait_time',wait_time)

        print("完成流式音频播放...")
        stream_out.stop_stream()
        stream_out.close()
    
if __name__ == "__main__":
    

    # 加载 ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False)


    rand_spk=chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )

    # 获取ChatTTS 流式推理generator
    streamchat = chat.infer(
        [
         '总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。',
         '总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。它们共同协作，让AI不仅仅是理论上的智能，而是能够在现实世界中发挥作用的智能。',
         '你太聪明啦。',
         '举个例子，大模型可能可以写代码，但它不能独立完成一个完整的软件开发项目。这时候，AI Agent就根据大模型的智能，结合记忆和规划，使用合适的工具，一步步实现从需求分析到产品上线。',
            '牛的牛的',
        ]   
            ,skip_refine_text=True
            ,params_infer_code=params_infer_code
            ,stream=True
        )
    
    # 分别开启一个写线程和读线程，进行流式播放
    streamer=ChatTTSstreamer()
    writer_thread = threading.Thread(target=streamer.write, args=(streamchat,))
    player_thread = threading.Thread(target=streamer.play_article, args=())

    writer_thread.start()
    player_thread.start()

    writer_thread.join()
    player_thread.join()


    # 获取完整声音
    complete_waveform=np.concatenate(streamer.accum_streamwavs,axis=1)
    Audio(complete_waveform[0], rate=24_000, autoplay=True)