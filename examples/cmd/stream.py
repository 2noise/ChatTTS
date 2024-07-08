
import time
import random

import numpy as np

from tools.audio import unsafe_float_to_int16

class BatchStreamer:
    @classmethod
    def _batch_unsafe_float_to_int16(cls,audios):
        valid_audios_info=[(index,audio.shape[0]) for index,audio in enumerate(audios) if audio is not None]
        allmax_size=max([i[1] for i in valid_audios_info])
        valid_audios=np.zeros((len(valid_audios_info),allmax_size))
        for valid_index,(index,maxsize) in enumerate(valid_audios_info):
            valid_audios[valid_index,:maxsize]=audios[index]
        
        valid_audios=unsafe_float_to_int16(valid_audios)
        result_audios=[None]*len(audios)
        for i,(valid_index,max_size) in enumerate(valid_audios_info):
            result_audios[valid_index]=valid_audios[i,:max_size]
        return result_audios
        
    # stream状态更新。数据量不足的stream，先存一段时间，直到拿到足够数据，监控小块数据情况
    @classmethod
    def _update_stream(cls,history_stream_wav,new_stream_wav,thre):
        result_stream=[]
        if history_stream_wav is not None:
            n_texts=len(new_stream_wav)
            for i in range(n_texts):
                if new_stream_wav[i] is not None:
                    result_stream.append(np.concatenate([history_stream_wav[i],new_stream_wav[i]]))
                else:
                    result_stream.append(history_stream_wav[i])
            is_keep_next=sum([i.shape[0] for i in result_stream if i is not None])<thre
            if random.random()>0.1:
                print("update_stream",is_keep_next,[i.shape if i is not None else None for i in result_stream])
        else:
            result_stream=new_stream_wav
            is_keep_next=sum([i.shape[0] for i in result_stream if i is not None])<thre

        return result_stream,is_keep_next
    
    # 已推理batch数据保存
    @classmethod
    def _accum(cls,accum_wavs,stream_wav):
        n_texts=len(stream_wav)
        if accum_wavs is None:
            accum_wavs=[[i] for i in stream_wav]
        else:
            for i_text in range(n_texts):
                if stream_wav[i_text] is not None:
                    accum_wavs[i_text].append(stream_wav[i_text])
        return accum_wavs
    
    # batch stream数据格式转化
    @classmethod
    def batch_stream_formatted(cls,stream_wav,output_format='PCM16_byte'):
        if output_format in ('PCM16_byte','PCM16'):
            format_data=cls._batch_unsafe_float_to_int16(stream_wav)
        else:
            format_data=stream_wav
        return format_data
    
    # 数据格式转化
    @classmethod
    def formatted(cls,data,output_format='PCM16_byte'):
        if output_format=='PCM16_byte':
            format_data=data.astype("<i2").tobytes()
        else:
            format_data=data
        return format_data
    
    # 数据生成
    @classmethod
    def generate(cls,chatstream,output_format=None,base_block_size=8000):
        assert output_format in ('PCM16_byte','PCM16',None)
        curr_sentence_index=0
        history_stream_wav=None
        article_streamwavs=None
        for stream_wav in chatstream:
            n_texts=len(stream_wav)
            n_valid_texts=len(list(filter(lambda x:x is not None,stream_wav)))
            if n_valid_texts==0:
                continue
            else:
                block_thre=n_valid_texts*base_block_size
                stream_wav,is_keep_next=cls._update_stream(history_stream_wav,stream_wav,block_thre)
                # 数据量不足，先保存状态
                if is_keep_next:
                    history_stream_wav=stream_wav
                    continue
                # 数据量足够，执行写入操作
                else:
                    history_stream_wav=None
                    stream_wav=cls.batch_stream_formatted(stream_wav,output_format)
                    article_streamwavs=cls._accum(article_streamwavs,stream_wav)
                    # 写入当前句子
                    if stream_wav[curr_sentence_index] is not None:
                        yield cls.formatted(stream_wav[curr_sentence_index],output_format)
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index<n_texts-1:
                        curr_sentence_index+=1
                        print('add next sentence')
                        finish_stream_wavs=np.concatenate(article_streamwavs[curr_sentence_index])
                        yield cls.formatted(finish_stream_wavs,output_format)
                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x:x is not None,stream_wav)))>0:
                stream_wav=cls.batch_stream_formatted(stream_wav,output_format)
                if stream_wav[curr_sentence_index] is not None:
                    yield cls.formatted(stream_wav[curr_sentence_index],output_format)
                    article_streamwavs=cls._accum(article_streamwavs,stream_wav)
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index+1,n_texts):
            finish_stream_wavs=np.concatenate(article_streamwavs[i_text])
            yield cls.formatted(finish_stream_wavs,output_format)



if __name__ == "__main__":
    import ChatTTS
    import pyaudio  # please install it manually

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

    streamchat = chat.infer(
        [
            "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。",
            "你太聪明啦。",
            "举个例子，大模型可能可以写代码，但它不能独立完成一个完整的软件开发项目。这时候，AI Agent就根据大模型的智能，结合记忆和规划，使用合适的工具，一步步实现从需求分析到产品上线。",
        ],
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        stream=True,
    )

    # 流式播放准备
    p = pyaudio.PyAudio()
    print(p.get_device_count())
    # 设置音频流参数
    FORMAT = pyaudio.paInt16  # 16位深度
    CHANNELS = 1  # 单声道
    RATE = 24000  # 采样率
    CHUNK = 1024  # 每块音频数据大小

    # 打开输出流（扬声器）
    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,)

    # 流式推理，以音频编码字节流格式输出，并流式播放
    for i in BatchStreamer.generate(streamchat,output_format='PCM16_byte'):
        stream_out.write(i)    
        

    stream_out.stop_stream()
    stream_out.close()