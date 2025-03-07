import wave
from io import BytesIO
import numpy as np
from .np import float_to_int16
from .av import wav2

def _pcm_to_wav_buffer(wav: np.ndarray, sample_rate: int = 24000) -> BytesIO:
    """
    将 PCM 音频数据转换为 WAV 格式的字节流（内部工具函数）。
    Convert PCM audio data to a WAV format byte stream (internal utility function).

    :param wav: PCM 数据，NumPy 数组，通常为浮点型 (float32)。
                PCM data, NumPy array, typically in float32 format.
    :param sample_rate: 采样率（单位：Hz），默认值为 24000。
                        Sample rate (in Hz), defaults to 24000.
    :return: WAV 格式的字节流，存储在 BytesIO 对象中。
             WAV format byte stream, stored in a BytesIO object.
    """
    # 创建内存中的字节流缓冲区 / Create an in-memory byte stream buffer
    buf = BytesIO()
    
    # 以写入模式打开 WAV 文件流 / Open a WAV file stream in write mode
    with wave.open(buf, "wb") as wf:
        # 设置声道数为 1（单声道） / Set number of channels to 1 (mono)
        wf.setnchannels(1)
        # 设置采样宽度为 2 字节（16-bit） / Set sample width to 2 bytes (16-bit)
        wf.setsampwidth(2)
        # 设置采样率 / Set sample rate
        wf.setframerate(sample_rate)
        # 将 PCM 数据转换为 16-bit 整数并写入 / Convert PCM to 16-bit integer and write
        wf.writeframes(float_to_int16(wav))
    
    # 将缓冲区指针重置到开头 / Reset buffer pointer to the beginning
    buf.seek(0, 0)
    return buf

def pcm_arr_to_mp3_view(wav: np.ndarray, sample_rate: int = 24000) -> memoryview:
    """
    将 PCM 音频数据转换为 MP3 格式。
    Convert PCM audio data to MP3 format.

    :param wav: PCM 数据，NumPy 数组，通常为浮点型 (float32)。
                PCM data, NumPy array, typically in float32 format.
    :param sample_rate: 采样率（单位：Hz），默认值为 24000。
                        Sample rate (in Hz), defaults to 24000.
    :return: MP3 格式的字节数据，以 memoryview 形式返回。
             MP3 format byte data, returned as a memoryview.
    """
    # 获取 WAV 格式的字节流 / Get WAV format byte stream
    buf = _pcm_to_wav_buffer(wav, sample_rate)
    
    # 创建输出缓冲区 / Create output buffer
    buf2 = BytesIO()
    # 将 WAV 数据转换为 MP3 / Convert WAV data to MP3
    wav2(buf, buf2, "mp3")
    # 返回 MP3 数据 / Return MP3 data
    return buf2.getbuffer()

def pcm_arr_to_ogg_view(wav: np.ndarray, sample_rate: int = 24000) -> memoryview:
    """
    将 PCM 音频数据转换为 OGG 格式（使用 Vorbis 编码）。
    Convert PCM audio data to OGG format (using Vorbis encoding).

    :param wav: PCM 数据，NumPy 数组，通常为浮点型 (float32)。
                PCM data, NumPy array, typically in float32 format.
    :param sample_rate: 采样率（单位：Hz），默认值为 24000。
                        Sample rate (in Hz), defaults to 24000.
    :return: OGG 格式的字节数据，以 memoryview 形式返回。
             OGG format byte data, returned as a memoryview.
    """
    # 获取 WAV 格式的字节流 / Get WAV format byte stream
    buf = _pcm_to_wav_buffer(wav, sample_rate)
    
    # 创建输出缓冲区 / Create output buffer
    buf2 = BytesIO()
    # 将 WAV 数据转换为 OGG / Convert WAV data to OGG
    wav2(buf, buf2, "ogg")
    # 返回 OGG 数据 / Return OGG data
    return buf2.getbuffer()

def pcm_arr_to_wav_view(wav: np.ndarray, sample_rate: int = 24000, include_header: bool = True) -> memoryview:
    """
    将 PCM 音频数据转换为 WAV 格式，可选择是否包含头部。
    Convert PCM audio data to WAV format, with an option to include header.

    :param wav: PCM 数据，NumPy 数组，通常为浮点型 (float32)。
                PCM data, NumPy array, typically in float32 format.
    :param sample_rate: 采样率（单位：Hz），默认值为 24000。
                        Sample rate (in Hz), defaults to 24000.
    :param include_header: 是否包含 WAV 文件头，默认为 True。
                           Whether to include WAV header, defaults to True.
    :return: WAV 格式或原始 PCM 的字节数据，以 memoryview 形式返回。
             WAV format or raw PCM byte data, returned as a memoryview.
    """
    if include_header:
        # 获取完整的 WAV 字节流 / Get complete WAV byte stream
        buf = _pcm_to_wav_buffer(wav, sample_rate)
        return buf.getbuffer()
    else:
        # 仅返回转换后的 16-bit PCM 数据 / Return only converted 16-bit PCM data
        pcm_data = float_to_int16(wav)
        return memoryview(pcm_data.tobytes())