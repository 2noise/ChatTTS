import wave
from io import BytesIO
import numpy as np
from .np import float_to_int16
from .av import wav2


def _pcm_to_wav_buffer(wav: np.ndarray, sample_rate: int = 24000) -> BytesIO:
    """
    Convert PCM audio data to a WAV format byte stream (internal utility function).

    :param wav: PCM data, NumPy array, typically in float32 format.
    :param sample_rate: Sample rate (in Hz), defaults to 24000.
    :return: WAV format byte stream, stored in a BytesIO object.
    """
    # Create an in-memory byte stream buffer
    buf = BytesIO()

    # Open a WAV file stream in write mode
    with wave.open(buf, "wb") as wf:
        # Set number of channels to 1 (mono)
        wf.setnchannels(1)
        # Set sample width to 2 bytes (16-bit)
        wf.setsampwidth(2)
        # Set sample rate
        wf.setframerate(sample_rate)
        # Convert PCM to 16-bit integer and write
        wf.writeframes(float_to_int16(wav))

    # Reset buffer pointer to the beginning
    buf.seek(0, 0)
    return buf


def pcm_arr_to_mp3_view(wav: np.ndarray, sample_rate: int = 24000) -> memoryview:
    """
    Convert PCM audio data to MP3 format.

    :param wav: PCM data, NumPy array, typically in float32 format.
    :param sample_rate: Sample rate (in Hz), defaults to 24000.
    :return: MP3 format byte data, returned as a memoryview.
    """
    # Get WAV format byte stream
    buf = _pcm_to_wav_buffer(wav, sample_rate)

    # Create output buffer
    buf2 = BytesIO()
    # Convert WAV data to MP3
    wav2(buf, buf2, "mp3")
    # Return MP3 data
    return buf2.getbuffer()


def pcm_arr_to_ogg_view(wav: np.ndarray, sample_rate: int = 24000) -> memoryview:
    """
    Convert PCM audio data to OGG format (using Vorbis encoding).

    :param wav: PCM data, NumPy array, typically in float32 format.
    :param sample_rate: Sample rate (in Hz), defaults to 24000.
    :return: OGG format byte data, returned as a memoryview.
    """
    # Get WAV format byte stream
    buf = _pcm_to_wav_buffer(wav, sample_rate)

    # Create output buffer
    buf2 = BytesIO()
    # Convert WAV data to OGG
    wav2(buf, buf2, "ogg")
    # Return OGG data
    return buf2.getbuffer()


def pcm_arr_to_wav_view(
    wav: np.ndarray, sample_rate: int = 24000, include_header: bool = True
) -> memoryview:
    """
    Convert PCM audio data to WAV format, with an option to include header.

    :param wav: PCM data, NumPy array, typically in float32 format.
    :param sample_rate: Sample rate (in Hz), defaults to 24000.
    :param include_header: Whether to include WAV header, defaults to True.
    :return: WAV format or raw PCM byte data, returned as a memoryview.
    """
    if include_header:
        # Get complete WAV byte stream
        buf = _pcm_to_wav_buffer(wav, sample_rate)
        return buf.getbuffer()
    else:
        # Return only converted 16-bit PCM data
        pcm_data = float_to_int16(wav)
        return memoryview(pcm_data.tobytes())
