import wave
from io import BytesIO

import numpy as np

from .np import unsafe_float_to_int16
from .av import wav2


def wav_arr_to_mp3_view(wav: np.ndarray):
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(unsafe_float_to_int16(wav))
    buf.seek(0, 0)
    buf2 = BytesIO()
    wav2(buf, buf2, "mp3")
    buf.seek(0, 0)
    return buf2.getbuffer()
