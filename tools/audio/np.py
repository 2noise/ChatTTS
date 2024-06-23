import numpy as np
from numba import jit

@jit
def unsafe_float_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    This function will destroy audio, use only once.
    """
    am = np.abs(audio).max() * 32768
    am = 32767 * 32768 / am
    np.multiply(audio, am, audio)
    audio16 = audio.astype(np.int16)
    return audio16
