import numpy as np
from numba import jit


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    This function will destroy audio, use only once.
    """
    am = np.abs(audio).max() * 32768
    am = 32767 * 32768 / am
    return np.multiply(audio, am).astype(np.int16)
