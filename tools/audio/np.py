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


@jit
def batch_unsafe_float_to_int16(audios: list[np.ndarray]) -> list[np.ndarray]:
    """
    This function will destroy audio, use only once.
    """

    valid_audios = [i for i in audios if i is not None]
    if len(valid_audios) > 1:
        am = np.abs(np.concatenate(valid_audios, axis=1)).max() * 32768
    else:
        am = np.abs(valid_audios[0]).max() * 32768
    am = 32767 * 32768 / am

    for i in range(len(audios)):
        if audios[i] is not None:
            np.multiply(audios[i], am, audios[i])
            audios[i] = audios[i].astype(np.int16)
    return audios
