import math

import numpy as np
from numba import jit
import torch


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def ndarray_to_tensor(audio: np.ndarray) -> torch.Tensor:
    # Assuming 'wavs' is a NumPy array of shape (num_samples,) or (num_channels, num_samples)
    wav_tensor = torch.from_numpy(audio.astype(np.float32))  # Ensure data is float32

    # If 'wavs' is 1D, add a channel dimension
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)

    return wav_tensor
