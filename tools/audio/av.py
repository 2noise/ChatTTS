from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict

import av
from av.audio.resampler import AudioResampler
import numpy as np


video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}


def wav2(i: BytesIO, o: BufferedWriter, format: str):
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L20
    """
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, format)

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> np.ndarray:
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L39
    """

    if not Path(file).exists():
        raise FileNotFoundError(f"File not found: {file}")

    try:
        container = av.open(file)
        resampler = AudioResampler(format="fltp", layout="mono", rate=sr)

        # Estimated maximum total number of samples to pre-allocate the array
        # AV stores length in microseconds by default
        estimated_total_samples = int(container.duration * sr // 1_000_000)
        decoded_audio = np.zeros(estimated_total_samples + 1, dtype=np.float32)

        offset = 0
        for frame in container.decode(audio=0):
            frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            resampled_frames = resampler.resample(frame)
            for resampled_frame in resampled_frames:
                frame_data = resampled_frame.to_ndarray()[0]
                end_index = offset + len(frame_data)

                # Check if decoded_audio has enough space, and resize if necessary
                if end_index > decoded_audio.shape[0]:
                    decoded_audio = np.resize(decoded_audio, end_index + 1)

                decoded_audio[offset:end_index] = frame_data
                offset += len(frame_data)

        # Truncate the array to the actual size
        decoded_audio = decoded_audio[:offset]
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return decoded_audio
