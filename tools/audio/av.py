from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import av
from av.audio.frame import AudioFrame
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


def load_audio(
    file: Union[str, BytesIO, Path],
    sr: Optional[int] = None,
    format: Optional[str] = None,
    mono=True,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L39
    """
    if (isinstance(file, str) and not Path(file).exists()) or (
        isinstance(file, Path) and not file.exists()
    ):
        raise FileNotFoundError(f"File not found: {file}")
    rate = 0

    container = av.open(file, format=format)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    channels = 1 if audio_stream.layout == "mono" else 2
    container.seek(0)
    resampler = (
        AudioResampler(format="fltp", layout=audio_stream.layout, rate=sr)
        if sr is not None
        else None
    )

    # Estimated maximum total number of samples to pre-allocate the array
    # AV stores length in microseconds by default
    estimated_total_samples = (
        int(container.duration * sr // 1_000_000) if sr is not None else 48000
    )
    decoded_audio = np.zeros(
        (
            estimated_total_samples + 1
            if channels == 1
            else (channels, estimated_total_samples + 1)
        ),
        dtype=np.float32,
    )

    offset = 0

    def process_packet(packet: List[AudioFrame]):
        frames_data = []
        rate = 0
        for frame in packet:
            # frame.pts = None  # 清除时间戳，避免重新采样问题
            resampled_frames = (
                resampler.resample(frame) if resampler is not None else [frame]
            )
            for resampled_frame in resampled_frames:
                frame_data = resampled_frame.to_ndarray()
                rate = resampled_frame.rate
                frames_data.append(frame_data)
        return (rate, frames_data)

    def frame_iter(container):
        for p in container.demux(container.streams.audio[0]):
            yield p.decode()

    for r, frames_data in map(process_packet, frame_iter(container)):
        if not rate:
            rate = r
        for frame_data in frames_data:
            end_index = offset + len(frame_data[0])

            # 检查 decoded_audio 是否有足够的空间，并在必要时调整大小
            if end_index > decoded_audio.shape[1]:
                decoded_audio = np.resize(
                    decoded_audio, (decoded_audio.shape[0], end_index * 4)
                )

            np.copyto(decoded_audio[..., offset:end_index], frame_data)
            offset += len(frame_data[0])

    container.close()

    # Truncate the array to the actual size
    decoded_audio = decoded_audio[..., :offset]

    if mono and decoded_audio.shape[0] > 1:
        decoded_audio = decoded_audio.mean(0)

    if sr is not None:
        return decoded_audio
    return decoded_audio, rate
