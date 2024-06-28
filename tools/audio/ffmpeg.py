from pydub.utils import which


def has_ffmpeg_installed() -> bool:
    return which("ffmpeg") and which("ffprobe")
