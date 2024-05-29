import io
import os
import wave
import numpy as np
from fastapi import FastAPI, Query, Depends
from fastapi.responses import StreamingResponse, Response

from omegaconf import OmegaConf
from pydantic import BaseModel

import ChatTTS
import uvicorn

import argparse
from loguru import logger


class TTS(BaseModel):
    """TTS GET/POST request"""
    text: str = Query("欢迎使用ChatTTS API", description="text to synthesize")


class Chat(ChatTTS.Chat):
    """重写一下load_models方法，方便我们自定义models的路径，照顾国内的宝宝"""

    def load_models(self, source='', force_redownload=False, local_path='<LOCAL_PATH>'):
        download_path = source
        self._load(**{k: os.path.join(download_path, v) for k, v in
                      OmegaConf.load(os.path.join(download_path, 'config', 'path.yaml')).items()})


app = FastAPI()

chat = Chat()


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


def infer(text):
    # 热心群友1142052965提的建议
    audio_data = chat.infer([text], use_decoder=True)[0]
    audio_data = audio_data / np.max(np.abs(audio_data))
    chunks = (audio_data * 32768).astype(np.int16)
    yield wave_header_chunk()
    for chunk in chunks:
        if chunk is not None:
            chunk = chunk.tobytes()
            yield chunk


@app.get("/")
async def index(params: TTS = Depends(TTS)):
    logger.debug(params)
    wavs = infer(params.text)
    return StreamingResponse(wavs, media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12456)
    parser.add_argument("--model-dir", type=str, default="./models")
    args = parser.parse_args()
    logger.debug(f"model_dir: {args.model_dir}")
    chat.load_models(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)
