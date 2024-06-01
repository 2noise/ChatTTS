import io

import os
import wave
import numpy as np
import torch
from fastapi import FastAPI, Query, Depends
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from LangSegment import LangSegment
import ChatTTS
import uvicorn

import argparse
from loguru import logger
from utils.zh_normalization import text_normalize
from utils.text_split_method import text_split_registry


class TTS(BaseModel):
    """TTS GET/POST request"""
    text: str = Query("欢迎使用ChatTTS API", description="text to synthesize")
    spk: str = Query("random", description="speaker id")
    text_split_method: str = Query("cut2", description="text split mode")


class Speaker(BaseModel):
    name: str = Query("", description="speaker name")


app = FastAPI()

chat = ChatTTS.Chat()
args = None

speaker = {}

curr_speaker = None
LangSegment.setfilters(["zh", "ja", "en"])


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


def infer(texts, spk="random"):
    global curr_speaker
    if speaker.get(spk) is None or spk == "random":
        spk_emb = chat.sample_random_speaker()
        curr_speaker = spk_emb
    else:
        spk_emb = speaker[spk]
    params_infer_code = {'spk_emb': spk_emb, }
    yield wave_header_chunk()
    for text in texts:
        audio_data = chat.infer(text, use_decoder=True,
                                params_infer_code=params_infer_code,
                                do_text_normalization=False)[0]
        audio_data = audio_data / np.max(np.abs(audio_data))
        chunks = (audio_data * 32768).astype(np.int16)
        for chunk in chunks:
            if chunk is not None:
                chunk = chunk.tobytes()
                yield chunk


def tts_handle(params: TTS):
    logger.debug(params)
    # 主要是为了格式化一下数字和一些文字的读法

    # 将长文本分割成短文本,最好就用cut2
    texts = text_split_registry[params.text_split_method](params.text)
    text_list = []
    for text in texts:
        for tmp in LangSegment.getTexts(text):
            normalize = text_normalize(tmp.get("text"))
            logger.debug(f"{tmp} {normalize}")
            if normalize != "" and tmp.get("lang") == "en" and normalize not in ["."]:
                if len(text_list) > 0:
                    text_list[-1] += normalize
                else:
                    text_list.append(normalize)
            elif tmp.get("lang") == "zh":
                text_list.append(normalize)
            else:
                text_list.append(tmp.get("text"))
    wavs = infer(text_list, params.spk)
    logger.debug(text_list)
    return StreamingResponse(wavs, media_type="audio/wav")


@app.get("/")
async def index(params: TTS = Depends(TTS)):
    return tts_handle(params)


@app.post("/")
async def index_post(params: TTS):
    return tts_handle(params)


@app.post("/speaker")
async def speaker_handle(params: Speaker):
    if params.name != "":
        # 使用torch将上一个发音人embedding保存到本地
        torch.save(curr_speaker, os.path.join(args.spk_dir, f"{params.name}.pt"))
        speaker[params.name] = curr_speaker
    return {'code': 200, 'msg': 'success'}


@app.get("/speaker")
async def speaker_list():
    l = list(speaker.keys())
    return {'code': 200, 'msg': 'success', 'data': l}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12456)
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--spk-dir", type=str, default="./configs")
    args = parser.parse_args()
    logger.debug(f"model_dir: {args.model_dir}")
    for file in os.listdir(args.spk_dir):
        if file.endswith(".pt"):
            logger.debug(f"loading speaker model:{file[:-3]}")
            speaker[file[:-3]] = torch.load(os.path.join(args.spk_dir, file))
    chat.load_models(source="local", force_redownload=False, local_path=args.model_dir, compile=args.compile)
    uvicorn.run(app, host=args.host, port=args.port)
