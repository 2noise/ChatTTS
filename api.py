import os
import re
import sys

import numpy as np
import torch
from fastapi import FastAPI, Response
from loguru import logger
from pydantic import BaseModel, Field
from pydub import AudioSegment

import ChatTTS

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "service.log")
valid_pattern = re.compile(r"[^\u4e00-\u9fffA-Za-z，。、,\. \[\]\_]")


def logger_setting(logfile: str, log_level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(logfile, level=log_level, rotation="10MB", retention=20)


logger_setting(LOG_FILE, LOG_LEVEL)


def deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


app = FastAPI(title="ChatTTS Api")

logger.info("Loading models...")
chat = ChatTTS.Chat()
chat.load_models(source="local", local_path="tts_model", compile=False)
# compile=True is faster, but it will take longer to start
logger.info("Models loaded, warming up...")
chat.infer(["你好"], use_decoder=True)
logger.info("Warmup done, ready to serve requests!")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Input text to be converted to speech")
    seed: int = Field(1111, description="Voice Seed, int")
    top_P: float = Field(0.7, description="Top P value for sampling")
    top_K: int = Field(20, description="Top K value for sampling")
    temperature: float = Field(0.3, description="Temperature value for sampling")
    skip_refine_text: bool = Field(False, description="Whether to refine the text")
    refine_text_prompt: str = Field(
        "[oral_2][laugh_0][break_6]", description="Prompt for refining the text"
    )


def wav_to_mp3(wav_data, sample_rate=24000, bitrate="48k"):
    audio = AudioSegment(
        data=wav_data, sample_width=2, frame_rate=sample_rate, channels=1
    )
    return audio.export(format="mp3", bitrate=bitrate)


def infer(request):
    # remove invalid characters, otherwise the model will raise an error
    # text = valid_pattern.sub("", request.text)
    text = request.text
    logger.info(f"Text after removing invalid characters: {text}")
    deterministic(request.seed)
    rnd_spk_emb = chat.sample_random_speaker()

    params_infer_code = {
        "spk_emb": rnd_spk_emb,
        "temperature": request.temperature,
        "top_P": request.top_P,
        "top_K": request.top_K,
    }

    params_refine_text = {}
    if request.skip_refine_text:
        params_refine_text = {"prompt": request.refine_text_prompt}

    wavs = chat.infer(
        [text],
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        skip_refine_text=request.skip_refine_text,
    )
    audio_data = wavs[0][0]
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32768).astype(np.int16)
    return wav_to_mp3(audio_data).read()


@app.post("/tts")
async def tts_stream(
    request: TTSRequest,
) -> Response:
    logger.info(f"Received request: {request}")

    audio = infer(request)
    logger.info("Generated audio")
    return Response(audio, media_type="audio/mp3")
