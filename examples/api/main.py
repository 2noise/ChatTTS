import os
import sys

import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from tools.audio.np import pcm_to_wav_bytes, pcm_to_bytes

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional, AsyncGenerator

import ChatTTS

from tools.logger import get_logger
import torch


from pydantic import BaseModel


logger = get_logger("Command")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global chat

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load(use_vllm=True):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)


class ChatTTSParams(BaseModel):
    text: list[str]
    stream: bool = False
    lang: Optional[str] = None
    skip_refine_text: bool = False
    refine_text_only: bool = False
    use_decoder: bool = True
    do_text_normalization: bool = True
    do_homophone_replacement: bool = False
    params_refine_text: Optional[ChatTTS.Chat.RefineTextParams] = None
    params_infer_code: Optional[ChatTTS.Chat.InferCodeParams] = None
    stream_batch_size: int = 16


@app.post("/generate_voice")
async def generate_voice(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # audio seed
    if params.params_infer_code.manual_seed is not None:
        torch.manual_seed(params.params_infer_code.manual_seed)
        params.params_infer_code.spk_emb = chat.sample_random_speaker()

    # text seed for text refining
    if params.params_refine_text and params.skip_refine_text is False:
        results_generator = chat.infer(
            text=params.text, skip_refine_text=False, refine_text_only=True
        )
        text = await next(results_generator)
        logger.info(f"Refined text: {text}")
    else:
        # no text refining
        text = params.text

    logger.info("Use speaker:")
    logger.info(params.params_infer_code.spk_emb)

    logger.info("Start voice inference.")

    results_generator = chat.infer(
        text=text,
        stream=params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )

    if params.stream:

        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for output in results_generator:
                yield pcm_to_bytes(output[0])

        return StreamingResponse(
            content=stream_results(), media_type="text/event-stream"
        )

    output = None
    async for request_output in results_generator:
        if output is None:
            output = request_output[0]
        else:
            output = np.concatenate((output, request_output[0]), axis=0)
    output = pcm_to_wav_bytes(output)
    return Response(
        content=output, media_type="audio/wav", headers={"Cache-Control": "no-cache"}
    )
