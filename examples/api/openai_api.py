"""
openai_api.py
This module implements a FastAPI-based text-to-speech API compatible with OpenAI's interface specification.

Main features and improvements:
- Use app.state to manage global state, ensuring thread safety
- Add exception handling and unified error responses to improve stability
- Support multiple voice options and audio formats for greater flexibility
- Add input validation to ensure the validity of request parameters
- Support additional OpenAI TTS parameters (e.g., speed) for richer functionality
- Implement health check endpoint for easy service status monitoring
- Use asyncio.Lock to manage model access, improving concurrency performance
- Load and manage speaker embedding files to support personalized speech synthesis
"""

import io
import os
import sys
import asyncio
import time
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

# Cross-platform compatibility settings
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set working directory and add to system path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Import necessary modules
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_ogg_view, pcm_arr_to_wav_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

# Initialize logger
logger = get_logger("Command")

# Initialize FastAPI application
app = FastAPI()

# Voice mapping table
# Download stable voices:
# ModelScope Community: https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
# HuggingFace: https://huggingface.co/spaces/taa/ChatTTS_Speaker
VOICE_MAP = {
    "default": "1528.pt",
    "alloy": "1384.pt",
    "echo": "2443.pt",
}

# Allowed audio formats
ALLOWED_FORMATS = {"mp3", "wav", "ogg"}


@app.on_event("startup")
async def startup_event():
    """Load ChatTTS model and default speaker embedding when the application starts"""
    # Initialize ChatTTS and async lock
    app.state.chat = ChatTTS.Chat(get_logger("ChatTTS"))
    app.state.model_lock = asyncio.Lock()  # Use async lock instead of thread lock

    # Register text normalizers
    app.state.chat.normalizer.register("en", normalizer_en_nemo_text())
    app.state.chat.normalizer.register("zh", normalizer_zh_tn())

    logger.info("Initializing ChatTTS...")
    if app.state.chat.load(source="huggingface"):
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model loading failed, exiting application.")
        raise RuntimeError("Failed to load ChatTTS model")

    # Load default speaker embedding
    # Preload all supported speaker embeddings into memory at startup to avoid repeated loading during runtime
    app.state.spk_emb_map = {}
    for voice, spk_path in VOICE_MAP.items():
        if os.path.exists(spk_path):
            app.state.spk_emb_map[voice] = torch.load(
                spk_path, map_location=torch.device("cpu")
            )
            logger.info(f"Preloading speaker embedding: {voice} -> {spk_path}")
        else:
            logger.warning(f"Speaker embedding not found: {spk_path}, skipping preload")
    app.state.spk_emb = app.state.spk_emb_map.get("default")  # Default embedding


# Request parameter whitelist
ALLOWED_PARAMS = {
    "model",
    "input",
    "voice",
    "response_format",
    "speed",
    "stream",
    "output_format",
}


class OpenAITTSRequest(BaseModel):
    """OpenAI TTS request data model"""

    model: str = Field(..., description="Speech synthesis model, fixed as 'tts-1'")
    input: str = Field(
        ..., description="Text content to synthesize", max_length=2048
    )  # Length limit
    voice: Optional[str] = Field(
        "default", description="Voice selection, supports: default, alloy, echo"
    )
    response_format: Optional[str] = Field(
        "mp3", description="Audio format: mp3, wav, ogg"
    )
    speed: Optional[float] = Field(
        1.0, ge=0.5, le=2.0, description="Speed, range 0.5-2.0"
    )
    stream: Optional[bool] = Field(False, description="Whether to stream")
    output_format: Optional[str] = "mp3"  # Optional formats: mp3, wav, ogg
    extra_params: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Unsupported extra parameters"
    )

    @classmethod
    def validate_request(cls, request_data: Dict):
        """Filter unsupported request parameters and unify model value to 'tts-1'"""
        request_data["model"] = "tts-1"  # Unify model value
        unsupported_params = set(request_data.keys()) - ALLOWED_PARAMS
        if unsupported_params:
            logger.warning(f"Ignoring unsupported parameters: {unsupported_params}")
        return {key: request_data[key] for key in ALLOWED_PARAMS if key in request_data}


# Unified error response
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """Custom exception handler"""
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}},
    )


@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """Handle speech synthesis request"""
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)

    logger.info(
        f"Received request: text={request.input}..., voice={request.voice}, stream={request.stream}"
    )

    # Validate audio format
    if request.response_format not in ALLOWED_FORMATS:
        raise HTTPException(
            400,
            detail=f"Unsupported audio format: {request.response_format}, supported formats: {', '.join(ALLOWED_FORMATS)}",
        )

    # Load speaker embedding for the specified voice
    spk_emb = app.state.spk_emb_map.get(request.voice, app.state.spk_emb)

    # Inference parameters
    params_infer_main = {
        "text": [request.input],
        "stream": request.stream,
        "lang": None,
        "skip_refine_text": True,  # Do not use text refinement
        "refine_text_only": False,
        "use_decoder": True,
        "audio_seed": 12345678,
        # "text_seed": 87654321,  # Random seed for text processing, used to control text refinement
        "do_text_normalization": True,  # Perform text normalization
        "do_homophone_replacement": True,  # Perform homophone replacement
    }

    # Inference code parameters
    params_infer_code = app.state.chat.InferCodeParams(
        # prompt=f"[speed_{int(request.speed * 10)}]",  # Convert to format supported by ChatTTS
        prompt="[speed_5]",
        top_P=0.5,
        top_K=10,
        temperature=0.1,
        repetition_penalty=1.1,
        max_new_token=2048,
        min_new_token=0,
        show_tqdm=True,
        ensure_non_empty=True,
        manual_seed=42,
        spk_emb=spk_emb,
        spk_smp=None,
        txt_smp=None,
        stream_batch=24,
        stream_speed=12000,
        pass_first_n_batches=2,
    )

    try:
        async with app.state.model_lock:
            wavs = app.state.chat.infer(
                text=params_infer_main["text"],
                stream=params_infer_main["stream"],
                lang=params_infer_main["lang"],
                skip_refine_text=params_infer_main["skip_refine_text"],
                use_decoder=params_infer_main["use_decoder"],
                do_text_normalization=params_infer_main["do_text_normalization"],
                do_homophone_replacement=params_infer_main["do_homophone_replacement"],
                # params_refine_text = params_refine_text,
                params_infer_code=params_infer_code,
            )
    except Exception as e:
        raise HTTPException(500, detail=f"Speech synthesis failed: {str(e)}")

    def generate_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
        """Generate WAV file header (without data length)"""
        header = bytearray()
        header.extend(b"RIFF")
        header.extend(b"\xff\xff\xff\xff")  # File size unknown
        header.extend(b"WAVEfmt ")
        header.extend((16).to_bytes(4, "little"))  # fmt chunk size
        header.extend((1).to_bytes(2, "little"))  # PCM format
        header.extend((channels).to_bytes(2, "little"))  # Channels
        header.extend((sample_rate).to_bytes(4, "little"))  # Sample rate
        byte_rate = sample_rate * channels * bits_per_sample // 8
        header.extend((byte_rate).to_bytes(4, "little"))  # Byte rate
        block_align = channels * bits_per_sample // 8
        header.extend((block_align).to_bytes(2, "little"))  # Block align
        header.extend((bits_per_sample).to_bytes(2, "little"))  # Bits per sample
        header.extend(b"data")
        header.extend(b"\xff\xff\xff\xff")  # Data size unknown
        return bytes(header)

    # Handle audio output format
    def convert_audio(wav, format):
        """Convert audio format"""
        if format == "mp3":
            return pcm_arr_to_mp3_view(wav)
        elif format == "wav":
            return pcm_arr_to_wav_view(
                wav, include_header=False
            )  # No header in streaming
        elif format == "ogg":
            return pcm_arr_to_ogg_view(wav)
        return pcm_arr_to_mp3_view(wav)

    # Return streaming audio data
    if request.stream:
        first_chunk = True

        async def audio_stream():
            nonlocal first_chunk
            for wav in wavs:
                if request.response_format == "wav" and first_chunk:
                    yield generate_wav_header()  # Send WAV header
                    first_chunk = False
                yield convert_audio(wav, request.response_format)

        media_type = "audio/wav" if request.response_format == "wav" else "audio/mpeg"
        return StreamingResponse(audio_stream(), media_type=media_type)

    # Return audio file directly
    if request.response_format == "wav":
        music_data = pcm_arr_to_wav_view(wavs[0])
    else:
        music_data = convert_audio(wavs[0], request.response_format)

    return StreamingResponse(
        io.BytesIO(music_data),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=output.{request.response_format}"
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": bool(app.state.chat)}
