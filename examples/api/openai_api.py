"""
openai_api.py
这个模块实现了一个基于 FastAPI 的语音合成 API，兼容 OpenAI 的接口规范。
This module implements a FastAPI-based text-to-speech API compatible with OpenAI's interface specification.

主要功能和改进：
Main features and improvements:
- 使用 app.state 管理全局状态，确保线程安全
  Use app.state to manage global state, ensuring thread safety
- 添加异常处理和统一的错误响应，提升稳定性
  Add exception handling and unified error responses to improve stability
- 支持多种语音选择和多种音频格式，增加灵活性
  Support multiple voice options and audio formats for greater flexibility
- 增加输入验证，确保请求参数的合法性
  Add input validation to ensure the validity of request parameters
- 支持更多 OpenAI TTS 参数（如 speed），提供更丰富的功能
  Support additional OpenAI TTS parameters (e.g., speed) for richer functionality
- 实现了健康检查端点，便于监控服务状态
  Implement health check endpoint for easy service status monitoring
- 使用异步锁（asyncio.Lock）管理模型访问，提升并发性能
  Use asyncio.Lock to manage model access, improving concurrency performance
- 加载和管理说话者嵌入文件，支持个性化语音合成
  Load and manage speaker embedding files to support personalized speech synthesis
"""
import io
import os
import sys
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

# 跨平台兼容性设置 / Cross-platform compatibility settings
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 设置工作目录并添加到系统路径 / Set working directory and add to system path
now_dir = os.getcwd()
sys.path.append(now_dir)

# 导入必要的模块 / Import necessary modules
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_ogg_view, pcm_arr_to_wav_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

# 初始化日志记录器 / Initialize logger
logger = get_logger("Command")

# 初始化 FastAPI 应用 / Initialize FastAPI application
app = FastAPI()

# 语音映射表 / Voice mapping table
# 下载稳定音色 / Download stable voices:
# 魔塔社区 https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
# HuggingFace https://huggingface.co/spaces/taa/ChatTTS_Speaker
VOICE_MAP = {
    "default": "1528.pt",
    "alloy": "1384.pt",
    "echo": "2443.pt",
    "speak_1000": "1000.pt",
    "speak_1509": "1509.pt",
    "speak_1996": "1996.pt",
    "speak_2115": "2115.pt",
    "speak_2166": "2166.pt",
    "speak_2218": "2218.pt"
}

# 允许的音频格式 / Allowed audio formats
ALLOWED_FORMATS = {"mp3", "wav", "ogg"}

@app.on_event("startup")
async def startup_event():
    """应用启动时加载 ChatTTS 模型及默认说话者嵌入
    Load ChatTTS model and default speaker embedding when the application starts"""
    # 初始化 ChatTTS 和异步锁 / Initialize ChatTTS and async lock
    app.state.chat = ChatTTS.Chat(get_logger("ChatTTS"))
    app.state.model_lock = asyncio.Lock()  # 使用异步锁替代线程锁 / Use async lock instead of thread lock
    
    # 注册文本规范化器 / Register text normalizers
    app.state.chat.normalizer.register("en", normalizer_en_nemo_text())
    app.state.chat.normalizer.register("zh", normalizer_zh_tn())
    
    logger.info("正在初始化 ChatTTS... / Initializing ChatTTS...")
    if app.state.chat.load(source="huggingface"):
        logger.info("模型加载成功。 / Model loaded successfully.")
    else:
        logger.error("模型加载失败，退出应用。 / Model loading failed, exiting application.")
        raise RuntimeError("Failed to load ChatTTS model")
    
    # 加载默认说话者嵌入 / Load default speaker embedding
    default_spk_path = VOICE_MAP["default"]
    if os.path.exists(default_spk_path):
        app.state.spk_emb = torch.load(default_spk_path, map_location=torch.device("cpu"))
        logger.info(f"成功加载默认说话者嵌入文件: {default_spk_path} / Successfully loaded default speaker embedding: {default_spk_path}")
    else:
        app.state.spk_emb = None
        logger.warning(f"未找到默认说话者嵌入文件 {default_spk_path}，使用默认配置。 / Default speaker embedding {default_spk_path} not found, using default configuration.")

# 请求参数白名单 / Request parameter whitelist
ALLOWED_PARAMS = {"model", "input", "voice", "response_format", "speed", "stream", "output_format"}

class OpenAITTSRequest(BaseModel):
    """OpenAI TTS 请求数据模型 / OpenAI TTS request data model"""
    model: str = Field(..., description="语音合成模型，固定为 'tts-1' / Speech synthesis model, fixed as 'tts-1'")
    input: str = Field(..., description="待合成的文本内容 / Text content to synthesize", max_length=2048)  # 限制长度 / Length limit
    voice: Optional[str] = Field("default", description="语音选择，支持: default, alloy, echo / Voice selection, supports: default, alloy, echo")
    response_format: Optional[str] = Field("mp3", description="音频格式: mp3, wav, ogg / Audio format: mp3, wav, ogg")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="语速，范围 0.5-2.0 / Speed, range 0.5-2.0")
    stream: Optional[bool] = Field(True, description="是否流式传输 / Whether to stream")
    output_format: Optional[str] = "mp3"  # 可选格式：mp3, wav, ogg / Optional formats: mp3, wav, ogg
    extra_params: Dict[str, Optional[str]] = Field(default_factory=dict, description="不支持的额外参数 / Unsupported extra parameters")

    @classmethod
    def validate_request(cls, request_data: Dict):
        """过滤不支持的请求参数，并统一 model 值为 'tts-1'
        Filter unsupported request parameters and unify model value to 'tts-1'"""
        request_data["model"] = "tts-1"  # 统一 model 值 / Unify model value
        unsupported_params = set(request_data.keys()) - ALLOWED_PARAMS
        if unsupported_params:
            logger.warning(f"忽略不支持的参数: {unsupported_params} / Ignoring unsupported parameters: {unsupported_params}")
        return {key: request_data[key] for key in ALLOWED_PARAMS if key in request_data}

# 统一错误响应 / Unified error response
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """自定义异常处理 / Custom exception handler"""
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}}
    )

@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """处理语音合成请求 / Handle speech synthesis request"""
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)
    
    logger.info(f"收到请求: text={request.input[:50]}..., voice={request.voice}, stream={request.stream} / Received request: text={request.input[:50]}..., voice={request.voice}, stream={request.stream}")
    
    # 加载指定语音的说话者嵌入 / Load speaker embedding for the specified voice
    spk_path = VOICE_MAP.get(request.voice)
    if spk_path and os.path.exists(spk_path):
        spk_emb = torch.load(spk_path, map_location=torch.device("cpu"))
        logger.info(f"说话人嵌入模型 {spk_path} 加载成功 / Speaker embedding model {spk_path} loaded successfully")
    else:
        spk_emb = app.state.spk_emb
        logger.warning(f"未找到语音 {request.voice} 的嵌入文件，使用默认配置 / Embedding file for voice {request.voice} not found, using default configuration")

    # 推理参数 / Inference parameters
    params_infer_main = {
        "text": [request.input],
        "stream": request.stream,
        "lang": None,
        "skip_refine_text": False,
        "refine_text_only": True,
        "use_decoder": True,
        "audio_seed": 12345678,
        "text_seed": 87654321,
        "do_text_normalization": True,
        "do_homophone_replacement": False,
    }
    
    # 精炼文本参数 / Refine text parameters
    params_refine_text = app.state.chat.RefineTextParams(
        prompt="",
        top_P=0.7,
        top_K=20,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_token=384,
        min_new_token=0,
        show_tqdm=True,
        ensure_non_empty=True,
        manual_seed=None,        
    )    
    # 推理代码参数 / Inference code parameters
    params_infer_code = app.state.chat.InferCodeParams(
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
        pass_first_n_batches=2
    )

    try:
        async with app.state.model_lock:
            # 第一步：单独精炼文本 / Step 1: Refine text separately
            refined_text = app.state.chat.infer(
                text=params_infer_main["text"],
                skip_refine_text=False,
                refine_text_only=True  # 只精炼文本，不生成语音 / Only refine text, do not generate speech
            )
            logger.info(f"Refined text: {refined_text}")
            
            # 第二步：用精炼后的文本生成语音 / Step 2: Generate speech with refined text
            wavs = app.state.chat.infer(
                text=params_infer_main["text"],
                stream=params_infer_main["stream"],
                lang=params_infer_main["lang"],
                skip_refine_text=True,  
                use_decoder=params_infer_main["use_decoder"],
                do_text_normalization=False, 
                do_homophone_replacement=params_infer_main['do_homophone_replacement'],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,   
            )
    except Exception as e:
        raise HTTPException(500, detail=f"语音合成失败 / Speech synthesis failed: {str(e)}")

    # 处理音频输出格式 / Handle audio output format
    def convert_audio(wav, format):
        """将 PCM 数据转换为指定音频格式 / Convert PCM data to specified audio format"""
        if format == "mp3":
            return pcm_arr_to_mp3_view(wav)
        elif format == "wav":
            return pcm_arr_to_wav_view(wav)  
        elif format == "ogg":
            return pcm_arr_to_ogg_view(wav) 
        return pcm_arr_to_mp3_view(wav)  # 默认 / Default

    if request.stream:
        async def audio_stream():
            """生成音频流 / Generate audio stream"""
            for wav in wavs:
                yield convert_audio(wav, request.response_format)
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    
    # 直接返回音频文件 / Return audio file directly
    music_data = convert_audio(wavs[0], request.response_format)
    return StreamingResponse(io.BytesIO(music_data), media_type="audio/mpeg", headers={
        "Content-Disposition": f"attachment; filename=output.{request.response_format}"
    })

@app.get("/health")
async def health_check():
    """健康检查端点 / Health check endpoint"""
    return {"status": "healthy", "model_loaded": bool(app.state.chat)}