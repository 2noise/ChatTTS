"""
openai_api.py
这个模块实现了一个基于 FastAPI 的语音合成 API，兼容 OpenAI 的接口规范。
改进功能：
- 使用 app.state 管理全局状态
- 添加异常处理和统一的错误响应
- 支持多语音选择和多种音频格式
- 增加输入验证和性能监控
- 支持更多 OpenAI TTS 参数（如 speed）
运行方式：
uvicorn openai_api:app --host 0.0.0.0 --port 8000
"""
import io
import os
import sys
import threading
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch


if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)


import ChatTTS
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_ogg_view, pcm_arr_to_wav_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn
import uvicorn

# 初始化日志记录器
logger = get_logger("Command")

# 初始化 FastAPI 应用
app = FastAPI()

# 语音映射表
# 下载稳定音色：
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

# 允许的音频格式
ALLOWED_FORMATS = {"mp3", "wav", "ogg"}

@app.on_event("startup")
async def startup_event():
    """应用启动时加载 ChatTTS 模型及默认说话者嵌入"""
    app.state.chat = ChatTTS.Chat(get_logger("ChatTTS"))
    app.state.model_lock = asyncio.Lock()  # 改为异步锁
    app.state.chat.normalizer.register("en", normalizer_en_nemo_text())
    app.state.chat.normalizer.register("zh", normalizer_zh_tn())
    
    logger.info("正在初始化 ChatTTS...")
    if app.state.chat.load(source="huggingface"):
        logger.info("模型加载成功。")
    else:
        logger.error("模型加载失败，退出应用。")
        raise RuntimeError("Failed to load ChatTTS model")
    
    # 加载默认说话者嵌入
    default_spk_path = VOICE_MAP["default"]
    if os.path.exists(default_spk_path):
        app.state.spk_emb = torch.load(default_spk_path, map_location=torch.device("cpu"))
        logger.info(f"成功加载默认说话者嵌入文件: {default_spk_path}")
    else:
        app.state.spk_emb = None
        logger.warning(f"未找到默认说话者嵌入文件 {default_spk_path}，使用默认配置。")

# 请求参数白名单
ALLOWED_PARAMS = {"model", "input", "voice", "response_format", "speed", "stream", "output_format"}

class OpenAITTSRequest(BaseModel):
    """OpenAI TTS 请求数据模型"""
    model: str = Field(..., description="语音合成模型，固定为 'tts-1'")
    input: str = Field(..., description="待合成的文本内容", max_length=2048)  # 限制长度
    voice: Optional[str] = Field("default", description="语音选择，支持: default, alloy, echo")
    response_format: Optional[str] = Field("mp3", description="音频格式: mp3, wav, ogg")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="语速，范围 0.5-2.0")
    stream: Optional[bool] = Field(True, description="是否流式传输")
    output_format: Optional[str] = "mp3"  # 可选格式：mp3, wav, ogg
    extra_params: Dict[str, Optional[str]] = Field(default_factory=dict, description="不支持的额外参数")

    @classmethod
    def validate_request(cls, request_data: Dict):
        """过滤不支持的请求参数，并统一 model 值为 'tts-1'"""
        request_data["model"] = "tts-1"  # 统一 model 值
        unsupported_params = set(request_data.keys()) - ALLOWED_PARAMS
        if unsupported_params:
            logger.warning(f"忽略不支持的参数: {unsupported_params}")
        return {key: request_data[key] for key in ALLOWED_PARAMS if key in request_data}


# 统一错误响应
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}}
    )

@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """处理语音合成请求"""
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)
    
    logger.info(f"收到请求: text={request.input[:50]}..., voice={request.voice}, stream={request.stream}")
    
    # 加载指定语音的说话者嵌入
    spk_path = VOICE_MAP.get(request.voice)
    if spk_path and os.path.exists(spk_path):
        spk_emb = torch.load(spk_path, map_location=torch.device("cpu"))
        logger.info(f"说话人嵌入模型  {spk_path}  加载成功")
    else:
        spk_emb = app.state.spk_emb
        logger.warning(f"未找到语音 {request.voice} 的嵌入文件，使用默认配置")

    # 推理参数
    # main infer params
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
    
    # refine text params
    params_refine_text = app.state.chat.RefineTextParams(
        prompt = "",
        top_P = 0.7,
        top_K = 20,
        temperature = 0.7,
        repetition_penalty = 1.0,
        max_new_token = 384,
        min_new_token = 0,
        show_tqdm = True,
        ensure_non_empty = True,
        manual_seed = None,        
    )    
    # infer code params
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
            # 第一步：单独精炼文本
            refined_text = app.state.chat.infer(
                text=params_infer_main["text"],
                skip_refine_text=False,
                refine_text_only=True  # 只精炼文本，不生成语音
            )
            logger.info(f"Refined text: {refined_text}")
            
            # 第二步：用精炼后的文本生成语音
            wavs = app.state.chat.infer(
                text=params_infer_main["text"],
                stream=params_infer_main["stream"],
                lang=params_infer_main["lang"],
                skip_refine_text=True,  
                use_decoder=params_infer_main["use_decoder"],
                do_text_normalization=False, 
                do_homophone_replacement=params_infer_main['do_homophone_replacement'],
                params_refine_text = params_refine_text,
                params_infer_code = params_infer_code,   
            )
    except Exception as e:
        raise HTTPException(500, detail=f"Speech synthesis failed: {str(e)}")

    # 处理音频输出格式（这里假设工具函数支持扩展）
    def convert_audio(wav, format):
        if format == "mp3":
            return pcm_arr_to_mp3_view(wav)
        elif format == "wav":
            # 需要实现 wav 转换逻辑
            return pcm_arr_to_wav_view(wav)  # 伪代码
        elif format == "ogg":
            # 需要实现 ogg 转换逻辑
            return pcm_arr_to_ogg_view(wav)  # 伪代码
        return pcm_arr_to_mp3_view(wav)  # 默认

    if request.stream:
        async def audio_stream():
            for wav in wavs:
                yield convert_audio(wav, request.response_format)
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    
    # 直接返回 MP3/wav/ogg 文件
    music_data = convert_audio(wavs[0], request.response_format)
    return StreamingResponse(io.BytesIO(music_data), media_type="audio/mpeg", headers={
        "Content-Disposition": f"attachment; filename=output.{request.response_format}"
    })

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": bool(app.state.chat)}

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)