"""
openai_api.py

这个模块实现了一个基于 FastAPI 的语音合成 API，兼容 OpenAI 的接口规范。

主要功能：
- 初始化 FastAPI 应用
- 加载 ChatTTS 语音合成模型和预训练的说话者嵌入
- 定义请求数据模型，并提供参数验证
- 实现语音合成接口，支持流式传输

类和函数：
- startup_event: 在应用启动时加载模型和说话者嵌入
- OpenAITTSRequest: 定义请求数据模型，并提供参数验证方法
- generate_voice: 处理语音合成请求，生成并返回音频数据

使用方法：
1. 启动 FastAPI 应用：`uvicorn openai_api:app --host 0.0.0.0 --port 8000`
2. 发送 POST 请求到 `/v1/audio/speech`，请求体包含文本输入及其他可选参数
3. 接收并处理生成的音频数据

注意事项：
- 确保在运行前安装所有依赖库，如 fastapi、pydantic、torch 等
- 根据需要调整模型加载路径和说话者嵌入文件路径
"""

import io
import os
import sys
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# 获取当前工作目录并添加到 Python 路径
now_dir = os.getcwd()
sys.path.append(now_dir)

from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn
import uvicorn



# 初始化日志记录器
logger = get_logger("Command")

# 初始化 FastAPI 应用
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """应用启动时执行，加载 ChatTTS 模型及说话者嵌入"""
    global chat, spk_emb
    
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    # 注册文本正则化模块
    chat.normalizer.register("en", normalizer_en_nemo_text())
    chat.normalizer.register("zh", normalizer_zh_tn())
    
    logger.info("正在初始化 ChatTTS...")
    if chat.load(source="huggingface"):
        logger.info("模型加载成功。")
    else:
        logger.error("模型加载失败。")
        sys.exit(1)
    
    # 加载预训练的说话者嵌入文件
    spk_emb_path = "1000.pt"
    if os.path.exists(spk_emb_path):
        spk_emb = torch.load(spk_emb_path, map_location=torch.device("cpu"))
        logger.info(f"成功加载说话者嵌入文件: {spk_emb_path}")
    else:
        spk_emb = None
        logger.warning(f"未找到说话者嵌入文件 {spk_emb_path}，将使用默认配置。")

# 允许的请求参数
ALLOWED_PARAMS = {"model", "input", "voice", "response_format", "stream", "output_format"}

class OpenAITTSRequest(BaseModel):
    """定义 OpenAI TTS 请求数据模型，并提供参数验证。
    其实也就input 和 stream 有用。 其他的参数都忽略了。
    输出的音频格式使用的mp3, 没有对其他格式的音频输出做设置。
    输出的声音特征在generate_voice函数中调整。"""
    
    model: str = Field(..., description="语音合成模型，将所有输入统一转换为 'tts-1'")
    input: str = Field(..., description="待合成的文本内容")
    voice: Optional[str] = "default"
    response_format: Optional[str] = "mp3"
    stream: Optional[bool] = False
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

@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """ 处理语音合成请求，并返回音频数据 """
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)
    
    logger.info(f"收到文本输入: {request.input}")
    
    # main infer params
    params_infer_main = {
        "text": [request.input],
        "stream": request.stream,
        "lang": None,
        "skip_refine_text": True,
        "refine_text_only": False,
        "use_decoder": True,
        "audio_seed": 12345678,
        "text_seed": 87654321,
        "do_text_normalization": True,
        "do_homophone_replacement": False,
    }
    
    # refine text params
    params_refine_text = ChatTTS.Chat.RefineTextParams(
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
    params_infer_code = ChatTTS.Chat.InferCodeParams(
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
    
    # 进行语音合成
    wavs = chat.infer(
        text = params_infer_main['text'],
        stream = params_infer_main["stream"],
        lang = None,  # 只支持中英文，模型自己就识别了，不用设置。
        skip_refine_text = params_infer_main['skip_refine_text'],
        use_decoder = params_infer_main['use_decoder'],
        do_text_normalization = params_infer_main['do_text_normalization'],
        do_homophone_replacement = params_infer_main['do_homophone_replacement'],
        params_refine_text = params_refine_text,
        params_infer_code = params_infer_code,   
    )
    
    if request.stream:
        async def audio_stream():
            for wav in wavs:
                yield pcm_arr_to_mp3_view(wav)
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    
    # 直接返回 MP3 文件
    mp3_data = pcm_arr_to_mp3_view(wavs[0])
    return StreamingResponse(io.BytesIO(mp3_data), media_type="audio/mpeg", headers={
        "Content-Disposition": "attachment; filename=output.mp3"
    })

#if __name__ == "__main__":
    # 运行 FastAPI 应用
    #uvicorn.run(app, host="0.0.0.0", port=8000)
