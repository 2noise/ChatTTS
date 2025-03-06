import io
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn
import uvicorn

now_dir = os.getcwd()
sys.path.append(now_dir)

# 初始化 FastAPI
logger = get_logger("Command")
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global chat, spk_emb
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    chat.normalizer.register("en", normalizer_en_nemo_text())
    chat.normalizer.register("zh", normalizer_zh_tn())
    logger.info("Initializing ChatTTS...")
    if chat.load(source="huggingface"):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)
    
    # 加载预训练的嵌入式模型
    spk_emb_path = "2443.pt"
    if os.path.exists(spk_emb_path):
        spk_emb = torch.load(spk_emb_path, map_location=torch.device("cpu"))
        logger.info(f"Loaded speaker embedding from {spk_emb_path}")
    else:
        spk_emb = None
        logger.warning(f"Speaker embedding file {spk_emb_path} not found, using default.")

# 允许的参数列表
ALLOWED_PARAMS = {"model", "input", "voice", "response_format", "stream", "output_format"}

class OpenAITTSRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., description="Text input for speech synthesis")
    voice: Optional[str] = "default"
    response_format: Optional[str] = "mp3"
    stream: Optional[bool] = False
    output_format: Optional[str] = "mp3"  # 可选：mp3, wav, ogg
    extra_params: Dict[str, Optional[str]] = Field(default_factory=dict, description="Unsupported parameters")

    @classmethod
    def validate_request(cls, request_data: Dict):
        unsupported_params = set(request_data.keys()) - ALLOWED_PARAMS
        if unsupported_params:
            logger.warning(f"Ignoring unsupported parameters: {unsupported_params}")
        return {key: request_data[key] for key in ALLOWED_PARAMS if key in request_data}

@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """ OpenAI 兼容的语音生成接口，自动过滤不支持的参数 """
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)
    
    logger.info(f"Received text input: {request.input}")
    
    # 仅支持 spk_emb 参数
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt = "[speed_5]",    # 输入的提示文本，用于引导模型生成内容。这里默认值是 "[speed_5]"，控制速度
        top_P = 0.5,             # Top-p 采样（也叫核采样，nucleus sampling）的概率阈值。模型会从累积概率达到 top_P 的最小词集合中采样。
        top_K = 10,              # Top-k 采样，模型从概率最高的前 top_K 个词中选择。
        temperature = 0.1,       # 温度参数，控制生成随机性。值越低，模型越倾向于选择高概率的词；值越高，随机性越强。
        repetition_penalty = 1.1, # 重复惩罚参数，用于减少生成内容中词语或短语的重复。值大于 1 表示施加惩罚。
        max_new_token = 2048,    # 生成的最大新 token 数量（可能是单词、子词等）。
        min_new_token = 0,       # 生成的最小新 token 数量。
        show_tqdm = True,        # 是否显示进度条（基于 Python 的 tqdm 库）。
        ensure_non_empty = True, # 确保生成结果非空。
        manual_seed = 42,        # 随机种子，用于控制生成的可重复性。
        spk_emb = spk_emb,       # “speaker embedding”（说话者嵌入）的路径或标识，用于指定语音生成的说话者特征。
        spk_smp = None,          # “speaker sample”（说话者样本）的路径，用于提供说话者参考音频。
        txt_smp = None,          # “text sample”（文本样本）的路径，用于提供参考文本。
        stream_batch = 24,       # 流式生成时的批次大小。
        stream_speed = 12000,    # 流式生成的处理速度（可能是每秒处理的样本数或 token 数）。
        pass_first_n_batches = 2 # 跳过前 N 个批次（可能是为了预热或避免初始不稳定输出）。
        )
    
    # 进行语音合成
    wavs = chat.infer(
        text=[request.input],
        stream=request.stream,
        lang="zh" if request.voice == "zh" else "en",
        use_decoder=True,
        do_text_normalization=True,
        params_infer_code=params_infer_code,
    )
    
    if request.stream:
        async def audio_stream():
            for wav in wavs:
                yield pcm_arr_to_mp3_view(wav)
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    
    # 直接返回 MP3 文件
    mp3_data = pcm_arr_to_mp3_view(wavs[0])  # 仅返回第一段音频
    return StreamingResponse(io.BytesIO(mp3_data), media_type="audio/mpeg", headers={
        "Content-Disposition": "attachment; filename=output.mp3"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)