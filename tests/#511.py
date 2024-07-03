import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import ChatTTS

from tools.logger import get_logger

logger = get_logger("Test #511")

chat = ChatTTS.Chat(logger)
chat.load(compile=False) # Set to True for better performance

texts = ["语音太短了会造成生成音频错误, 这是占位占位, 老大爷觉得车夫的想法很有道理", 
         "评分只是衡量音色的稳定性,不代表音色的好坏, 可以根据自己的需求选择合适的音色",
         "举个简单的例子,如果一个沙哑且结巴的音色一直很稳定，那么它的评分就会很高。",
         "语音太短了会造成生成音频错误, 这是占位占位。我使用 seed id 去生成音频, 但是生成的音频不稳定",
         "seed id只是一个参考ID 不同的环境下音色不一定一致。还是推荐使用 .pt 文件载入音色",
         "语音太短了会造成生成音频错误, 这是占位占位。音色标的男女准确吗",
         "当前第一批测试的音色有两千条, 根据声纹相似性简单打标, 准确度不高, 特别是特征一项",
         "语音太短了会造成生成音频错误, 这是占位占位。仅供参考。如果大家有更好的标注方法,欢迎 PR。",
         ]

rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.005,        # top P decode
    top_K = 1,         # top K decode
)

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_0][laugh_0][break_4]',
)

fail = False

for i in range(4):

    wavs = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    for k, wav in enumerate(wavs):
        if wav is None:
            logger.warn("iter", i, "index", k, "is None")
            fail = True

if fail:
    import sys
    sys.exit(1)
