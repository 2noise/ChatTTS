import datetime
import os
import zipfile
from io import BytesIO

import requests

chattts_service_host = os.environ.get("CHATTTS_SERVICE_HOST", "localhost")
chattts_service_port = os.environ.get("CHATTTS_SERVICE_PORT", "8000")

CHATTTS_URL = f"http://{chattts_service_host}:{chattts_service_port}/generate_voice"


# main infer params
body = {
    "text": [
        "四川美食确实以辣闻名，但也有不辣的选择。",
        "比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
    ],
    "stream": False,
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
params_refine_text = {
    "prompt": "",
    "top_P": 0.7,
    "top_K": 20,
    "temperature": 0.7,
    "repetition_penalty": 1,
    "max_new_token": 384,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": 24,
}
body["params_refine_text"] = params_refine_text

# infer code params
params_infer_code = {
    "prompt": "[speed_5]",
    "top_P": 0.1,
    "top_K": 20,
    "temperature": 0.3,
    "repetition_penalty": 1.05,
    "max_new_token": 2048,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": True,
    "spk_emb": None,
}
body["params_infer_code"] = params_infer_code


try:
    response = requests.post(CHATTTS_URL, json=body)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
        # save files for each request in a different folder
        dt = datetime.datetime.now()
        ts = int(dt.timestamp())
        tgt = f"./output/{ts}/"
        os.makedirs(tgt, 0o755)
        zip_ref.extractall(tgt)
        print("Extracted files into", tgt)

except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
