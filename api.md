# API使用说明

API是使用FastAPI开发的，目前没考虑并发，在4090 没使用flash attn推理，速度在80左右。

# 更新说明
## 2024-5-31 1.0.0
```text
1. 支持长文本一次生成，支持流式返回。(目前不知道为什么速度会降低？)
2. 支持固定音色，支持将上次生成的音色保存下来，下次生成可以指定
```

# API 说明

## 获取音频
```http request
GET /?spk=可选1&text=你好
Content-Type: audio/wav
```
## 保存音色
```http request
POST /speaker
Content-Type: application/json
{
"name": "音色名"
}
```

# 环境要求
python: 3.9+1
torch: 2.2
transformers: 4.41


# 安装
```bash
pip install -r requirements.txt
# 需要自己提前下载 https://huggingface.co/2Noise/ChatTTS
python api.py --model-dir "本地权重路径"
```

# 使用
```
curl http://localhost:12456?text=你好 -o output.wav
```

# 已知问题
1. 文本过长会有问题，暂时不要超过100个字。
2. 不支持并发，请不要尝试
3. 采样率为24000，有一些软件用起来会有问题
4. 不能固定音色，等后续更新