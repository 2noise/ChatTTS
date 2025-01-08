FROM pytorch/torchserve:latest-gpu

WORKDIR /app

RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

RUN pip install --upgrade pip \
    && pip install ChatTTS nvgpu soundfile nemo_text_processing WeTextProcessing --no-cache-dir  # 安装 ChatTTS 库


COPY ./model_store /app/model_store
COPY ./config.properties /app/config.properties

CMD ["torchserve", "--start", "--model-store", "/app/model_store", "--models", "chattts=chattts.mar", "--ts-config", "/app/config.properties"]
