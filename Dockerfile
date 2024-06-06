FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    vim \
    curl \
    ca-certificates \
    curl \
    git \
    bzip2 \
    libsndfile1 \
    gcc \
    cmake \
    make \
    g++ \
    vim \
    wget \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY tts_model /app/tts_model
# 避免跟官方仓库的 requirements 冲突
COPY ./requirements-chat-tts.txt /app/requirements.txt

RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY ./ChatTTS /app/ChatTTS 

COPY ./requirements-api-ui.txt /app/requirements-api-ui.txt
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple && \
    pip install --no-cache-dir -r /app/requirements-api-ui.txt

COPY ./api.py /app
COPY ./webui_api.py /app

ENV LOG_LEVEL=INFO
ENV LOG_FILE=/app/logs/service.log
ENV WORKERS=1
ENV PYTHONIOENCODING=UTF-8
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8
ENV API_PORT=8080

CMD streamlit run webui_api.py --browser.gatherUsageStats False & \
    gunicorn api:app -w ${WORKERS} -k uvicorn.workers.UvicornWorker -b :${API_PORT}
