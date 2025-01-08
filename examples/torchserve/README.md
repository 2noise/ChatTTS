# Generating voice with ChatTTS via TorchServe for high-performance inference

## Why We Use TorchServe

TorchServe is designed to deliver high performance for serving PyTorch models, and it excels in the following key areas:

1. Batching Requests: TorchServe automatically batches incoming requests, processing multiple predictions in parallel. This reduces overhead, improves throughput, and ensures efficient use of resources, especially when dealing with large volumes of requests.

2. Horizontal Scaling: TorchServe allows for horizontal scaling, meaning it can easily scale across multiple machines or containers to handle increasing traffic. This ensures that the system remains responsive and can handle large volumes of inference requests without sacrificing performance.

## Install requirements

``` bash
pip install -r requirements.txt
```

## Download the model if needed

``` bash
huggingface-cli download 2Noise/ChatTTS
```

## Store the Model

Replace `/path/to/your/model` with the actual path to your model files, for example: `/home/username/.cache/huggingface/hub/models--2Noise--ChatTTS/snapshots/1a3cxxxx`

``` bash

torch-model-archiver --model-name chattts \
                     --version 1.0 \
                     --serialized-file /path/to/your/model/asset/Decoder.pt \
                     --handler model_handler.py \
                     --extra-files "/path/to/your/model" \
                     --export-path model_store
```

## Optional： TorchServe Model Configuration

TorchServe support batch inference which aggregates inference requests and sending this aggregated requests through the ML/DL framework for inference all at once. TorchServe was designed to natively support batching of incoming inference requests. This functionality enables you to use your host resources optimally, because most ML/DL frameworks are optimized for batch requests.

Started from Torchserve 0.4.1, there are two methods to configure TorchServe to use the batching feature:

The configuration properties that we are interested in are the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle, in this example we set `batch_size` to `32`

2. `max_batch_delay`: This is the maximum batch delay time in ms TorchServe waits to receive batch_size number of requests. If TorchServe doesn’t receive batch_size number of requests before this timer time’s out, it sends what ever requests that were received to the model handler, in this example we set `max_batch_delay` to `1000`

## Start TorchServe locally

``` bash
pip install ChatTTS nvgpu soundfile nemo_text_processing WeTextProcessing

torchserve --start --model-store model_store --models chattts=chattts.mar  --ts-config config.properties
```

## Optional: Start TorchServe with docker

### Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)

### 1. Build the docker image

```bash
docker build -t torchserve-chattts:latest-gpu .
```

### 2. Start the container

```bash
docker run --name torchserve-chattts-gpu --gpus all -p 8080:8080 -p 8081:8081 torchserve-chattts:latest-gpu
```

## Inference with restful api

Note that the `text` parameter takes a string instead of a list, TorchServe will automaticly batch multiple inferences into one request

``` bash
curl --location --request GET 'http://127.0.0.1:8080/predictions/chattts' \
--header 'Content-Type: application/json' \
--data '{
    "text": "今天天气不错哦",
    "stream": false,
    "temperature": 0.3,
    "lang": "zh",
    "skip_refine_text": true,
    "refine_text_only": false,
    "use_decoder": true,
    "do_text_normalization": true,
    "do_homophone_replacement": false,
    "params_infer_code": {
        "prompt": "",
        "top_P": 0.7,
        "top_K": 20,
        "temperature": 0.3,
        "repetition_penalty": 1.05,
        "max_new_token": 2048,
        "min_new_token": 0,
        "show_tqdm": true,
        "ensure_non_empty": true,
        "manual_seed": 888,
        "stream_batch": 24,
        "stream_speed": 12000,
        "pass_first_n_batches": 2
    }
}' --output test.wav
```
