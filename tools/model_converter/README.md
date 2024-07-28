# Export onnx or JIT models for deployment

## Run `pip install onnx -U`.

## Export GPT

3. Run `python tools/model_converter/exporter.py --gpt`


## Export other models
Run `python tools/model_converter/exporter.py --decoder --vocos`

## Reference
[Run LLMs on Sophon TPU](https://github.com/sophgo/LLM-TPU)