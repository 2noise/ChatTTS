# Export onnx models for deployment

## Export GPT
1. Downgrade the transformers package to v4.32.
   - `pip install transformers==4.32`
   - remove `from transformers.cache_utils import Cache` in `./ChatTTS/model/gpt.py`
   
2. Replace `modeling_llama.py` with `./onnx_export/assets/modeling_llama.py`.
   Run `pip show transformers` to get the path to the transformers package, and run `cp ./onnx_export/assets/modeling_llama.py [path_to_transformers]/src/transformers/models/llama/modeling_llama.py`.

3. Run `trace_gpt.py` 

## Export other models
Run `trace_others.py` 

## Reference
[Run LLMs on Sophon TPU](https://github.com/sophgo/LLM-TPU)