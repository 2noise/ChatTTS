
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

from ..utils.infer import CustomRepetitionPenaltyLogitsProcessorRepeat
from ..utils.io import del_all
from ..model.gpt import GPT

def infer_code(
    models,
    text, 
    spk_emb = None,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.3, 
    repetition_penalty = 1.05,
    max_new_token = 2048,
    stream=False,
    device="cpu",
    **kwargs
):

    gpt: GPT = models['gpt']

    if not isinstance(text, list): 
        text = [text]
        
    if not isinstance(temperature, list):
        temperature = [temperature] * gpt.num_vq
    
    if spk_emb is not None:
        text = [f'[Stts][spk_emb]{i}[Ptts]' for i in text] 
    else:
        text = [f'[Stts][empty_spk]{i}[Ptts]' for i in text]
    
    text_token_tmp = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True)
    text_token = text_token_tmp.to(device)
    del text_token_tmp
    input_ids = text_token['input_ids'][...,None].expand(-1, -1, gpt.num_vq).to(gpt.device_gpt)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=gpt.device_gpt)

    emb = gpt(input_ids, text_mask)
    del text_mask

    if spk_emb is not None:
        n = F.normalize(spk_emb.to(emb.dtype)[None].expand(len(text), -1), p=2.0, dim=1, eps=1e-12).to(gpt.device_gpt)
        emb[input_ids[..., 0] == models['tokenizer'].convert_tokens_to_ids('[spk_emb]')] = n
        del n

    num_code = int(gpt.emb_code[0].num_embeddings - 1)

    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(\
            repetition_penalty, num_code, 16))
    
    result = gpt.generate(
        emb, input_ids, 
        temperature = torch.tensor(temperature, device=device), 
        attention_mask = text_token['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = num_code, 
        max_new_token = max_new_token, 
        infer_text = False,
        stream = stream,
        **kwargs
    )

    del_all(text_token)
    del emb, text_token, input_ids
    del_all(LogitsWarpers)
    del_all(LogitsProcessors)

    return result


def refine_text(
    models, 
    text,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.7, 
    repetition_penalty = 1.0,
    max_new_token = 384,
    prompt = '',
    device="cpu",
    **kwargs
):

    gpt: GPT = models['gpt']

    if not isinstance(text, list): 
        text = [text]
    
    assert len(text), 'text should not be empty'

    text = [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)

    input_ids = text_token['input_ids'][...,None].expand(-1, -1, gpt.num_vq)
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, len(models['tokenizer']), 16))

    emb = gpt(input_ids,text_mask)
    del text_mask

    result = gpt.generate(
        emb, input_ids, 
        temperature = torch.tensor([temperature,], device=device), 
        attention_mask = text_token['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = torch.tensor(models['tokenizer'].convert_tokens_to_ids('[Ebreak]'), device=device)[None], 
        max_new_token = max_new_token, 
        infer_text = True,
        stream = False,
        **kwargs
    )

    del_all(text_token)
    del emb, text_token, input_ids
    del_all(LogitsWarpers)
    del_all(LogitsProcessors)

    return next(result)
