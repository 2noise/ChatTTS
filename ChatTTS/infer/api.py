
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ..utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat

def infer_code(
    models,
    text, 
    spk_emb = None,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.3, 
    repetition_penalty = 1.05,
    max_new_token = 2048,
    **kwargs
):
    
    device = next(models['gpt'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
        
    if not isinstance(temperature, list):
        temperature = [temperature] * models['gpt'].num_vq
    
    if spk_emb is not None:
        text = [f'[Stts][spk_emb]{i}[Ptts]' for i in text] 
    else:
        text = [f'[Stts][empty_spk]{i}[Ptts]' for i in text]
    
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    input_ids = text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)
    
    inputs = {
        'input_ids': input_ids,
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }

    emb = models['gpt'].get_emb(**inputs)
    if spk_emb is not None:
        emb[inputs['input_ids'][..., 0] == models['tokenizer'].convert_tokens_to_ids('[spk_emb]')] = \
            F.normalize(spk_emb.to(device).to(emb.dtype)[None].expand(len(text), -1), p=2.0, dim=1, eps=1e-12)  
    
    num_code = models['gpt'].emb_code[0].num_embeddings - 1
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(\
            repetition_penalty, num_code, 16))
    
    result = models['gpt'].generate(
        emb, inputs['input_ids'], 
        temperature = torch.tensor(temperature, device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = num_code, 
        max_new_token = max_new_token, 
        infer_text = False,
        **kwargs
    )
    
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
    **kwargs
):
    
    device = next(models['gpt'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
    
    assert len(text), 'text should not be empty'

    text = [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)

    inputs = {
        'input_ids': text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq),
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, len(models['tokenizer']), 16))
    
    result = models['gpt'].generate(
        models['gpt'].get_emb(**inputs), inputs['input_ids'], 
        temperature = torch.tensor([temperature,], device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = torch.tensor(models['tokenizer'].convert_tokens_to_ids('[Ebreak]'), device=device)[None], 
        max_new_token = max_new_token, 
        infer_text = True,
        **kwargs
    )
    return result