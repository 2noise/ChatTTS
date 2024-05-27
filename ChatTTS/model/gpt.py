import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm
from einops import rearrange
from transformers.cache_utils import Cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaModel, LlamaConfig
    
    
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
    
class GPT_warpper(nn.Module):
    def __init__(
        self, 
        gpt_config, 
        num_audio_tokens,
        num_text_tokens,
        num_vq=4,
        **kwargs,
        ):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.gpt = self.build_model(gpt_config)
        self.model_dim = self.gpt.config.hidden_size 

        self.num_vq = num_vq
        self.emb_code = nn.ModuleList([nn.Embedding(num_audio_tokens, self.model_dim) for i in range(self.num_vq)])
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)
        self.head_text = weight_norm(nn.Linear(self.model_dim, num_text_tokens, bias=False), name='weight')
        self.head_code = nn.ModuleList([weight_norm(nn.Linear(self.model_dim, num_audio_tokens, bias=False), name='weight') for i in range(self.num_vq)])

    def build_model(self, config):
        
        configuration = LlamaConfig(**config)
        model = LlamaModel(configuration)
        del model.embed_tokens
        
        return model
    
    def get_emb(self, input_ids, text_mask, **kwargs):

        emb_text = self.emb_text(input_ids[text_mask][:, 0])
        
        emb_code = [self.emb_code[i](input_ids[~text_mask][:, i]) for i in range(self.num_vq)]
        emb_code = torch.stack(emb_code, 2).sum(2)
        
        emb = torch.zeros((input_ids.shape[:-1])+(emb_text.shape[-1],), device=emb_text.device, dtype=emb_text.dtype)
        emb[text_mask] = emb_text
        emb[~text_mask] = emb_code.to(emb.dtype)
        
        return emb
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(self.gpt.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def generate(
        self, 
        emb, 
        inputs_ids, 
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 2048, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        infer_text=False,
        return_attn=False,
        return_hidden=False,
    ):
        
        with torch.no_grad():   
        
            attentions = []
            hiddens = []
            
            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            temperature = temperature[None].expand(inputs_ids.shape[0], -1)
            temperature = rearrange(temperature, "b n -> (b n) 1")

            attention_mask_cache = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]+max_new_token,), dtype=torch.bool, device=inputs_ids.device)
            if attention_mask is not None:
                attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask
            
            for i in tqdm(range(max_new_token)):
        
                model_input = self.prepare_inputs_for_generation(inputs_ids, 
                    outputs.past_key_values if i!=0 else None, 
                    attention_mask_cache[:, :inputs_ids.shape[1]], use_cache=True)
            
                if i == 0:
                    model_input['inputs_embeds'] = emb
                else:
                    if infer_text:
                        model_input['inputs_embeds'] = self.emb_text(model_input['input_ids'][:,:,0])
                    else:
                        code_emb = [self.emb_code[i](model_input['input_ids'][:,:,i]) for i in range(self.num_vq)]
                        model_input['inputs_embeds'] = torch.stack(code_emb, 3).sum(3)
                
                model_input['input_ids'] = None
                outputs = self.gpt.forward(**model_input, output_attentions=return_attn)
                attentions.append(outputs.attentions)
                hidden_states = outputs[0] # 🐻
                if return_hidden:
                    hiddens.append(hidden_states[:, -1])

                with P.cached():
                    if infer_text:
                        logits = self.head_text(hidden_states) 
                    else:
                        logits = torch.stack([self.head_code[i](hidden_states) for i in range(self.num_vq)], 3)
        
                logits = logits[:, -1].float()

                if not infer_text:
                    logits = rearrange(logits, "b c n -> (b n) c")
                    logits_token = rearrange(inputs_ids[:, start_idx:], "b c n -> (b n) c")
                else:
                    logits_token = inputs_ids[:, start_idx:, 0]
                    
                logits = logits / temperature
                
                for logitsProcessors in LogitsProcessors:
                    logits = logitsProcessors(logits_token, logits)
                    
                for logitsWarpers in LogitsWarpers:
                    logits = logitsWarpers(logits_token, logits)
                    
                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf
                
                scores = F.softmax(logits, dim=-1)
            
                idx_next = torch.multinomial(scores, num_samples=1)
                
                if not infer_text:
                    idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                    finish = finish | (idx_next == eos_token).any(1)
                    inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)
                else:
                    finish = finish | (idx_next == eos_token).any(1)
                    inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(-1).expand(-1, -1, self.num_vq)], 1)

                end_idx = end_idx + (~finish).int()
            
                if finish.all():
                    break
            
            inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            inputs_ids = [i[:, 0] for i in inputs_ids] if infer_text else inputs_ids
            
            if return_hidden:
                hiddens = torch.stack(hiddens, 1)
                hiddens = [hiddens[idx, :i] for idx, i in enumerate(end_idx.int())]
                    
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
                   
            return {
                'ids': inputs_ids, 
                'attentions': attentions,
                'hiddens':hiddens,
            }