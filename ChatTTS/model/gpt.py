import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
import logging
from typing import Union, List, Optional, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import LlamaModel, LlamaConfig, LogitsWarper
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..utils.infer import CustomRepetitionPenaltyLogitsProcessorRepeat
from ..utils.io import del_all


"""class LlamaMLP(nn.Module):
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
        return down_proj"""


class GPT(nn.Module):
    def __init__(
        self, 
        gpt_config: omegaconf.DictConfig, 
        num_audio_tokens: int,
        num_text_tokens: int,
        num_vq=4,
        device=torch.device("cpu"),
        logger=logging.getLogger(__name__)
    ):
        super().__init__()

        self.logger = logger
        self.device = device
        self.device_gpt = device if "mps" not in str(device) else torch.device("cpu")
        self.num_vq = num_vq
        self.num_audio_tokens = num_audio_tokens

        self.gpt = self._build_llama(gpt_config, self.device_gpt)      
        self.model_dim = int(self.gpt.config.hidden_size)
        self.emb_code = nn.ModuleList(
            [nn.Embedding(
                num_audio_tokens, self.model_dim, device=self.device_gpt,
            ) for _ in range(num_vq)],
        )
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim, device=self.device_gpt)

        self.head_text = weight_norm(
            nn.Linear(
                self.model_dim, num_text_tokens, bias=False, device=device,
            ),
            name='weight',
        )
        self.head_code = nn.ModuleList(
            [weight_norm(
                nn.Linear(
                    self.model_dim, num_audio_tokens, bias=False, device=device,
                ),
                name='weight',
            ) for _ in range(self.num_vq)],
        )

    def _build_llama(self, config: omegaconf.DictConfig, device: torch.device) -> LlamaModel:

        model = LlamaModel(LlamaConfig(**config))
        del model.embed_tokens

        return model.to(device)
    
    def __call__(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        get_emb
        """
        return super().__call__(input_ids, text_mask)

    def forward(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        get_emb
        """

        emb_text: torch.Tensor = self.emb_text(input_ids[text_mask].narrow(1, 0, 1).squeeze_(1).to(self.device_gpt))

        text_mask_inv = ~(text_mask.to(self.device_gpt))
        masked_input_ids: torch.Tensor = input_ids[text_mask_inv].to(self.device_gpt)

        emb_code = [self.emb_code[i](masked_input_ids[:, i]) for i in range(self.num_vq)]
        emb_code = torch.stack(emb_code, 2).sum(2)

        emb = torch.zeros((input_ids.shape[:-1])+(emb_text.shape[-1],), device=emb_text.device, dtype=emb_text.dtype)
        emb[text_mask] = emb_text
        emb[text_mask_inv] = emb_code.to(emb.dtype)

        del emb_text, emb_code, text_mask_inv

        return emb
    
    @dataclass(repr=False, eq=False)
    class _GenerationInputs():
        position_ids: torch.Tensor
        cache_position: torch.Tensor
        use_cache: bool
        input_ids: Optional[torch.Tensor]=None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None
        attention_mask: Optional[torch.Tensor]=None
        inputs_embeds: Optional[torch.Tensor]=None

        def to(self, device: torch.device):
            if self.attention_mask  is not None: self.attention_mask = self.attention_mask.to(device)
            if self.position_ids    is not None: self.position_ids   = self.position_ids.to(device)
            if self.inputs_embeds   is not None: self.inputs_embeds  = self.inputs_embeds.to(device)
            if self.cache_position  is not None: self.cache_position = self.cache_position.to(device)

    def _prepare_generation_inputs(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        cache_position: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.Tensor]=None,
        use_cache = True,
    ) -> _GenerationInputs:
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

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs = self._GenerationInputs(
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs.inputs_embeds = inputs_embeds
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs.input_ids = input_ids.contiguous()

        model_inputs.past_key_values = past_key_values
        model_inputs.attention_mask = attention_mask

        return model_inputs

    @dataclass(repr=False, eq=False)
    class GenerationOutputs():
        ids: List[torch.Tensor]
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]]
        hiddens: List[torch.Tensor]

        def destroy(self):
            del_all(self.ids)
            del_all(self.attentions)
            del_all(self.hiddens)


    def _prepare_generation_outputs(
        self,
        inputs_ids: torch.Tensor,
        start_idx: int,
        end_idx: torch.Tensor,
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]],
        hiddens: List[torch.Tensor],
        infer_text: bool,
    ) -> GenerationOutputs:
        inputs_ids = [inputs_ids[idx].narrow(0, start_idx, i) for idx, i in enumerate(end_idx)]
        if infer_text:
            inputs_ids = [i.narrow(1, 0, 1).squeeze_(1) for i in inputs_ids]

        if len(hiddens) > 0:
            hiddens = torch.stack(hiddens, 1)
            hiddens = [hiddens[idx].narrow(0, 0, i) for idx, i in enumerate(end_idx.int())]

        return self.GenerationOutputs(
            ids=inputs_ids,
            attentions=attentions,
            hiddens=hiddens,
        )


    def generate(
        self, 
        emb: torch.Tensor, 
        inputs_ids: torch.Tensor, 
        temperature: torch.Tensor, 
        eos_token: Union[int, torch.Tensor], 
        attention_mask = None,
        max_new_token = 2048, 
        min_new_token = 0,
        LogitsWarpers: List[LogitsWarper] = [],
        LogitsProcessors: List[CustomRepetitionPenaltyLogitsProcessorRepeat] = [],
        infer_text=False,
        return_attn=False,
        return_hidden=False,
        stream=False,
    ):
        
        with torch.no_grad():

            attentions: List[Optional[Tuple[torch.FloatTensor, ...]]] = []
            hiddens = []

            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            temperature = temperature.unsqueeze_(0).expand(inputs_ids.shape[0], -1).contiguous().view(-1, 1)
            # temperature = rearrange(temperature, "b n -> (b n) 1")

            attention_mask_cache = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]+max_new_token,), dtype=torch.bool, device=inputs_ids.device)
            if attention_mask is not None:
                attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask

            with tqdm(
                total=max_new_token,
                desc="text" if infer_text else "code",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]',
            ) as pbar:

                past_key_values = None

                for i in range(max_new_token):
                    model_input = self._prepare_generation_inputs(
                        inputs_ids, 
                        past_key_values, 
                        attention_mask_cache[:, :inputs_ids.shape[1]],
                        use_cache=True,
                    )

                    if i > 0:
                        del emb
                        inputs_ids_emb = model_input.input_ids.to(self.device_gpt)
                        if infer_text:
                            emb: torch.Tensor = self.emb_text(inputs_ids_emb[:,:,0])
                        else:
                            code_emb = [self.emb_code[i](inputs_ids_emb[:,:,i]) for i in range(self.num_vq)]
                            emb = torch.stack(code_emb, 3).sum(3)
                        del inputs_ids_emb, model_input.input_ids
                    model_input.inputs_embeds = emb

                    model_input.to(self.device_gpt)

                    outputs: BaseModelOutputWithPast = self.gpt(
                        attention_mask=model_input.attention_mask,
                        position_ids=model_input.position_ids,
                        past_key_values=model_input.past_key_values,
                        inputs_embeds=model_input.inputs_embeds,
                        use_cache=model_input.use_cache,
                        output_attentions=return_attn,
                        cache_position=model_input.cache_position,
                    )
                    del_all(model_input)
                    attentions.append(outputs.attentions)
                    hidden_states = outputs.last_hidden_state.to(self.device) # 🐻
                    past_key_values = outputs.past_key_values
                    del_all(outputs)
                    if return_hidden:
                        hiddens.append(hidden_states.narrow(1, -1, 1).squeeze_(1))

                    with P.cached():
                        if infer_text:
                            logits: torch.Tensor = self.head_text(hidden_states)
                        else:
                            # logits = torch.stack([self.head_code[i](hidden_states) for i in range(self.num_vq)], 3)
                            logits = torch.empty(
                                hidden_states.size(0), hidden_states.size(1),
                                self.num_audio_tokens, self.num_vq,
                                dtype=torch.float, device=self.device,
                            )
                            for i in range(self.num_vq):
                                x: torch.Tensor = self.head_code[i](hidden_states)
                                logits[..., i] = x
                                del x

                    # logits = logits[:, -1].float()
                    logits = logits.narrow(1, -1, 1).squeeze_(1).float()

                    if not infer_text:
                        # logits = rearrange(logits, "b c n -> (b n) c")
                        logits = logits.permute(0, 2, 1)
                        logits = logits.reshape(-1, logits.size(2))
                        # logits_token = rearrange(inputs_ids[:, start_idx:], "b c n -> (b n) c")
                        inputs_ids_sliced = inputs_ids[:, start_idx:].permute(0, 2, 1)
                        logits_token = inputs_ids_sliced.reshape(
                            inputs_ids_sliced.size(0)*inputs_ids_sliced.size(1), -1,
                        )
                    else:
                        logits_token = inputs_ids[:, start_idx:, 0]

                    logits /= temperature

                    for logitsProcessors in LogitsProcessors:
                        logits = logitsProcessors(logits_token, logits)

                    for logitsWarpers in LogitsWarpers:
                        logits = logitsWarpers(logits_token, logits)

                    del logits_token

                    if i < min_new_token:
                        logits[:, eos_token] = -torch.inf

                    scores = F.softmax(logits, dim=-1)

                    del logits

                    idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

                    if not infer_text:
                        # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                        idx_next = idx_next.view(-1, self.num_vq)
                        finish_or = (idx_next == eos_token).any(1)
                        finish |= finish_or
                        del finish_or
                        inputs_ids_tmp = torch.cat([inputs_ids, idx_next.unsqueeze_(1)], 1)
                    else:
                        finish_or = (idx_next == eos_token).any(1)
                        finish |= finish_or
                        del finish_or
                        inputs_ids_tmp = torch.cat([inputs_ids, idx_next.unsqueeze_(-1).expand(-1, -1, self.num_vq)], 1)

                    del inputs_ids
                    inputs_ids = inputs_ids_tmp
                    del inputs_ids_tmp, idx_next

                    if stream:
                        minus_prev_end_index = -end_idx
                    end_idx += (~finish.to(end_idx.device)).int()
                    if stream:
                        if end_idx.all() and (end_idx%24 == 0).any() and torch.add(end_idx, minus_prev_end_index, out=minus_prev_end_index).any():
                            self.logger.debug("yield stream result, end: %d", end_idx)
                            yield self._prepare_generation_outputs(
                                inputs_ids, start_idx, end_idx, attentions, hiddens,
                                infer_text,
                            )
                        del minus_prev_end_index

                    if finish.all(): break

                    pbar.update(1)

            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')

            del finish

            yield self._prepare_generation_outputs(
                inputs_ids, start_idx, end_idx, attentions, hiddens,
                infer_text,
            )
