import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
"""

from typing import List, Tuple, Optional, Union

import torch
from transformers import BertTokenizerFast

from ..utils import del_all


class Tokenizer:
    def __init__(
        self,
        tokenizer_path: torch.serialization.FILE_LIKE,
    ):
        """
        tokenizer: BertTokenizerFast = torch.load(
            tokenizer_path, map_location=device, mmap=True
        )
        # tokenizer.save_pretrained("asset/tokenizer", legacy_format=False)
        """
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_path)
        self._tokenizer = tokenizer

        self.len = len(tokenizer)
        self.spk_emb_ids = tokenizer.convert_tokens_to_ids("[spk_emb]")
        self.break_0_ids = tokenizer.convert_tokens_to_ids("[break_0]")
        self.eos_token = tokenizer.convert_tokens_to_ids("[Ebreak]")

    @torch.inference_mode()
    def encode(
        self,
        text: List[str],
        num_vq: int,
        prompt: Optional[torch.Tensor] = None,
        device="cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_ids_lst = []
        attention_mask_lst = []
        max_input_ids_len = -1
        max_attention_mask_len = -1
        prompt_size = 0

        if prompt is not None:
            assert prompt.size(0) == num_vq, "prompt dim 0 must equal to num_vq"
            prompt_size = prompt.size(1)

        # avoid random speaker embedding of tokenizer in the other dims
        for t in text:
            x = self._tokenizer.encode_plus(
                t, return_tensors="pt", add_special_tokens=False, padding=True
            )
            input_ids_lst.append(x["input_ids"].squeeze_(0))
            attention_mask_lst.append(x["attention_mask"].squeeze_(0))
            del_all(x)
            ids_sz = input_ids_lst[-1].size(0)
            if ids_sz > max_input_ids_len:
                max_input_ids_len = ids_sz
            attn_sz = attention_mask_lst[-1].size(0)
            if attn_sz > max_attention_mask_len:
                max_attention_mask_len = attn_sz

        if prompt is not None:
            max_input_ids_len += prompt_size
            max_attention_mask_len += prompt_size

        input_ids = torch.zeros(
            len(input_ids_lst),
            max_input_ids_len,
            device=device,
            dtype=input_ids_lst[0].dtype,
        )
        for i in range(len(input_ids_lst)):
            input_ids.narrow(0, i, 1).narrow(
                1,
                max_input_ids_len - prompt_size - input_ids_lst[i].size(0),
                input_ids_lst[i].size(0),
            ).copy_(
                input_ids_lst[i]
            )  # left padding
        del_all(input_ids_lst)

        attention_mask = torch.zeros(
            len(attention_mask_lst),
            max_attention_mask_len,
            device=device,
            dtype=attention_mask_lst[0].dtype,
        )
        for i in range(len(attention_mask_lst)):
            attn = attention_mask.narrow(0, i, 1)
            attn.narrow(
                1,
                max_attention_mask_len - prompt_size - attention_mask_lst[i].size(0),
                attention_mask_lst[i].size(0),
            ).copy_(
                attention_mask_lst[i]
            )  # left padding
            if prompt_size > 0:
                attn.narrow(
                    1,
                    max_attention_mask_len - prompt_size,
                    prompt_size,
                ).fill_(1)
        del_all(attention_mask_lst)

        text_mask = attention_mask.bool()
        new_input_ids = input_ids.unsqueeze_(-1).expand(-1, -1, num_vq).clone()
        del input_ids

        if prompt_size > 0:
            text_mask.narrow(1, max_input_ids_len - prompt_size, prompt_size).fill_(0)
            prompt_t = prompt.t().unsqueeze_(0).expand(new_input_ids.size(0), -1, -1)
            new_input_ids.narrow(
                1,
                max_input_ids_len - prompt_size,
                prompt_size,
            ).copy_(prompt_t)
            del prompt_t

        return new_input_ids, attention_mask, text_mask

    @torch.inference_mode
    def decode(
        self,
        sequences: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ):
        return self._tokenizer.batch_decode(
            sequences, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )
