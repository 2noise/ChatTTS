from typing import List, Tuple

import torch
from transformers import BertTokenizerFast

from ..utils import del_all


class Tokenizer:
    def __init__(
        self, tokenizer_path: torch.serialization.FILE_LIKE, device: torch.device
    ):
        tokenizer: BertTokenizerFast = torch.load(
            tokenizer_path, map_location=device, mmap=True
        )
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

        self.len = len(tokenizer)
        self.spk_emb_ids = tokenizer.convert_tokens_to_ids("[spk_emb]")
        self.break_0_ids = tokenizer.convert_tokens_to_ids("[break_0]")
        self.eos_token = tokenizer.convert_tokens_to_ids("[Ebreak]")

        self.decode = self._tokenizer.batch_decode

    @torch.inference_mode()
    def encode(
        self, text: List[str], num_vq: int, device="cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_ids_lst = []
        attention_mask_lst = []
        max_input_ids_len = -1
        max_attention_mask_len = -1
        # avoid random speaker embedding of tokenizer in the other dims
        for t in text:
            x = self._tokenizer(
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
        input_ids = torch.zeros(
            len(input_ids_lst),
            max_input_ids_len,
            device=device,
            dtype=input_ids_lst[0].dtype,
        )
        for i in range(len(input_ids_lst)):
            input_ids.narrow(0, i, 1).narrow(1, 0, input_ids_lst[i].size(0)).copy_(
                input_ids_lst[i]
            )
        del_all(input_ids_lst)
        attention_mask = torch.zeros(
            len(attention_mask_lst),
            max_attention_mask_len,
            device=device,
            dtype=attention_mask_lst[0].dtype,
        )
        for i in range(len(attention_mask_lst)):
            attention_mask.narrow(0, i, 1).narrow(
                1, 0, attention_mask_lst[i].size(0)
            ).copy_(attention_mask_lst[i])
        del_all(attention_mask_lst)

        text_mask = torch.ones(input_ids.shape, dtype=bool, device=device)
        input_ids = input_ids.unsqueeze_(-1).expand(-1, -1, num_vq)

        return input_ids, attention_mask, text_mask
