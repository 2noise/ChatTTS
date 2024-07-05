import torch
from transformers import BertTokenizerFast


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

        self.batch_encode = self._tokenizer.__call__
        self.batch_decode = self._tokenizer.batch_decode
