import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from modeling_llama import LlamaModel, LlamaConfig


class GPT(nn.Module):
    def __init__(
        self,
        gpt_config: dict,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_vq=4,
        use_flash_attn=False,
        device=torch.device("cpu"),
        logger=logging.getLogger(__name__),
    ):
        super().__init__()

        self.logger = logger

        self.device = device
        self.device_gpt = device if "mps" not in str(device) else torch.device("cpu")

        self.num_vq = num_vq
        self.num_audio_tokens = num_audio_tokens

        self.use_flash_attn = use_flash_attn

        self.gpt, self.llama_config = self._build_llama(gpt_config, self.device_gpt)
        self.is_te_llama = False
        self.model_dim = int(self.gpt.config.hidden_size)
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(
                    num_audio_tokens,
                    self.model_dim,
                    device=self.device_gpt,
                )
                for _ in range(num_vq)
            ],
        )
        self.emb_text = nn.Embedding(
            num_text_tokens, self.model_dim, device=self.device_gpt
        )

        self.head_text = weight_norm(
            nn.Linear(
                self.model_dim,
                num_text_tokens,
                bias=False,
                device=device,
            ),
            name="weight",
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        self.model_dim,
                        num_audio_tokens,
                        bias=False,
                        device=device,
                    ),
                    name="weight",
                )
                for _ in range(self.num_vq)
            ],
        )

    def from_pretrained(self, file_path: str):
        self.load_state_dict(
            torch.load(file_path, weights_only=True, mmap=True), strict=False
        )

    def _build_llama(
        self,
        config: dict,
        device: torch.device,
    ) -> Tuple[LlamaModel, LlamaConfig]:

        llama_config = LlamaConfig(**config)
        model = LlamaModel(llama_config)
        del model.embed_tokens
        return model.to(device), llama_config
