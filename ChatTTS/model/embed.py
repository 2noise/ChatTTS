import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from ..utils import load_safetensors


class Embed(nn.Module):
    def __init__(
        self, hidden_size: int, num_audio_tokens: int, num_text_tokens: int, num_vq=4
    ):
        super().__init__()

        self.num_vq = num_vq
        self.num_audio_tokens = num_audio_tokens

        self.model_dim = hidden_size
        self.emb_code = nn.ModuleList(
            [nn.Embedding(num_audio_tokens, self.model_dim) for _ in range(num_vq)],
        )
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)

        self.head_text = weight_norm(
            nn.Linear(self.model_dim, num_text_tokens, bias=False),
            name="weight",
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(self.model_dim, num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(self.num_vq)
            ],
        )

    @torch.inference_mode()
    def load_pretrained(self, filename: str, device: torch.device):
        state_dict_tensors = load_safetensors(filename)
        self.load_state_dict(state_dict_tensors)
        self.to(device)

    def __call__(
        self, input_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        get_emb
        """
        return super().__call__(input_ids, text_mask)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        get_emb
        """
        device = next(self.parameters()).device
        emb_text: torch.Tensor = self.emb_text(
            input_ids[text_mask].narrow(1, 0, 1).squeeze_(1).to(device)
        )

        text_mask_inv = text_mask.logical_not().to(device)
        masked_input_ids: torch.Tensor = input_ids[text_mask_inv].to(device)

        emb_code = [
            self.emb_code[i](masked_input_ids[:, i]) for i in range(self.num_vq)
        ]
        emb_code = torch.stack(emb_code, 2).sum(2)

        emb = torch.zeros(
            (input_ids.shape[:-1]) + (emb_text.shape[-1],),
            device=emb_text.device,
            dtype=emb_text.dtype,
        )
        emb[text_mask] = emb_text
        emb[text_mask_inv] = emb_code.to(emb.dtype)

        del emb_text, emb_code, text_mask_inv

        return emb
