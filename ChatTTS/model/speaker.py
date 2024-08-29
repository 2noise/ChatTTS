import lzma
from typing import List, Optional, Union

import pybase16384 as b14
import numpy as np
import torch
import torch.nn.functional as F


class Speaker:
    def __init__(self, dim: int, spk_cfg: str, device=torch.device("cpu")) -> None:
        spk_stat = torch.from_numpy(
            np.frombuffer(b14.decode_from_string(spk_cfg), dtype=np.float16).copy()
        ).to(device=device)
        self.std, self.mean = spk_stat.requires_grad_(False).chunk(2)
        self.dim = dim

    def sample_random(self) -> str:
        return self._encode(self._sample_random())

    @torch.inference_mode()
    def apply(
        self,
        emb: torch.Tensor,
        spk_emb: Union[str, torch.Tensor],
        input_ids: torch.Tensor,
        spk_emb_ids: int,
        device: torch.device,
        inplace: bool = True,
    ) -> torch.Tensor:
        if isinstance(spk_emb, str):
            spk_emb_tensor = torch.from_numpy(self._decode(spk_emb))
        else:
            spk_emb_tensor = spk_emb
        n = (
            F.normalize(
                spk_emb_tensor,
                p=2.0,
                dim=0,
                eps=1e-12,
            )
            .to(device)
            .unsqueeze_(0)
            .expand(emb.size(0), -1)
            .unsqueeze_(1)
            .expand(emb.shape)
        )
        cond = input_ids.narrow(-1, 0, 1).eq(spk_emb_ids).expand(emb.shape)
        out = torch.where(cond, n, emb, out=emb if inplace else None)
        if inplace:
            del cond, n
        return out

    @staticmethod
    @torch.no_grad()
    def decorate_code_prompts(
        text: List[str],
        prompt: str,
        txt_smp: Optional[str],
        spk_emb: Optional[str],
    ) -> List[str]:
        for i, t in enumerate(text):
            text[i] = (
                t.replace("[Stts]", "")
                .replace("[spk_emb]", "")
                .replace("[empty_spk]", "")
                .strip()
            )
            """
            see https://github.com/2noise/ChatTTS/issues/459
            """

        if prompt:
            text = [prompt + i for i in text]

        txt_smp = "" if txt_smp is None else txt_smp
        if spk_emb is not None:
            text = [f"[Stts][spk_emb]{txt_smp}{i}[Ptts]" for i in text]
        else:
            text = [f"[Stts][empty_spk]{txt_smp}{i}[Ptts]" for i in text]

        return text

    @staticmethod
    @torch.no_grad()
    def decorate_text_prompts(text: List[str], prompt: str) -> List[str]:
        return [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]

    @staticmethod
    @torch.no_grad()
    def encode_prompt(prompt: torch.Tensor) -> str:
        arr: np.ndarray = prompt.cpu().numpy().astype(np.uint16)
        shp = arr.shape
        assert len(shp) == 2, "prompt must be a 2D tensor"
        s = b14.encode_to_string(
            np.array(shp, dtype="<u2").tobytes()
            + lzma.compress(
                arr.astype("<u2").tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
        )
        del arr
        return s

    @staticmethod
    @torch.no_grad()
    def decode_prompt(prompt: str) -> torch.Tensor:
        dec = b14.decode_from_string(prompt)
        shp = np.frombuffer(dec[:4], dtype="<u2")
        p = np.frombuffer(
            lzma.decompress(
                dec[4:],
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype="<u2",
        ).copy()
        del dec
        return torch.from_numpy(p.astype(np.int32)).view(*shp)

    @torch.no_grad()
    def _sample_random(self) -> torch.Tensor:
        spk = (
            torch.randn(self.dim, device=self.std.device, dtype=self.std.dtype)
            .mul_(self.std)
            .add_(self.mean)
        )
        return spk

    @staticmethod
    @torch.no_grad()
    def _encode(spk_emb: torch.Tensor) -> str:
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
        )
        del arr
        return s

    @staticmethod
    def _decode(spk_emb: str) -> np.ndarray:
        return np.frombuffer(
            lzma.decompress(
                b14.decode_from_string(spk_emb),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype=np.float16,
        ).copy()
