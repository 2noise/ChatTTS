import os
import logging
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional, List, Callable, Tuple, Dict
from json import load
from pathlib import Path
import lzma

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from vocos import Vocos
from huggingface_hub import snapshot_download
import pybase16384 as b14

from .model import DVAE, GPT, gen_logits
from .utils import (
    check_all_assets,
    download_all_assets,
    select_device,
    get_latest_modified_file,
    del_all,
)
from .utils import logger as utils_logger

from .norm import Normalizer


class Chat:
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        utils_logger.set_logger(logger)

        self.pretrain_models = {}
        self.normalizer = Normalizer(
            os.path.join(os.path.dirname(__file__), "res", "homophones_map.json"),
            logger,
        )
        with open(
            os.path.join(os.path.dirname(__file__), "res", "sha256_map.json")
        ) as f:
            self.sha256_map: Dict[str, str] = load(f)

        self.context = GPT.Context()

    def has_loaded(self, use_decoder=False):
        not_finish = False
        check_list = ["vocos", "_vocos_decode", "gpt", "tokenizer"]

        if use_decoder:
            check_list.append("decoder")
        else:
            check_list.append("dvae")

        for module in check_list:
            if not hasattr(self, module) and module not in self.pretrain_models:
                self.logger.warning(f"{module} not initialized.")
                not_finish = True

        if not not_finish:
            self.logger.info("all models has been initialized.")

        return not not_finish

    def download_models(
        self,
        source: Literal["huggingface", "local", "custom"] = "local",
        force_redownload=False,
        custom_path: Optional[torch.serialization.FILE_LIKE] = None,
    ) -> Optional[str]:
        if source == "local":
            download_path = os.getcwd()
            if not check_all_assets(Path(download_path), self.sha256_map, update=True) or force_redownload:
                with tempfile.TemporaryDirectory() as tmp:
                    download_all_assets(tmpdir=tmp)
                if not check_all_assets(Path(download_path), self.sha256_map, update=False):
                    self.logger.error(
                        "download to local path %s failed.", download_path
                    )
                    return None
        elif source == "huggingface":
            hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            try:
                download_path = get_latest_modified_file(
                    os.path.join(hf_home, "hub/models--2Noise--ChatTTS/snapshots")
                )
            except:
                download_path = None
            if download_path is None or force_redownload:
                self.logger.log(
                    logging.INFO,
                    f"download from HF: https://huggingface.co/2Noise/ChatTTS",
                )
                try:
                    download_path = snapshot_download(
                        repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"]
                    )
                except:
                    download_path = None
            else:
                self.logger.log(
                    logging.INFO, f"load latest snapshot from cache: {download_path}"
                )
            if download_path is None:
                self.logger.error("download from huggingface failed.")
                return None
        elif source == "custom":
            self.logger.log(logging.INFO, f"try to load from local: {custom_path}")
            if not check_all_assets(Path(custom_path), self.sha256_map, update=False):
                self.logger.error(
                    "check models in custom path %s failed.", custom_path
                )
                return None
            download_path = custom_path

        return download_path

    def load(
        self,
        source: Literal["huggingface", "local", "custom"] = "local",
        force_redownload=False,
        compile: bool = True,
        custom_path: Optional[torch.serialization.FILE_LIKE] = None,
        device: Optional[torch.device] = None,
        coef: Optional[torch.Tensor] = None,
    ) -> bool:
        download_path = self.download_models(source, force_redownload, custom_path)
        if download_path is None:
            return False
        return self._load(
            device=device,
            compile=compile,
            coef=coef,
            **{
                k: os.path.join(download_path, v)
                for k, v in OmegaConf.load(
                    os.path.join(download_path, "config", "path.yaml")
                ).items()
            },
        )

    def unload(self):
        logger = self.logger
        del_all(self.pretrain_models)
        self.normalizer.destroy()
        del self.normalizer
        del self.sha256_map
        del_list = ["vocos", "_vocos_decode", "gpt", "decoder", "dvae"]
        for module in del_list:
            if hasattr(self, module):
                delattr(self, module)
        self.__init__(logger)

    def sample_random_speaker(self) -> str:
        with torch.no_grad():
            spk = self._sample_random_speaker()
            arr: np.ndarray = spk.cpu().numpy()
            s = b14.encode_to_string(
                lzma.compress(
                    arr.tobytes(),
                    format=lzma.FORMAT_RAW,
                    filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
                ),
            )
            del arr, spk
        return s

    def _sample_random_speaker(self) -> torch.Tensor:
        with torch.no_grad():
            dim: int = self.gpt.gpt.layers[0].mlp.gate_proj.in_features
            out: torch.Tensor = self.pretrain_models["spk_stat"]
            std, mean = out.chunk(2)
            spk = torch.randn(dim, device=std.device, dtype=torch.float16).mul_(std).add_(mean)
            del out, std, mean
            return spk

    @dataclass(repr=False, eq=False)
    class RefineTextParams:
        prompt: str = ""
        top_P: float = 0.7
        top_K: int = 20
        temperature: float = 0.7
        repetition_penalty: float = 1.0
        max_new_token: int = 384
        min_new_token: int = 0

    @dataclass(repr=False, eq=False)
    class InferCodeParams:
        prompt: str = "[speed_5]"
        spk_emb: Optional[str] = None
        top_P: float = 0.7
        top_K: int = 20
        temperature: float = 0.3
        repetition_penalty: float = 1.05
        max_new_token: int = 2048
        min_new_token: int = 0

    def infer(
        self,
        text,
        stream=False,
        lang=None,
        skip_refine_text=False,
        refine_text_only=False,
        use_decoder=True,
        do_text_normalization=True,
        do_homophone_replacement=True,
        params_refine_text=RefineTextParams(),
        params_infer_code=InferCodeParams(),
    ):
        self.context.set(False)
        res_gen = self._infer(
            text,
            stream,
            lang,
            skip_refine_text,
            refine_text_only,
            use_decoder,
            do_text_normalization,
            do_homophone_replacement,
            params_refine_text,
            params_infer_code,
        )
        if stream:
            return res_gen
        else:
            return next(res_gen)

    def interrupt(self):
        self.context.set(True)

    def _load(
        self,
        vocos_config_path: str = None,
        vocos_ckpt_path: str = None,
        dvae_config_path: str = None,
        dvae_ckpt_path: str = None,
        gpt_config_path: str = None,
        gpt_ckpt_path: str = None,
        decoder_config_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        device: Optional[torch.device] = None,
        compile: bool = True,
        coef: Optional[str] = None,
    ):
        if device is None:
            device = select_device()
            self.logger.log(logging.INFO, f"use {device}")
        self.device = device

        if vocos_config_path:
            vocos = (
                Vocos.from_hparams(vocos_config_path)
                .to(
                    # vocos on mps will crash, use cpu fallback
                    "cpu"
                    if "mps" in str(device)
                    else device
                )
                .eval()
            )
            assert vocos_ckpt_path, "vocos_ckpt_path should not be None"
            vocos.load_state_dict(
                torch.load(vocos_ckpt_path, weights_only=True, mmap=True)
            )
            self.vocos = vocos
            if "mps" in str(self.device):
                self._vocos_decode: Callable[[torch.Tensor], np.ndarray] = (
                    lambda spec: self.vocos.decode(spec.cpu()).cpu().numpy()
                )
            else:
                self._vocos_decode: Callable[[torch.Tensor], np.ndarray] = (
                    lambda spec: self.vocos.decode(spec).cpu().numpy()
                )
            self.logger.log(logging.INFO, "vocos loaded.")

        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg, coef=coef).to(device).eval()
            coef = str(dvae)
            assert dvae_ckpt_path, "dvae_ckpt_path should not be None"
            dvae.load_state_dict(
                torch.load(dvae_ckpt_path, weights_only=True, mmap=True)
            )
            self.dvae = dvae
            self.logger.log(logging.INFO, "dvae loaded.")

        if gpt_config_path:
            cfg = OmegaConf.load(gpt_config_path)
            gpt = GPT(**cfg, device=device, logger=self.logger).eval()
            assert gpt_ckpt_path, "gpt_ckpt_path should not be None"
            gpt.load_state_dict(torch.load(gpt_ckpt_path, weights_only=True, mmap=True))
            if compile and "cuda" in str(device):
                try:
                    gpt.gpt.forward = torch.compile(
                        gpt.gpt.forward, backend="inductor", dynamic=True
                    )
                except RuntimeError as e:
                    self.logger.warning(
                        f"compile failed: {e}. fallback to normal mode."
                    )
            self.gpt = gpt
            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), "spk_stat.pt")
            assert os.path.exists(
                spk_stat_path
            ), f"Missing spk_stat.pt: {spk_stat_path}"
            self.pretrain_models["spk_stat"] = torch.load(
                spk_stat_path, weights_only=True, mmap=True
            ).to(device)
            self.logger.log(logging.INFO, "gpt loaded.")

        if decoder_config_path:
            cfg = OmegaConf.load(decoder_config_path)
            decoder = DVAE(**cfg, coef=coef).to(device).eval()
            coef = str(decoder)
            assert decoder_ckpt_path, "decoder_ckpt_path should not be None"
            decoder.load_state_dict(
                torch.load(decoder_ckpt_path, weights_only=True, mmap=True)
            )
            self.decoder = decoder
            self.logger.log(logging.INFO, "decoder loaded.")

        if tokenizer_path:
            tokenizer = torch.load(tokenizer_path, map_location=device, mmap=True)
            tokenizer.padding_side = "left"
            self.pretrain_models["tokenizer"] = tokenizer
            self.tokenizer_len = len(tokenizer)
            self.tokenizer_spk_emb_ids: torch.Tensor = tokenizer.convert_tokens_to_ids("[spk_emb]")
            self.tokenizer_break_0_ids: torch.Tensor = tokenizer.convert_tokens_to_ids("[break_0]")
            self.tokenizer_eos_token: torch.Tensor = torch.tensor(
                tokenizer.convert_tokens_to_ids("[Ebreak]"), device=gpt.device_gpt
            ).unsqueeze_(0)
            self.logger.log(logging.INFO, "tokenizer loaded.")

        self.coef = coef

        return self.has_loaded()

    def _infer(
        self,
        text,
        stream=False,
        lang=None,
        skip_refine_text=False,
        refine_text_only=False,
        use_decoder=True,
        do_text_normalization=True,
        do_homophone_replacement=True,
        params_refine_text=RefineTextParams(),
        params_infer_code=InferCodeParams(),
    ):

        assert self.has_loaded(use_decoder=use_decoder)

        if not isinstance(text, list):
            text = [text]

        text = [
            self.normalizer(
                t,
                do_text_normalization,
                do_homophone_replacement,
                lang,
            )
            for t in text
        ]

        with torch.no_grad():

            if not skip_refine_text:
                refined = self._refine_text(
                    text,
                    self.device,
                    params_refine_text,
                )
                text_tokens = refined.ids
                text_tokens = [
                    i[i.less(self.tokenizer_break_0_ids)]
                    for i in text_tokens
                ]
                text = self.pretrain_models["tokenizer"].batch_decode(text_tokens)
                refined.destroy()
                if refine_text_only:
                    yield text
                    return

            length = [0 for _ in range(len(text))]
            for result in self._infer_code(
                text,
                stream,
                self.device,
                use_decoder,
                params_infer_code,
            ):
                wav = self._decode_to_wavs(result, length, use_decoder)
                yield wav

    def _decode_to_wavs(
        self, result: GPT.GenerationOutputs, start_seeks: List[int], use_decoder: bool
    ):
        x = result.hiddens if use_decoder else result.ids
        wavs: List[np.ndarray] = []
        for i, chunk_data in enumerate(x):
            start_seek = start_seeks[i]
            length = len(chunk_data)
            if length <= start_seek:
                wavs.append(None)
                continue
            start_seeks[i] = length
            chunk_data = chunk_data[start_seek:]
            decoder = self.decoder if use_decoder else self.dvae
            mel_spec = decoder(chunk_data[None].permute(0, 2, 1).to(self.device))
            del chunk_data
            wavs.append(self._vocos_decode(mel_spec))
            del_all(mel_spec)
        result.destroy()
        del_all(x)
        return wavs

    def _text_to_token(self, text: str, device="cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        gpt = self.gpt
        tokenizer = self.pretrain_models["tokenizer"]

        text_token_tmp = tokenizer(
            text, return_tensors="pt", add_special_tokens=False, padding=True
        )
        text_token = text_token_tmp.to(device)
        del text_token_tmp

        input_ids = text_token["input_ids"].unsqueeze(-1).expand(-1, -1, gpt.num_vq)
        text_mask = torch.ones(text_token["input_ids"].shape, dtype=bool, device=device)
        attention_mask = text_token["attention_mask"]

        del_all(text_token)

        return input_ids, attention_mask, text_mask
    
    @staticmethod
    def _decode_spk_emb(spk_emb: str) -> np.ndarray:
        return np.frombuffer(lzma.decompress(
            b14.decode_from_string(spk_emb),
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
        ), dtype=np.float16).copy()

    def _apply_spk_emb(
        self,
        emb: torch.Tensor,
        spk_emb: str,
        input_ids: torch.Tensor,
        text_len: int,
    ):
        n = F.normalize(
            torch.from_numpy(
                self._decode_spk_emb(spk_emb),
            ).unsqueeze(0).expand(text_len, -1), p=2.0, dim=1, eps=1e-12
        ).to(self.gpt.device_gpt).unsqueeze_(1).expand(emb.shape)
        cond = input_ids.narrow(-1, 0, 1).eq(self.tokenizer_spk_emb_ids).expand(emb.shape)
        torch.where(cond, n, emb, out=emb)
        del cond, n

    def _infer_code(
        self,
        text: Tuple[List[str], str],
        stream: bool,
        device: torch.device,
        return_hidden: bool,
        params: InferCodeParams,
    ):

        gpt = self.gpt

        if not isinstance(text, list):
            text = [text]

        assert len(text), "text should not be empty"

        if not isinstance(params.temperature, list):
            temperature = [params.temperature] * gpt.num_vq
        else:
            temperature = params.temperature
        
        for i, t in enumerate(text):
            text[i] = t.replace('[Stts]', '').replace('[spk_emb]', '').replace('[empty_spk]', '').strip()
            """
            see https://github.com/2noise/ChatTTS/issues/459
            """

        if params.prompt:
            text = [params.prompt + i for i in text]

        if params.spk_emb is not None:
            text = [f"[Stts][spk_emb]{i}[Ptts]" for i in text]
        else:
            text = [f"[Stts][empty_spk]{i}[Ptts]" for i in text]

        input_ids, attention_mask, text_mask = self._text_to_token(text, gpt.device_gpt)

        emb = gpt(input_ids, text_mask)
        del text_mask

        if params.spk_emb is not None:
            self._apply_spk_emb(emb, params.spk_emb, input_ids, len(text))

        num_code = int(gpt.emb_code[0].num_embeddings - 1)

        logits_warpers, logits_processors = gen_logits(
            num_code=num_code,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        result = gpt.generate(
            emb,
            input_ids,
            temperature=torch.tensor(temperature, device=device),
            eos_token=num_code,
            attention_mask=attention_mask,
            max_new_token=params.max_new_token,
            min_new_token=params.min_new_token,
            logits_warpers=logits_warpers,
            logits_processors=logits_processors,
            infer_text=False,
            return_hidden=return_hidden,
            stream=stream,
            context=self.context,
        )

        del emb, input_ids
        del_all(logits_warpers)
        del_all(logits_processors)

        return result

    def _refine_text(
        self,
        text: str,
        device: torch.device,
        params: RefineTextParams,
    ):

        gpt = self.gpt

        if not isinstance(text, list):
            text = [text]

        text = [f"[Sbreak]{i}[Pbreak]{params.prompt}" for i in text]

        input_ids, attention_mask, text_mask = self._text_to_token(text, gpt.device_gpt)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.tokenizer_len,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        emb = gpt(input_ids, text_mask)
        del text_mask

        result = next(gpt.generate(
            emb,
            input_ids,
            temperature=torch.tensor([params.temperature], device=device),
            eos_token=self.tokenizer_eos_token,
            attention_mask=attention_mask,
            max_new_token=params.max_new_token,
            min_new_token=params.min_new_token,
            logits_warpers=logits_warpers,
            logits_processors=logits_processors,
            infer_text=True,
            stream=False,
            context=self.context,
        ))

        del emb, input_ids
        del_all(logits_warpers)
        del_all(logits_processors)

        return result
