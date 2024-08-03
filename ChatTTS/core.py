import os
import logging
import tempfile
from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Tuple, Dict, Union
from json import load
from pathlib import Path
import lzma

import numpy as np
import torch
from vocos import Vocos
from vocos.pretrained import instantiate_class
from huggingface_hub import snapshot_download
import pybase16384 as b14

from .config import Config
from .model import DVAE, GPT, gen_logits, Tokenizer
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

        self.config = Config()

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
        check_list = ["vocos", "gpt", "tokenizer"]

        if use_decoder:
            check_list.append("decoder")
        else:
            check_list.append("dvae")

        for module in check_list:
            if not hasattr(self, module):
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
            if (
                not check_all_assets(Path(download_path), self.sha256_map, update=True)
                or force_redownload
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    download_all_assets(tmpdir=tmp)
                if not check_all_assets(
                    Path(download_path), self.sha256_map, update=False
                ):
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
                self.logger.error("check models in custom path %s failed.", custom_path)
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
        use_flash_attn=False,
    ) -> bool:
        download_path = self.download_models(source, force_redownload, custom_path)
        if download_path is None:
            return False
        return self._load(
            device=device,
            compile=compile,
            coef=coef,
            use_flash_attn=use_flash_attn,
            **{
                k: os.path.join(download_path, v)
                for k, v in asdict(self.config.path).items()
            },
        )

    def unload(self):
        logger = self.logger
        self.normalizer.destroy()
        del self.normalizer
        del self.sha256_map
        del_list = ["vocos", "gpt", "decoder", "dvae", "tokenizer"]
        for module in del_list:
            if hasattr(self, module):
                delattr(self, module)
        self.__init__(logger)

    def sample_random_speaker(self) -> str:
        return self.tokenizer._encode_spk_emb(self._sample_random_speaker())

    @torch.inference_mode()
    def sample_audio_speaker(self, wav: Union[np.ndarray, torch.Tensor]) -> str:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).to(self.device)
        return self.tokenizer._encode_prompt(self.dvae(wav, "encode").squeeze_(0))

    @torch.no_grad()
    def _sample_random_speaker(self) -> torch.Tensor:
        dim: int = self.gpt.gpt.layers[0].mlp.gate_proj.in_features
        spk = (
            torch.randn(dim, device=self.std.device, dtype=self.std.dtype)
            .mul_(self.std)
            .add_(self.mean)
        )
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
        show_tqdm: bool = True
        ensure_non_empty: bool = True

    @dataclass(repr=False, eq=False)
    class InferCodeParams(RefineTextParams):
        prompt: str = "[speed_5]"
        spk_emb: Optional[str] = None
        spk_smp: Optional[str] = None
        txt_smp: Optional[str] = None
        temperature: float = 0.3
        repetition_penalty: float = 1.05
        max_new_token: int = 2048
        stream_batch: int = 24
        stream_speed: int = 12000
        pass_first_n_batches: int = 2

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

    @torch.no_grad()
    def _load(
        self,
        vocos_ckpt_path: str = None,
        dvae_ckpt_path: str = None,
        gpt_ckpt_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        device: Optional[torch.device] = None,
        compile: bool = True,
        coef: Optional[str] = None,
        use_flash_attn=False,
    ):
        if device is None:
            device = select_device()
            self.logger.info("use device %s", str(device))
        self.device = device
        self.compile = compile

        feature_extractor = instantiate_class(
            args=(), init=asdict(self.config.vocos.feature_extractor)
        )
        backbone = instantiate_class(args=(), init=asdict(self.config.vocos.backbone))
        head = instantiate_class(args=(), init=asdict(self.config.vocos.head))
        vocos = (
            Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
            .to(
                # vocos on mps will crash, use cpu fallback
                "cpu"
                if "mps" in str(device)
                else device
            )
            .eval()
        )
        assert vocos_ckpt_path, "vocos_ckpt_path should not be None"
        vocos.load_state_dict(torch.load(vocos_ckpt_path, weights_only=True, mmap=True))
        self.vocos = vocos
        self.logger.log(logging.INFO, "vocos loaded.")

        dvae = (
            DVAE(
                decoder_config=asdict(self.config.dvae.decoder),
                encoder_config=asdict(self.config.dvae.encoder),
                vq_config=asdict(self.config.dvae.vq),
                dim=self.config.dvae.decoder.idim,
                coef=coef,
            )
            .to(device)
            .eval()
        )
        coef = str(dvae)
        assert dvae_ckpt_path, "dvae_ckpt_path should not be None"
        dvae.load_state_dict(torch.load(dvae_ckpt_path, weights_only=True, mmap=True))
        self.dvae = dvae
        self.logger.log(logging.INFO, "dvae loaded.")

        gpt = GPT(
            gpt_config=asdict(self.config.gpt),
            use_flash_attn=use_flash_attn,
            device=device,
            logger=self.logger,
        ).eval()
        assert gpt_ckpt_path, "gpt_ckpt_path should not be None"
        gpt.from_pretrained(gpt_ckpt_path)
        gpt.prepare(compile=compile and "cuda" in str(device))
        self.gpt = gpt
        spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), "spk_stat.pt")
        assert os.path.exists(spk_stat_path), f"Missing spk_stat.pt: {spk_stat_path}"
        spk_stat: torch.Tensor = torch.load(
            spk_stat_path,
            weights_only=True,
            mmap=True,
            map_location=device,
        )
        self.std, self.mean = spk_stat.requires_grad_(False).chunk(2)
        self.logger.log(logging.INFO, "gpt loaded.")

        decoder = (
            DVAE(
                decoder_config=asdict(self.config.decoder),
                dim=self.config.decoder.idim,
                coef=coef,
            )
            .to(device)
            .eval()
        )
        coef = str(decoder)
        assert decoder_ckpt_path, "decoder_ckpt_path should not be None"
        decoder.load_state_dict(
            torch.load(decoder_ckpt_path, weights_only=True, mmap=True)
        )
        self.decoder = decoder
        self.logger.log(logging.INFO, "decoder loaded.")

        if tokenizer_path:
            self.tokenizer = Tokenizer(tokenizer_path, device)
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

        if not skip_refine_text:
            refined = self._refine_text(
                text,
                self.device,
                params_refine_text,
            )
            text_tokens = refined.ids
            text_tokens = [i[i.less(self.tokenizer.break_0_ids)] for i in text_tokens]
            text = self.tokenizer.decode(text_tokens)
            refined.destroy()
            if refine_text_only:
                yield text
                return

        if stream:
            length = 0
            pass_batch_count = 0
        for result in self._infer_code(
            text,
            stream,
            self.device,
            use_decoder,
            params_infer_code,
        ):
            wavs = self._decode_to_wavs(
                result.hiddens if use_decoder else result.ids,
                use_decoder,
            )
            result.destroy()
            if stream:
                pass_batch_count += 1
                if pass_batch_count <= params_infer_code.pass_first_n_batches:
                    continue
                a = length
                b = a + params_infer_code.stream_speed
                if b > wavs.shape[1]:
                    b = wavs.shape[1]
                new_wavs = wavs[:, a:b]
                length = b
                yield new_wavs
            else:
                yield wavs
        if stream:
            new_wavs = wavs[:, length:]
            # Identify rows with non-zero elements using np.any
            # keep_rows = np.any(array != 0, axis=1)
            keep_cols = np.sum(new_wavs != 0, axis=0) > 0
            # Filter both rows and columns using slicing
            yield new_wavs[:][:, keep_cols]

    @torch.inference_mode()
    def _vocos_decode(self, spec: torch.Tensor) -> np.ndarray:
        if "mps" in str(self.device):
            return self.vocos.decode(spec.cpu()).cpu().numpy()
        else:
            return self.vocos.decode(spec).cpu().numpy()

    @torch.inference_mode()
    def _decode_to_wavs(
        self,
        result_list: List[torch.Tensor],
        use_decoder: bool,
    ):
        decoder = self.decoder if use_decoder else self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src
        del_all(result_list)
        mel_specs = decoder(batch_result)
        del batch_result
        wavs = self._vocos_decode(mel_specs)
        del mel_specs
        return wavs

    @torch.no_grad()
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
            text[i] = (
                t.replace("[Stts]", "")
                .replace("[spk_emb]", "")
                .replace("[empty_spk]", "")
                .strip()
            )
            """
            see https://github.com/2noise/ChatTTS/issues/459
            """

        if params.prompt:
            text = [params.prompt + i for i in text]

        txt_smp = "" if params.txt_smp is None else params.txt_smp
        if params.spk_emb is not None:
            text = [f"[Stts][spk_emb]{txt_smp}{i}[Ptts]" for i in text]
        else:
            text = [f"[Stts][empty_spk]{txt_smp}{i}[Ptts]" for i in text]

        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            text,
            self.gpt.num_vq,
            prompt_str=params.spk_smp,
            device=gpt.device_gpt,
        )

        emb = gpt(input_ids, text_mask)

        del text_mask

        if params.spk_emb is not None:
            self.tokenizer.apply_spk_emb(
                emb, params.spk_emb, input_ids, self.gpt.device_gpt
            )

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
            show_tqdm=params.show_tqdm,
            ensure_non_empty=params.ensure_non_empty,
            stream_batch=params.stream_batch,
            context=self.context,
        )

        del emb, input_ids
        del_all(logits_warpers)
        del_all(logits_processors)

        return result

    @torch.no_grad()
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

        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            text,
            self.gpt.num_vq,
            device=gpt.device_gpt,
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.tokenizer.len,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        emb = gpt(input_ids, text_mask)

        del text_mask

        result = next(
            gpt.generate(
                emb,
                input_ids,
                temperature=torch.tensor([params.temperature], device=device),
                eos_token=self.tokenizer.eos_token,
                attention_mask=attention_mask,
                max_new_token=params.max_new_token,
                min_new_token=params.min_new_token,
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
                infer_text=True,
                stream=False,
                show_tqdm=params.show_tqdm,
                ensure_non_empty=params.ensure_non_empty,
                context=self.context,
            )
        )

        del emb, input_ids
        del_all(logits_warpers)
        del_all(logits_processors)

        return result
