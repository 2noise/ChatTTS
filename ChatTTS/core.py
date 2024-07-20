import os
import logging
import tempfile
from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Tuple, Dict, Union
from json import load
from pathlib import Path
import lzma
import pathlib
from ChatTTS.vllm_engine.post_model import Post_model
from safetensors.torch import save_file, safe_open
from omegaconf import OmegaConf
import numpy as np
import torch
from vocos import Vocos
from vocos.pretrained import instantiate_class
from huggingface_hub import snapshot_download
import pybase16384 as b14
from ChatTTS.vllm_engine.llm import LLM
from ChatTTS.vllm_engine.sampling_params import SamplingParams
import yaml
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
import pybase16384 as b14


class Chat:
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        utils_logger.set_logger(logger)

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
                for k, v in OmegaConf.load(
                    os.path.join(download_path, "config", "path.yaml")
                ).items()
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
        dim: int = self.hidden_size
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
        use_flash_attn=False,
    ):
        if device is None:
            device = select_device()
            self.logger.info("use device %s", str(device))
        self.device = device
        self.compile = compile

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
            self.num_vq = 4
            if not os.path.exists("asset/vllm_model"):
                gpt = GPT(
                    **cfg, use_flash_attn=use_flash_attn, device=device, logger=self.logger
                ).eval()
                assert gpt_ckpt_path, "gpt_ckpt_path should not be None"
                gpt.load_state_dict(torch.load(gpt_ckpt_path, weights_only=True, mmap=True))
                gpt.prepare(compile=compile and "cuda" in str(device))
                self.gpt = gpt
                pathlib.Path("asset/vllm_model").mkdir(parents=True, exist_ok=True)
                self.gpt.gpt.save_pretrained("asset/vllm_model/gpt")
                self.post_model = Post_model(
                    cfg.gpt_config.hidden_size,
                    cfg.num_audio_tokens,
                    cfg.num_text_tokens,
                    device = device
                ).to(device).eval()
                
                self.post_model.emb_code = self.gpt.emb_code
                self.post_model.emb_text = self.gpt.emb_text
                self.post_model.head_text = self.gpt.head_text
                self.post_model.head_code = self.gpt.head_code
                save_file(self.post_model.state_dict(), "asset/vllm_model/post_model.safetensors")
            
            self.num_audio_tokens = cfg.num_audio_tokens
            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), "spk_stat.pt")
            assert os.path.exists(
                spk_stat_path
            ), f"Missing spk_stat.pt: {spk_stat_path}"
            spk_stat: torch.Tensor = torch.load(
                spk_stat_path,
                weights_only=True,
                mmap=True,
                map_location=device,
            )
            self.std, self.mean = spk_stat.requires_grad_(False).chunk(2)
            self.logger.log(logging.INFO, "gpt loaded.")
            
            self.hidden_size = cfg.gpt_config.hidden_size
            self.gpt = LLM(
                model="asset/vllm_model/gpt",
                num_audio_tokens = cfg.num_audio_tokens,
                num_text_tokens = cfg.num_text_tokens,
                post_model_path="asset/vllm_model/post_model.safetensors",
            )
            
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

    @staticmethod
    def _decode_spk_emb(spk_emb: str) -> np.ndarray:
        return np.frombuffer(
            lzma.decompress(
                b14.decode_from_string(spk_emb),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype=np.float16,
        ).copy()

    @dataclass(repr=False, eq=False)
    class GenerationOutputs:
        ids: List[torch.Tensor]
        # attentions: List[Optional[Tuple[torch.FloatTensor, ...]]]
        hiddens: List[torch.Tensor]

        def destroy(self):
            del_all(self.ids)
            # del_all(self.attentions)
            # del_all(self.hiddens)
            
    @torch.no_grad()
    def _infer_code(
        self,
        text: Tuple[List[str], str],
        stream: bool,
        device: torch.device,
        return_hidden: bool,
        params: InferCodeParams,
    ):

        gpt: LLM = self.gpt

        if not isinstance(text, list):
            text = [text]

        assert len(text), "text should not be empty"

        if not isinstance(params.temperature, list):
            temperature = [params.temperature] * self.num_vq
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
            self.num_vq,
            prompt_str=params.spk_smp,
            device=self.device,
        )
        start_idx = input_ids.shape[-2]
    
        num_code = self.num_audio_tokens - 1

        logits_warpers, logits_processors = gen_logits(
            num_code=num_code,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )
        
        sample_params = SamplingParams(
            temperature=temperature,
            max_new_token=params.max_new_token,
            max_tokens = 8192,
            min_new_token=params.min_new_token,
            logits_processors=(logits_warpers, logits_processors),
            eos_token = num_code,
            infer_text=False,
            start_idx=start_idx
        )
        input_ids = [i.tolist() for i in input_ids]
        
        result = gpt.generate(
            None,
            sample_params,
            input_ids,
        )
        
        token_ids = []
        hidden_states = []
        for i in result:
            token_ids.append(torch.tensor(i.outputs[0].token_ids))
            hidden_states.append(i.outputs[0].hidden_states.to(torch.float32).to(self.device))
        return [self.GenerationOutputs(
            ids=token_ids,
            hiddens=hidden_states
        ),]

    @torch.no_grad()
    def _refine_text(
        self,
        text: str,
        device: torch.device,
        params: RefineTextParams,
    ):

        gpt:LLM = self.gpt

        if not isinstance(text, list):
            text = [text]

        text = [f"[Sbreak]{i}[Pbreak]{params.prompt}" for i in text]

        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            text,
            self.num_vq,
            device=self.device,
        )
        
        start_idx = input_ids.shape[-2]
        # print(start_idx)
        logits_warpers, logits_processors = gen_logits(
            num_code=self.tokenizer.len,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        sample_params = SamplingParams(
            temperature=params.temperature,
            max_new_token=params.max_new_token,
            max_tokens = 8192,
            min_new_token=params.min_new_token,
            logits_processors=(logits_warpers, logits_processors),
            eos_token = self.tokenizer.eos_token,
            infer_text=True,
            start_idx=start_idx
        )
        input_ids = [i.tolist() for i in input_ids]
        
        result = gpt.generate(
            None,
            sample_params,
            input_ids
        )
        token_ids = []
        hidden_states = []
        for i in result:
            token_ids.append(torch.tensor(i.outputs[0].token_ids))
            hidden_states.append(i.outputs[0].hidden_states)
        return self.GenerationOutputs(
            ids=token_ids,
            hiddens=hidden_states
        )
