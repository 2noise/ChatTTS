
import os
import json
import logging
from functools import partial
from typing import Literal
import tempfile

import torch
from omegaconf import OmegaConf
from vocos import Vocos
from huggingface_hub import snapshot_download

from .model.dvae import DVAE
from .model.gpt import GPT_warpper
from .utils.gpu_utils import select_device
from .utils.infer_utils import count_invalid_characters, detect_language, apply_character_map, apply_half2full_map, HomophonesReplacer
from .utils.io_utils import get_latest_modified_file
from .infer.api import refine_text, infer_code
from .utils.download import check_all_assets, download_all_assets

logging.basicConfig(level = logging.INFO)


class Chat:
    def __init__(self, ):
        self.pretrain_models = {}
        self.normalizer = {}
        self.homophones_replacer = None
        self.logger = logging.getLogger(__name__)
        
    def check_model(self, level = logging.INFO, use_decoder = False):
        not_finish = False
        check_list = ['vocos', 'gpt', 'tokenizer']
        
        if use_decoder:
            check_list.append('decoder')
        else:
            check_list.append('dvae')
            
        for module in check_list:
            if module not in self.pretrain_models:
                self.logger.log(logging.WARNING, f'{module} not initialized.')
                not_finish = True
                
        if not not_finish:
            self.logger.log(level, f'All initialized.')
            
        return not not_finish

    def load_models(
        self,
        source: Literal['huggingface', 'local', 'custom']='local',
        force_redownload=False,
        custom_path='<LOCAL_PATH>',
        **kwargs,
    ):
        if source == 'local':
            download_path = os.getcwd()
            if not check_all_assets(update=True):
                with tempfile.TemporaryDirectory() as tmp:
                    download_all_assets(tmpdir=tmp)
                if not check_all_assets(update=False):
                    logging.error("counld not satisfy all assets needed.")
                    exit(1)
        elif source == 'huggingface':
            hf_home = os.getenv('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            try:
                download_path = get_latest_modified_file(os.path.join(hf_home, 'hub/models--2Noise--ChatTTS/snapshots'))
            except:
                download_path = None
            if download_path is None or force_redownload: 
                self.logger.log(logging.INFO, f'Download from HF: https://huggingface.co/2Noise/ChatTTS')
                download_path = snapshot_download(repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"])
            else:
                self.logger.log(logging.INFO, f'Load from cache: {download_path}')
        elif source == 'custom':
            self.logger.log(logging.INFO, f'Load from local: {custom_path}')
            download_path = custom_path

        self._load(**{k: os.path.join(download_path, v) for k, v in OmegaConf.load(os.path.join(download_path, 'config', 'path.yaml')).items()}, **kwargs)
        
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
        device: str = None,
        compile: bool = True,
    ):
        if not device:
            device = select_device(4096)
            self.logger.log(logging.INFO, f'use {device}')
            
        if vocos_config_path:
            vocos = Vocos.from_hparams(vocos_config_path).to(
                # vocos on mps will crash, use cpu fallback
                "cpu" if torch.backends.mps.is_available() else device
            ).eval()
            assert vocos_ckpt_path, 'vocos_ckpt_path should not be None'
            vocos.load_state_dict(torch.load(vocos_ckpt_path))
            self.pretrain_models['vocos'] = vocos
            self.logger.log(logging.INFO, 'vocos loaded.')
        
        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg).to(device).eval()
            assert dvae_ckpt_path, 'dvae_ckpt_path should not be None'
            dvae.load_state_dict(torch.load(dvae_ckpt_path))
            self.pretrain_models['dvae'] = dvae
            self.logger.log(logging.INFO, 'dvae loaded.')
            
        if gpt_config_path:
            cfg = OmegaConf.load(gpt_config_path)
            gpt = GPT_warpper(**cfg).to(device).eval()
            assert gpt_ckpt_path, 'gpt_ckpt_path should not be None'
            gpt.load_state_dict(torch.load(gpt_ckpt_path))
            if compile and 'cuda' in str(device):
                try:
                    gpt.gpt.forward = torch.compile(gpt.gpt.forward, backend='inductor', dynamic=True)
                except RuntimeError as e:
                    logging.warning(f'Compile failed,{e}. fallback to normal mode.')
            self.pretrain_models['gpt'] = gpt
            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), 'spk_stat.pt')
            assert os.path.exists(spk_stat_path), f'Missing spk_stat.pt: {spk_stat_path}'
            self.pretrain_models['spk_stat'] = torch.load(spk_stat_path).to(device)
            self.logger.log(logging.INFO, 'gpt loaded.')
            
        if decoder_config_path:
            cfg = OmegaConf.load(decoder_config_path)
            decoder = DVAE(**cfg).to(device).eval()
            assert decoder_ckpt_path, 'decoder_ckpt_path should not be None'
            decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location='cpu'))
            self.pretrain_models['decoder'] = decoder
            self.logger.log(logging.INFO, 'decoder loaded.')
        
        if tokenizer_path:
            tokenizer = torch.load(tokenizer_path, map_location='cpu')
            tokenizer.padding_side = 'left'
            self.pretrain_models['tokenizer'] = tokenizer
            self.logger.log(logging.INFO, 'tokenizer loaded.')
            
        self.check_model()
    
    def _infer(
        self, 
        text, 
        skip_refine_text=False, 
        refine_text_only=False, 
        params_refine_text={}, 
        params_infer_code={'prompt':'[speed_5]'}, 
        use_decoder=True,
        do_text_normalization=True,
        lang=None,
        stream=False,
        do_homophone_replacement=True
    ):
        
        assert self.check_model(use_decoder=use_decoder)
        
        if not isinstance(text, list): 
            text = [text]
        if do_text_normalization:
            for i, t in enumerate(text):
                _lang = detect_language(t) if lang is None else lang
                if self.init_normalizer(_lang):
                    text[i] = self.normalizer[_lang](t)
                    if _lang == 'zh':
                        text[i] = apply_half2full_map(text[i])
        for i, t in enumerate(text):
            invalid_characters = count_invalid_characters(t)
            if len(invalid_characters):
                self.logger.log(logging.WARNING, f'Invalid characters found! : {invalid_characters}')
                text[i] = apply_character_map(t)
            if do_homophone_replacement and self.init_homophones_replacer():
                text[i] = self.homophones_replacer.replace(t)
                if t != text[i]:
                    self.logger.log(logging.INFO, f'Homophones replace: {t} -> {text[i]}')

        if not skip_refine_text:
            text_tokens = refine_text(
                self.pretrain_models,
                text,
                **params_refine_text,
            )['ids']
            text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
            text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
            if refine_text_only:
                yield text
                return

        text = [params_infer_code.get('prompt', '') + i for i in text]
        params_infer_code.pop('prompt', '')
        result_gen = infer_code(self.pretrain_models, text, **params_infer_code, return_hidden=use_decoder, stream=stream)
        if use_decoder:
            field = 'hiddens'
            docoder_name = 'decoder'
        else:
            field = 'ids'
            docoder_name = 'dvae'
        vocos_decode = lambda spec: [self.pretrain_models['vocos'].decode(
                    i.cpu() if torch.backends.mps.is_available() else i
                ).cpu().numpy() for i in spec]
        if stream:

            length = 0
            for result in result_gen:
                chunk_data = result[field][0]
                assert len(result[field]) == 1
                start_seek = length
                length = len(chunk_data)
                self.logger.debug(f'{start_seek=} total len: {length}, new len: {length - start_seek = }')
                chunk_data = chunk_data[start_seek:]
                if not len(chunk_data):
                    continue
                self.logger.debug(f'new hidden {len(chunk_data)=}')
                mel_spec = [self.pretrain_models[docoder_name](i[None].permute(0,2,1)) for i in [chunk_data]]
                wav = vocos_decode(mel_spec)
                self.logger.debug(f'yield wav chunk {len(wav[0])=} {len(wav[0][0])=}')
                yield wav
            return
        mel_spec = [self.pretrain_models[docoder_name](i[None].permute(0,2,1)) for i in next(result_gen)[field]]
        yield vocos_decode(mel_spec)

    def infer(
        self, 
        text, 
        skip_refine_text=False, 
        refine_text_only=False, 
        params_refine_text={}, 
        params_infer_code={'prompt':'[speed_5]'}, 
        use_decoder=True,
        do_text_normalization=True,
        lang=None,
        stream=False,
        do_homophone_replacement=True,
    ):
        res_gen = self._infer(
            text,
            skip_refine_text,
            refine_text_only,
            params_refine_text,
            params_infer_code,
            use_decoder,
            do_text_normalization,
            lang,
            stream,
            do_homophone_replacement,
        )
        if stream:
            return res_gen
        else:
            return next(res_gen)
    
    def sample_random_speaker(self, ):
        
        dim = self.pretrain_models['gpt'].gpt.layers[0].mlp.gate_proj.in_features
        std, mean = self.pretrain_models['spk_stat'].chunk(2)
        return torch.randn(dim, device=std.device) * std + mean
    
    def init_normalizer(self, lang) -> bool:

        if lang in self.normalizer:
            return True

        if lang == 'zh':
            try:
                from tn.chinese.normalizer import Normalizer
                self.normalizer[lang] = Normalizer().normalize
                return True
            except:
                self.logger.log(
                    logging.WARNING,
                    'Package WeTextProcessing not found!',
                )
                self.logger.log(
                    logging.WARNING,
                    'Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing',
                )   
        else:
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer
                self.normalizer[lang] = partial(Normalizer(input_case='cased', lang=lang).normalize, verbose=False, punct_post_process=True)
                return True
            except:
                self.logger.log(
                    logging.WARNING,
                    'Package nemo_text_processing not found!',
                )
                self.logger.log(
                    logging.WARNING,
                    'Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing',
                )
        return False

    def init_homophones_replacer(self):
        if self.homophones_replacer:
            return True
        else:
            try:
                self.homophones_replacer = HomophonesReplacer(os.path.join(os.path.dirname(__file__), 'res', 'homophones_map.json'))
                self.logger.log(logging.INFO, 'homophones_replacer loaded.')
                return True
            except (IOError, json.JSONDecodeError) as e:
                self.logger.log(logging.WARNING, f'Error loading homophones map: {e}')
            except Exception as e:
                self.logger.log(logging.WARNING, f'Error loading homophones_replacer: {e}')
        return False
