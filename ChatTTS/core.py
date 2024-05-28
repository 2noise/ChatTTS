
import os
import logging
from omegaconf import OmegaConf

import torch
from vocos import Vocos
from .model.dvae import DVAE
from .model.gpt import GPT_warpper
from .utils.gpu_utils import select_device
from .infer.api import refine_text, infer_code

from huggingface_hub import snapshot_download

logging.basicConfig(level = logging.INFO)


class Chat:
    def __init__(self, ):
        self.pretrain_models = {}
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
        
    def load_models(self, source='huggingface'):
        if source == 'huggingface':
            download_path = snapshot_download(repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"])
            self._load(**{k: os.path.join(download_path, v) for k, v in OmegaConf.load(os.path.join(download_path, 'config', 'path.yaml')).items()})
            
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
        device: str = None
    ):
        if not device:
            device = select_device(4096)
            self.logger.log(logging.INFO, f'use {device}')
            
        if vocos_config_path:
            vocos = Vocos.from_hparams(vocos_config_path).to(device).eval()
            assert vocos_ckpt_path, 'vocos_ckpt_path should not be None'
            vocos.load_state_dict(torch.load(vocos_ckpt_path))
            self.pretrain_models['vocos'] = vocos
            self.logger.log(logging.INFO, 'vocos loaded.')
        
        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg).to(device).eval()
            assert dvae_ckpt_path, 'dvae_ckpt_path should not be None'
            dvae.load_state_dict(torch.load(dvae_ckpt_path, map_location='cpu'))
            self.pretrain_models['dvae'] = dvae
            self.logger.log(logging.INFO, 'dvae loaded.')
            
        if gpt_config_path:
            cfg = OmegaConf.load(gpt_config_path)
            gpt = GPT_warpper(**cfg).to(device).eval()
            assert gpt_ckpt_path, 'gpt_ckpt_path should not be None'
            gpt.load_state_dict(torch.load(gpt_ckpt_path, map_location='cpu'))
            self.pretrain_models['gpt'] = gpt
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
    
    def infer(self, text, skip_refine_text=False, params_refine_text={}, params_infer_code={}, use_decoder=False, return_text=False):
        assert self.check_model(use_decoder=use_decoder)
        if not skip_refine_text:
            text_tokens = refine_text(self.pretrain_models, text, **params_refine_text)['ids']
            text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
            text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
        result = infer_code(self.pretrain_models, text, **params_infer_code, return_hidden=use_decoder)
        if use_decoder:
            mel_spec = [self.pretrain_models['decoder'](i[None].permute(0,2,1)) for i in result['hiddens']]
        else:
            mel_spec = [self.pretrain_models['dvae'](i[None].permute(0,2,1)) for i in result['ids']]
        wav = [self.pretrain_models['vocos'].decode(i).cpu().numpy() for i in mel_spec]

        if return_text:
            return wav, text
        else:
            return wav    
        

