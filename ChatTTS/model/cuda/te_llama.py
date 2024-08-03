# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# From https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py
#
# Edited by fumiama.

import re
from contextlib import contextmanager
from typing import Dict

import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

import torch

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaConfig,
)
from transformers.modeling_utils import _load_state_dict_into_model

from .patch import LlamaRMSNorm


@contextmanager
def replace_decoder(te_decoder_cls, llama_rms_norm_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    original_llama_rms_norm_cls = transformers.models.llama.modeling_llama.LlamaRMSNorm
    transformers.models.llama.modeling_llama.LlamaRMSNorm = llama_rms_norm_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
            original_llama_decoder_cls
        )
        transformers.models.llama.modeling_llama.LlamaRMSNorm = (
            original_llama_rms_norm_cls
        )


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            ),
        )


class TELlamaModel:
    """
    LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaModel`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(
            te_decoder_cls=TELlamaDecoderLayer, llama_rms_norm_cls=LlamaRMSNorm
        ):
            model = LlamaModel(config)
        return model

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        config: LlamaConfig,
    ):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """

        vanilla_model = cls(config)

        # replace_params copies parameters relevant only to TransformerEngine
        _replace_params(state_dict, vanilla_model.state_dict(), config)
        # _load_state_dict_into_model copies parameters other than those in TransformerEngine
        _load_state_dict_into_model(vanilla_model, state_dict, start_prefix="")

        return vanilla_model


def _replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"
            ].data[:] = hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.query_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.key_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.value_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.o_proj.weight"].data[:]
            )

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = (
                hf_state_dict[layer_prefix + "post_attention_layernorm.weight"].data[:]
            )

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                : config.intermediate_size
            ] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                config.intermediate_size :
            ] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = (
                hf_state_dict[layer_prefix + "mlp.down_proj.weight"].data[:]
            )
    return all_layer_prefixes
