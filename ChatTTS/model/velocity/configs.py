from typing import Optional, Union, Tuple
import os

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip

import argparse
import dataclasses
from dataclasses import dataclass


logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        num_audio_tokens: int = 1024,
        num_text_tokens: int = 80,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens

        if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import (
                snapshot_download,
            )  # pylint: disable=C

            model_path = snapshot_download(
                model_id=model, cache_dir=download_dir, revision=revision
            )
            self.model = model_path
            self.download_dir = model_path
            self.tokenizer = model_path

        self.hf_config = get_config(self.model, trust_remote_code, revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        supported_load_format = ["auto", "pt", "safetensors", "npcache", "dummy"]
        rocm_not_supported_load_format = []
        if load_format not in supported_load_format:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
            )
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f
                for f in supported_load_format
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. "
                f"Supported load format are "
                f"{rocm_supported_load_format}"
            )

        # TODO: Remove this check once HF updates the pt weights of Mixtral.
        architectures = getattr(self.hf_config, "architectures", [])
        if "MixtralForCausalLM" in architectures and load_format == "pt":
            raise ValueError(
                "Currently, the 'pt' format is not supported for Mixtral. "
                "Please use the 'safetensors' format instead. "
            )
        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'."
            )
        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = ["awq", "gptq", "squeezellm"]
        rocm_not_supported_quantization = ["awq"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            if self.quantization is None:
                self.quantization = hf_quant_method
            elif self.quantization != hf_quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({hf_quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization})."
                )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )
            if is_hip() and self.quantization in rocm_not_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not supported "
                    f"in ROCm."
                )
            logger.warning(
                f"{self.quantization} quantization is not fully "
                "optimized yet. The speed can be slower than "
                "non-quantized models."
            )

    def _verify_cuda_graph(self) -> None:
        if self.max_context_len_to_capture is None:
            self.max_context_len_to_capture = self.max_model_len
        self.max_context_len_to_capture = min(
            self.max_context_len_to_capture, self.max_model_len
        )

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size})."
            )

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.sliding_window = sliding_window
        self._verify_args()

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}."
            )

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (
            f"{cpu_memory_usage / _GB:.2f} GiB out of "
            f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
            "allocated for the swap space."
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
        max_parallel_loading_workers: Optional[int] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError("Pipeline parallelism is not supported yet.")


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32
                # models.
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if is_hip() and torch_dtype == torch.float32:
        rocm_supported_dtypes = [
            k
            for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items()
            if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
        ]
        raise ValueError(
            f"dtype '{dtype}' is not supported in ROCm. "
            f"Supported dtypes are {rocm_supported_dtypes}"
        )

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            f"{possible_keys}. Assuming the model's maximum length is "
            f"{default_max_len}."
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
            derived_max_model_len = rope_scaling["original_max_position_embeddings"]
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(
            f"User-specified max_model_len ({max_model_len}) is greater than "
            f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
            " in model's config.json). This may lead to incorrect model "
            "outputs or CUDA errors. Make sure the value is correct and "
            "within the model context size."
        )
    return int(max_model_len)


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    num_audio_tokens: int = 1024
    num_text_tokens: int = 80

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            "--model",
            type=str,
            default="facebook/opt-125m",
            help="name or path of the huggingface model to use",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=EngineArgs.tokenizer,
            help="name or path of the huggingface tokenizer to use",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="the specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--tokenizer-revision",
            type=str,
            default=None,
            help="the specific tokenizer version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help='tokenizer mode. "auto" will use the fast '
            'tokenizer if available, and "slow" will '
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="trust remote code from huggingface",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=EngineArgs.download_dir,
            help="directory to download and load the weights, "
            "default to the default cache dir of "
            "huggingface",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=EngineArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=EngineArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="data type for model weights and activations. "
            'The "auto" option will use FP16 precision '
            "for FP32 and FP16 models, and BF16 precision "
            "for BF16 models.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="model context length. If unspecified, "
            "will be automatically derived from the model.",
        )
        # Parallel arguments
        parser.add_argument(
            "--worker-use-ray",
            action="store_true",
            help="use Ray for distributed serving, will be "
            "automatically set when using more than 1 GPU",
        )
        parser.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=EngineArgs.pipeline_parallel_size,
            help="number of pipeline stages",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="number of tensor parallel replicas",
        )
        parser.add_argument(
            "--max-parallel-loading-workers",
            type=int,
            help="load model sequentially in multiple batches, "
            "to avoid RAM OOM when using tensor "
            "parallel and large models",
        )
        # KV cache arguments
        parser.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            choices=[8, 16, 32],
            help="token block size",
        )
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument(
            "--seed", type=int, default=EngineArgs.seed, help="random seed"
        )
        parser.add_argument(
            "--swap-space",
            type=int,
            default=EngineArgs.swap_space,
            help="CPU swap space size (GiB) per GPU",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="the fraction of GPU memory to be used for "
            "the model executor, which can range from 0 to 1."
            "If unspecified, will use the default value of 0.9.",
        )
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="maximum number of batched tokens per " "iteration",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="maximum number of sequences per iteration",
        )
        parser.add_argument(
            "--max-paddings",
            type=int,
            default=EngineArgs.max_paddings,
            help="maximum number of paddings in a batch",
        )
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="disable logging statistics",
        )
        # Quantization settings.
        parser.add_argument(
            "--quantization",
            "-q",
            type=str,
            choices=["awq", "gptq", "squeezellm", None],
            default=None,
            help="Method used to quantize the weights. If "
            "None, we first check the `quantization_config` "
            "attribute in the model config file. If that is "
            "None, we assume the model weights are not "
            "quantized and use `dtype` to determine the data "
            "type of the weights.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Always use eager-mode PyTorch. If False, "
            "will use eager mode and CUDA graph in hybrid "
            "for maximal performance and flexibility.",
        )
        parser.add_argument(
            "--max-context-len-to-capture",
            type=int,
            default=EngineArgs.max_context_len_to_capture,
            help="maximum context length covered by CUDA "
            "graphs. When a sequence has context length "
            "larger than this, we fall back to eager mode.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        model_config = ModelConfig(
            self.model,
            self.tokenizer,
            self.tokenizer_mode,
            self.trust_remote_code,
            self.download_dir,
            self.load_format,
            self.dtype,
            self.seed,
            self.revision,
            self.tokenizer_revision,
            self.max_model_len,
            self.quantization,
            self.enforce_eager,
            self.max_context_len_to_capture,
            self.num_audio_tokens,
            self.num_text_tokens,
        )
        cache_config = CacheConfig(
            self.block_size,
            self.gpu_memory_utilization,
            self.swap_space,
            model_config.get_sliding_window(),
        )
        parallel_config = ParallelConfig(
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.worker_use_ray,
            self.max_parallel_loading_workers,
        )
        scheduler_config = SchedulerConfig(
            self.max_num_batched_tokens,
            self.max_num_seqs,
            model_config.max_model_len,
            self.max_paddings,
        )
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--engine-use-ray",
            action="store_true",
            help="use Ray to start the LLM engine in a "
            "separate process as the server process.",
        )
        parser.add_argument(
            "--disable-log-requests",
            action="store_true",
            help="disable logging requests",
        )
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=None,
            help="max number of prompt characters or prompt "
            "ID numbers being printed in log. "
            "Default: unlimited.",
        )
        return parser
