import copy
from collections import defaultdict
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from .scheduler import Scheduler, SchedulerOutputs
from .configs import EngineArgs
from vllm.engine.metrics import record_metrics
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from .output import RequestOutput
from .sampling_params import SamplingParams
from .sequence import (
    SamplerOutput,
    Sequence,
    SequenceGroup,
    SequenceGroupOutput,
    SequenceOutput,
    SequenceStatus,
)
from vllm.transformers_utils.tokenizer import detokenize_incrementally, get_tokenizer
from vllm.utils import Counter, set_cuda_visible_devices, get_ip, get_open_port
import numpy as np

if ray:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        placement_group: Optional["PlacementGroup"],
        post_model_path: str,
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"seed={model_config.seed}), "
            f"post_model_path={post_model_path!r}"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()
        self.post_model_path = post_model_path
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(placement_group)
        else:
            self._init_workers()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _init_workers(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from .worker import Worker

        assert (
            self.parallel_config.world_size == 1
        ), "Ray is required if parallel_config.world_size > 1."

        self.workers: List[Worker] = []
        distributed_init_method = f"tcp://{get_ip()}:{get_open_port()}"
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            post_model_path=self.post_model_path,
        )
        self._run_workers("init_model")
        self._run_workers("load_model")

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        if self.parallel_config.tensor_parallel_size == 1:
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            num_gpus = 1

        self.driver_dummy_worker: RayWorkerVllm = None
        self.workers: List[RayWorkerVllm] = []

        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
            else:
                self.workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node."
            )

        driver_node_id, driver_gpu_ids = ray.get(
            self.driver_dummy_worker.get_node_and_gpu_ids.remote()
        )
        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in self.workers]
        )

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        node_workers[driver_node_id].append(0)
        node_gpus[driver_node_id].extend(driver_gpu_ids)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids, start=1):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # Set CUDA_VISIBLE_DEVICES for the driver.
        set_cuda_visible_devices(node_gpus[driver_node_id])
        for worker, (node_id, _) in zip(self.workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(node_gpus[node_id])

        distributed_init_method = f"tcp://{driver_ip}:{get_open_port()}"

        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        # Initialize torch distributed process group for the workers.
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)

        for rank, (worker, (node_id, _)) in enumerate(
            zip(self.workers, worker_node_and_gpu_ids), start=1
        ):
            local_rank = node_workers[node_id].index(rank)
            worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    model_config,
                    parallel_config,
                    scheduler_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                )
            )

        driver_rank = 0
        driver_local_rank = node_workers[driver_node_id].index(driver_rank)
        self.driver_worker = Worker(
            model_config,
            parallel_config,
            scheduler_config,
            driver_local_rank,
            driver_rank,
            distributed_init_method,
            is_driver_worker=True,
        )

        self._run_workers("init_model")
        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,
        )

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(
            f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}"
        )

        if num_gpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when "
                "initializing the engine."
            )
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine."
            )

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self._run_workers("warm_up_model")

    @classmethod
    def from_engine_args(
        cls, engine_args: EngineArgs, post_model_path=None
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        placement_group = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(
            *engine_configs,
            placement_group,
            log_stats=not engine_args.disable_log_stats,
            post_model_path=post_model_path,
        )
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        assert prompt_token_ids is not None, "prompt_token_ids must be provided"
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
        )
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() + sampling_params.max_tokens,
                    self.scheduler_config.max_model_len,
                )
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    seq_len=max_possible_length,
                )
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_outputs(
        self, seq_group: SequenceGroup, outputs: SequenceGroupOutput
    ) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {parent_seq.seq_id: [] for parent_seq in parent_seqs}
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(
                    child_sample.output_token,
                    child_sample.logprobs,
                    child_sample.hidden_states,
                    child_sample.finished,
                )
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(
                last_child_sample.output_token,
                last_child_sample.logprobs,
                last_child_sample.hidden_states,
                last_child_sample.finished,
            )
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            # self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
        new_finished_seqs = [
            (seq, parent, True) for seq, parent in child_seqs if seq.is_finished()
        ]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            ),
            reverse=True,
        )
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [
            (seq, parent) for seq, parent in child_seqs if not seq.is_finished()
        ]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            ),
            reverse=True,
        )

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params,
                best_running_seq,
                current_worst_seq,
            )

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
        self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs
    ) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in scheduled_seq_groups + scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(
                scheduler_outputs.prompt_run, scheduler_outputs.num_batched_tokens
            )
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            all_outputs = self._run_workers(
                "execute_model",
                driver_kwargs={
                    "seq_group_metadata_list": seq_group_metadata_list,
                    "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                    "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                    "blocks_to_copy": scheduler_outputs.blocks_to_copy,
                },
            )

            # Only the driver worker returns the sampling results.
            output = all_outputs[0]
        else:
            output = []

        return self._process_model_outputs(output, scheduler_outputs)

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [
            (t, n) for t, n in self.num_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC
        ]
        self.num_generation_tokens = [
            (t, n)
            for t, n in self.num_generation_tokens
            if now - t < _LOGGING_INTERVAL_SEC
        ]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = self.scheduler.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = self.scheduler.block_manager.get_num_free_cpu_blocks()
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info(
            "Avg prompt throughput: "
            f"{avg_prompt_throughput:.1f} tokens/s, "
            "Avg generation throughput: "
            f"{avg_generation_throughput:.1f} tokens/s, "
            f"Running: {len(self.scheduler.running)} reqs, "
            f"Swapped: {len(self.scheduler.swapped)} reqs, "
            f"Pending: {len(self.scheduler.waiting)} reqs, "
            f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
            f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%"
        )
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset, read_offset) = (
            detokenize_incrementally(
                self.tokenizer,
                all_input_ids=seq.get_token_ids(),
                prev_tokens=seq.tokens,
                prefix_offset=seq.prefix_offset,
                read_offset=seq.read_offset,
                skip_special_tokens=prms.skip_special_tokens,
                spaces_between_special_tokens=prms.spaces_between_special_tokens,
            )
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                if not sampling_params.include_stop_str_in_output:
                    # Truncate the output text so that the stop string is
                    # not included in the output.
                    seq.output_text = seq.output_text[: -len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.data.finished:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        for token_id in seq.get_last_token_id():
            if token_id == sampling_params.eos_token:
                seq.status = SequenceStatus.FINISHED_STOPPED
                return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if (not sampling_params.ignore_eos) and seq.get_last_token_id()[
            0
        ] == sampling_params.eos_token:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        # Start the ray workers first.
        ray_worker_outputs = [
            worker.execute_method.remote(method, *args, **kwargs)
            for worker in self.workers
        ]

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_output = getattr(self.driver_worker, method)(
            *driver_args, **driver_kwargs
        )

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs
