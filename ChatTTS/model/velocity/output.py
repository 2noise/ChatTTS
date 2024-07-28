from typing import List, Optional
import torch

from .sequence import (
    PromptLogprobs,
    SampleLogprobs,
    SequenceGroup,
    SequenceStatus,
)


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[SampleLogprobs],
        finish_reason: Optional[str] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.hidden_states = hidden_states

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"hidden_states={self.hidden_states.shape if self.hidden_states is not None else None})"
        )


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        # Get the top-n sequences.
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        if seq_group.sampling_params.use_beam_search:
            sorting_key = lambda seq: seq.get_beam_search_score(
                seq_group.sampling_params.length_penalty
            )
        else:
            sorting_key = lambda seq: seq.get_cumulative_logprob()
        sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
        top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs is None:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = None
            finshed_reason = SequenceStatus.get_finished_reason(seq.status)
            output = CompletionOutput(
                seqs.index(seq),
                seq.output_text,
                seq.get_output_token_ids(),
                seq.get_cumulative_logprob(),
                logprobs,
                finshed_reason,
                seq.data.hidden_states,
            )
            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
        finished = seq_group.is_finished()
        return cls(
            seq_group.request_id,
            prompt,
            prompt_token_ids,
            prompt_logprobs,
            outputs,
            finished,
        )

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished})"
        )
