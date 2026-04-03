import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .masking import masked_normalize


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    prompt_tokenized = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    output_tokenized = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have either pad_token_id or eos_token_id")

    examples = []
    for prompt_ids, output_ids in zip(
        prompt_tokenized["input_ids"],
        output_tokenized["input_ids"],
    ):
        full_ids = list(prompt_ids) + list(output_ids)
        if len(full_ids) < 2:
            raise ValueError("Each prompt+output pair must contain at least 2 tokens total")

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        prompt_len = len(prompt_ids)
        response_mask = [(i + 1) >= prompt_len for i in range(len(full_ids) - 1)]

        examples.append((input_ids, labels, response_mask))

    max_seq_len = max(len(input_ids) for input_ids, _, _ in examples)

    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for input_ids, labels, response_mask in examples:
        pad_len = max_seq_len - len(input_ids)

        batch_input_ids.append(input_ids + [pad_token_id] * pad_len)
        batch_labels.append(labels + [pad_token_id] * pad_len)
        batch_response_mask.append(response_mask + [False] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask, dtype=torch.bool),
    }



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    raise NotImplementedError


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raise NotImplementedError
