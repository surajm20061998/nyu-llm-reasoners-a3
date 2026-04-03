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

    full_input_ids = []
    full_response_masks = []

    for prompt_ids, output_ids in zip(
        prompt_tokenized["input_ids"],
        output_tokenized["input_ids"],
    ):
        prompt_ids = list(prompt_ids)
        output_ids = list(output_ids)

        full_ids = prompt_ids + output_ids
        full_mask = [False] * len(prompt_ids) + [True] * len(output_ids)

        full_input_ids.append(full_ids)
        full_response_masks.append(full_mask)

    max_full_len = max(len(x) for x in full_input_ids)

    padded_full_input_ids = []
    padded_full_response_masks = []

    for full_ids, full_mask in zip(full_input_ids, full_response_masks):
        pad_len = max_full_len - len(full_ids)

        padded_full_input_ids.append(full_ids + [pad_token_id] * pad_len)
        padded_full_response_masks.append(full_mask + [False] * pad_len)

    padded_full_input_ids = torch.tensor(padded_full_input_ids, dtype=torch.long)
    padded_full_response_masks = torch.tensor(padded_full_response_masks, dtype=torch.bool)

    return {
        "input_ids": padded_full_input_ids[:, :-1],
        "labels": padded_full_input_ids[:, 1:],
        "response_mask": padded_full_response_masks[:, 1:],
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
