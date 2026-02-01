#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Unlearning style memorization metrics.

Implements:
  - Exact Memorization (EM)
  - Extraction Strength (ES)
  - Probability (normalized exp(-avg_loss))
  - Truth Ratio (correct/(correct+wrong))  [OpenUnlearning meta-eval]
  - Memorization (Mem) = HM(1-ES, 1-EM, 1-ParaProb, 1-TruthRatio)

This is a lightweight, self-contained version adapted from the
OpenUnlearning evaluation code to fit this repo's data format.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    index: int


def _build_prompt(question: str, prefix: Optional[str] = None) -> str:
    if prefix:
        return f"Question: {question}\nAnswer: {prefix}"
    return f"Question: {question}\nAnswer:"


def _encode_example(
    tokenizer,
    question: str,
    answer: str,
    index: int,
    max_length: int,
    prefix: Optional[str] = None,
    add_eos: bool = True,
    use_chat_template: bool = False,
) -> EncodedExample:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        # Build chat-style prompt/response
        user_msg = _build_prompt(question, prefix)
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
        chat_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        if chat_ids[-1] != tokenizer.eos_token_id and add_eos and tokenizer.eos_token_id is not None:
            chat_ids = chat_ids + [tokenizer.eos_token_id]
        # Truncate from left if needed
        if len(chat_ids) > max_length:
            chat_ids = chat_ids[-max_length:]
            prompt_ids = prompt_ids[-max_length:]
        input_ids = torch.tensor(chat_ids, dtype=torch.long)
        labels = torch.tensor(
            [IGNORE_INDEX] * len(prompt_ids) + chat_ids[len(prompt_ids):],
            dtype=torch.long,
        )
    else:
        prompt = _build_prompt(question, prefix)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids

        eos_id = tokenizer.eos_token_id
        if add_eos and eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        # Truncate if needed: keep answer, trim prompt first, then answer
        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > max_length:
            keep_prompt = max(0, max_length - len(answer_ids))
            if keep_prompt == 0:
                prompt_ids = []
                answer_ids = answer_ids[:max_length]
            else:
                prompt_ids = prompt_ids[-keep_prompt:]
            total_len = len(prompt_ids) + len(answer_ids)
            if total_len > max_length:
                answer_ids = answer_ids[: max_length - len(prompt_ids)]

        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX] * len(prompt_ids) + answer_ids, dtype=torch.long)

    return EncodedExample(input_ids=input_ids, labels=labels, index=index)


def _collate_batch(examples: List[EncodedExample], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(e.input_ids) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    indices = []
    for i, ex in enumerate(examples):
        L = len(ex.input_ids)
        input_ids[i, :L] = ex.input_ids
        labels[i, :L] = ex.labels
        attention_mask[i, :L] = 1
        indices.append(ex.index)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "index": torch.tensor(indices, dtype=torch.long),
    }


def _batch_forward(
    model,
    batch: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, List[Optional[float]], List[Optional[float]]]:
    """Return (avg_loss, prob, em_list, es_list) for a batch."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
    logits = outputs.logits
    labels = batch["labels"]

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX

    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fn(shift_logits.transpose(-1, -2), shift_labels)
    token_losses = token_losses * mask
    sum_loss = token_losses.sum(dim=-1)
    num_tok = mask.sum(dim=-1).clamp(min=1)
    avg_loss = (sum_loss.float() / num_tok.float()).detach().cpu().numpy()
    prob = np.exp(-avg_loss)

    # EM / ES
    preds = torch.argmax(shift_logits, dim=-1)
    em_list: List[Optional[float]] = []
    es_list: List[Optional[float]] = []
    for i in range(preds.size(0)):
        valid_mask = mask[i]
        if valid_mask.sum().item() == 0:
            em_list.append(None)
            es_list.append(None)
            continue
        labels_i = shift_labels[i][valid_mask]
        preds_i = preds[i][valid_mask]
        em = (preds_i == labels_i).float().mean().item()
        em_list.append(em)
        # Extraction strength (OpenUnlearning / Carlini-style)
        valid_len = labels_i.numel()
        k = valid_len - 1
        for kk in range(valid_len):
            if torch.equal(preds_i[kk:], labels_i[kk:]):
                k = kk
                break
        es = 1.0 - (k / valid_len)
        es_list.append(es)
    return avg_loss, prob, em_list, es_list


def eval_prob_em_es(
    model,
    tokenizer,
    questions: List[str],
    answers: List[str],
    prefixes: Optional[List[str]] = None,
    batch_size: int = 8,
    max_length: int = 512,
    add_eos: bool = True,
    use_chat_template: bool = False,
) -> Dict[int, Dict[str, Optional[float]]]:
    """Evaluate prob, avg_loss, EM, ES for single-answer data."""
    pad_id = tokenizer.pad_token_id
    results: Dict[int, Dict[str, Optional[float]]] = {}
    examples: List[EncodedExample] = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        prefix = prefixes[i] if prefixes else None
        examples.append(
            _encode_example(
                tokenizer,
                q,
                a,
                index=i,
                max_length=max_length,
                prefix=prefix,
                add_eos=add_eos,
                use_chat_template=use_chat_template,
            )
        )

    for start in tqdm(range(0, len(examples), batch_size), desc="Mem eval"):
        batch = _collate_batch(examples[start:start + batch_size], pad_id)
        avg_loss, prob, em_list, es_list = _batch_forward(model, batch)
        indices = batch["index"].cpu().numpy().tolist()
        for idx, al, pr, em, es in zip(indices, avg_loss, prob, em_list, es_list):
            results[idx] = {
                "avg_loss": float(al),
                "prob": float(pr),
                "em": em,
                "es": es,
            }
    return results


def eval_prob_only_multi(
    model,
    tokenizer,
    questions: List[str],
    answers_list: List[List[str]],
    prefixes: Optional[List[str]] = None,
    batch_size: int = 8,
    max_length: int = 512,
    add_eos: bool = True,
    use_chat_template: bool = False,
) -> Dict[int, List[Dict[str, float]]]:
    """Evaluate prob/avg_loss for multi-answer per example."""
    pad_id = tokenizer.pad_token_id
    flat_examples: List[EncodedExample] = []
    flat_map: List[Tuple[int, int]] = []  # (example_idx, answer_idx)
    for i, answers in enumerate(answers_list):
        prefix = prefixes[i] if prefixes else None
        for j, ans in enumerate(answers):
            flat_examples.append(
                _encode_example(
                    tokenizer,
                    questions[i],
                    ans,
                    index=len(flat_examples),
                    max_length=max_length,
                    prefix=prefix,
                    add_eos=add_eos,
                    use_chat_template=use_chat_template,
                )
            )
            flat_map.append((i, j))

    # If no answers, return empty
    if not flat_examples:
        return {i: [] for i in range(len(questions))}

    # Prepare storage
    out: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(len(questions))}

    for start in tqdm(range(0, len(flat_examples), batch_size), desc="Mem eval (multi)"):
        batch = _collate_batch(flat_examples[start:start + batch_size], pad_id)
        avg_loss, prob, _, _ = _batch_forward(model, batch)
        # Map back
        for local_i, (al, pr) in enumerate(zip(avg_loss, prob)):
            global_i = start + local_i
            ex_idx, _ans_idx = flat_map[global_i]
            out[ex_idx].append({"avg_loss": float(al), "prob": float(pr)})
    return out


def truth_ratio_from_probs(correct: Dict[int, Dict[str, float]],
                           wrong: Dict[int, List[Dict[str, float]]]) -> Dict[int, Optional[float]]:
    """Truth ratio: correct_prob / (correct_prob + sum(wrong_prob))."""
    tr: Dict[int, Optional[float]] = {}
    for idx, c in correct.items():
        if c is None or c.get("prob") is None:
            tr[idx] = None
            continue
        wrong_list = wrong.get(idx, [])
        if not wrong_list:
            tr[idx] = None
            continue
        wrong_prob = sum(x["prob"] for x in wrong_list if x.get("prob") is not None)
        tr[idx] = c["prob"] / (c["prob"] + wrong_prob + 1e-10)
    return tr


def harmonic_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    denom = 0.0
    for v in vals:
        if v <= 0:
            return None
        denom += 1.0 / v
    return len(vals) / denom if denom > 0 else None


def compute_mem_score(es: Optional[float], em: Optional[float],
                      para_prob: Optional[float], truth_ratio: Optional[float]) -> Optional[float]:
    if es is None or em is None or para_prob is None or truth_ratio is None:
        return None
    return harmonic_mean([1 - es, 1 - em, 1 - para_prob, 1 - truth_ratio])


def sample_wrong_answers(all_answers: List[str], exclude_idx: int, k: int) -> List[str]:
    pool = [a for i, a in enumerate(all_answers) if i != exclude_idx]
    if not pool:
        return []
    k = min(k, len(pool))
    return random.sample(pool, k)
