#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Unlearning style privacy evaluation.

Computes MIA AUC on TOFU forget vs holdout split using selected attacks.
Optional retain/full references enable sMIA scores and Privacy HM over:
  LOSS, ZLib, Min-k, Min-k++
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import get_model_id
from .models import load_model, load_tokenizer

IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    index: int


def _encode_qa_example(
    tokenizer,
    question: str,
    answer: str,
    index: int,
    max_length: int,
    add_eos: bool = True,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> EncodedExample:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        date_info = {"date_string": date_string} if date_string else {}
        chat_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, **date_info
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, **date_info
        )
        if add_eos and tokenizer.eos_token_id is not None:
            if len(chat_ids) == 0 or chat_ids[-1] != tokenizer.eos_token_id:
                chat_ids = chat_ids + [tokenizer.eos_token_id]
        if len(chat_ids) > max_length:
            chat_ids = chat_ids[-max_length:]
            prompt_ids = prompt_ids[-max_length:]
        input_ids = torch.tensor(chat_ids, dtype=torch.long)
        labels = torch.tensor(
            [IGNORE_INDEX] * len(prompt_ids) + chat_ids[len(prompt_ids):],
            dtype=torch.long,
        )
    else:
        prompt_ids = tokenizer(question, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if add_eos and tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
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


def evaluate_probability(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt.clamp(min=1)
    normalized_probs = torch.exp(-avg_losses)
    avg_losses = avg_losses.float().cpu().numpy().tolist()
    normalized_probs = normalized_probs.float().cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def tokenwise_logprobs(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    for i in range(target_log_probs.size(0)):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if actual_indices.numel() == 0:
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Align with Open-Unlearning: use next-token logprob starting at start_idx-1
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
    return log_probs_batch


def tokenwise_vocab_logprobs(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    out = []
    for i in range(log_probs.size(0)):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if actual_indices.numel() == 0:
            out.append(torch.empty((0, log_probs.size(-1)), device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Align with Open-Unlearning: use next-token logprob starting at start_idx-1
        out.append(log_probs[i, start_idx - 1 : end_idx, :])
    return out


def _evaluate_probability_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute avg_loss and prob from precomputed logits."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fn(shift_logits.transpose(-1, -2), shift_labels)
    token_losses = token_losses * mask
    sum_loss = token_losses.sum(dim=-1)
    num_tok = mask.sum(dim=-1).clamp(min=1)
    avg_loss = (sum_loss.float() / num_tok.float()).detach().cpu().numpy()
    prob = np.exp(-avg_loss)
    return avg_loss, prob


def _extract_target_texts(tokenizer, labels: torch.Tensor) -> List[str]:
    """Decode target texts from labels (IGNORED tokens removed)."""
    texts = []
    labels_cpu = labels.detach().cpu()
    for i in range(labels_cpu.size(0)):
        toks = labels_cpu[i][labels_cpu[i] != IGNORE_INDEX]
        text = tokenizer.decode(toks.tolist(), skip_special_tokens=True)
        texts.append(text)
    return texts


def _compute_attack_scores_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    k: float,
) -> Dict[str, List[float]]:
    """Compute LOSS/Min-k/Min-k++/ZLib scores from a single forward pass."""
    import zlib

    avg_loss, _ = _evaluate_probability_from_logits(logits, labels)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    next_tokens = input_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)

    texts = _extract_target_texts(tokenizer, labels)

    scores = {"loss": [], "min_k": [], "min_k++": [], "zlib": []}
    for i in range(target_log_probs.size(0)):
        lbls = labels[i]
        actual_indices = (lbls != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if actual_indices.numel() == 0:
            scores["loss"].append(float(avg_loss[i]))
            scores["min_k"].append(0.0)
            scores["min_k++"].append(0.0)
            text = texts[i]
            zlib_entropy = len(zlib.compress(text.encode("utf-8")))
            scores["zlib"].append(float(avg_loss[i]) / max(zlib_entropy, 1))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Align with Open-Unlearning: use start_idx-1
        lp = target_log_probs[i, start_idx - 1 : end_idx]
        vlp = log_probs[i, start_idx - 1 : end_idx, :]

        # LOSS
        scores["loss"].append(float(avg_loss[i]))

        # Min-k
        lp_np = lp.float().cpu().numpy()
        if lp_np.size == 0:
            scores["min_k"].append(0.0)
        else:
            num_k = max(1, int(len(lp_np) * k))
            sorted_vals = np.sort(lp_np)
            scores["min_k"].append(float(-np.mean(sorted_vals[:num_k])))

        # Min-k++
        if lp.numel() == 0:
            scores["min_k++"].append(0.0)
        else:
            mu = (torch.exp(vlp) * vlp).sum(-1).float()
            sigma = (torch.exp(vlp) * torch.square(vlp)).sum(-1).float() - torch.square(mu)
            sigma = torch.clamp(sigma, min=1e-6).float()
            z = (lp.float().cpu().numpy() - mu.cpu().numpy()) / torch.sqrt(sigma).cpu().numpy()
            num_k = max(1, int(len(z) * k))
            scores["min_k++"].append(float(-np.mean(sorted(z)[:num_k])))

        # ZLib
        text = texts[i]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        scores["zlib"].append(float(avg_loss[i]) / max(zlib_entropy, 1))

    return scores


def mia_auc_all_attacks(model, data, collator, batch_size, tokenizer, k=0.4):
    """Compute AUCs for LOSS/Min-k/Min-k++/ZLib with one forward per batch."""
    attacks = ["loss", "zlib", "min_k", "min_k++"]
    all_scores = {a: {"forget": [], "holdout": []} for a in attacks}

    def run_split(split_name: str, split_data):
        for start in tqdm(range(0, len(split_data), batch_size), leave=False):
            batch_examples = split_data[start : start + batch_size]
            batch = collator(batch_examples)
            batch.pop("index")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            scores = _compute_attack_scores_from_logits(
                outputs.logits, batch["input_ids"], batch["labels"], tokenizer, k
            )
            for a in attacks:
                all_scores[a][split_name].extend(scores[a])

    run_split("forget", data["forget"])
    run_split("holdout", data["holdout"])

    aucs = {}
    for a in attacks:
        forget_scores = all_scores[a]["forget"]
        holdout_scores = all_scores[a]["holdout"]
        scores = np.array(forget_scores + holdout_scores)
        labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
        aucs[a] = float(roc_auc_score(labels, scores))
    return aucs


class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        self.model = model
        self.data = data
        self.collator = collator
        self.batch_size = batch_size
        self.setup(**kwargs)

    def setup(self, **kwargs):
        pass

    def compute_batch_values(self, batch):
        raise NotImplementedError

    def compute_score(self, sample_stats):
        raise NotImplementedError

    def attack(self):
        all_scores = []
        all_indices = []
        for start in tqdm(range(0, len(self.data), self.batch_size), leave=False):
            batch_examples = self.data[start : start + self.batch_size]
            batch = self.collator(batch_examples)
            indices = batch.pop("index").cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]
            all_scores.extend(scores)
            all_indices.extend(indices)
        scores_by_index = {
            str(idx): {"score": float(score)}
            for idx, score in zip(all_indices, all_scores)
        }
        return {"agg_value": float(np.mean(all_scores)), "value_by_index": scores_by_index}


class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        return sample_stats["avg_loss"]


class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        return tokenwise_logprobs(self.model, batch)

    def compute_score(self, sample_stats):
        lp = sample_stats.float().cpu().numpy()
        if lp.size == 0:
            return 0.0
        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)
        return -float(np.mean(sorted_vals[:num_k]))


class MinKPlusPlusAttack(MinKProbAttack):
    def compute_batch_values(self, batch):
        vocab_log_probs = tokenwise_vocab_logprobs(self.model, batch)
        token_log_probs = tokenwise_logprobs(self.model, batch)
        return [
            {"vocab_log_probs": vlp, "token_log_probs": tlp}
            for vlp, tlp in zip(vocab_log_probs, token_log_probs)
        ]

    def compute_score(self, sample_stats):
        all_probs = sample_stats["vocab_log_probs"]
        target_prob = sample_stats["token_log_probs"]
        if len(target_prob) == 0:
            return 0.0
        mu = (torch.exp(all_probs) * all_probs).sum(-1).float()
        sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1).float() - torch.square(mu)
        sigma = torch.clamp(sigma, min=1e-6).float()
        scores = (
            target_prob.float().cpu().numpy() - mu.cpu().numpy()
        ) / torch.sqrt(sigma).cpu().numpy()
        num_k = max(1, int(len(scores) * self.k))
        return -float(np.mean(sorted(scores)[:num_k]))


class ZLIBAttack(Attack):
    def setup(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer

    def compute_batch_values(self, batch):
        import zlib
        eval_results = evaluate_probability(self.model, batch)
        # decode target texts
        labels = batch["labels"]
        texts = []
        for i in range(labels.size(0)):
            toks = labels[i][labels[i] != IGNORE_INDEX]
            text = self.tokenizer.decode(toks.tolist(), skip_special_tokens=True)
            texts.append(text)
        return [{"loss": r["avg_loss"], "text": t} for r, t in zip(eval_results, texts)]

    def compute_score(self, sample_stats):
        import zlib
        text = sample_stats["text"]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return sample_stats["loss"] / max(zlib_entropy, 1)


def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    output = {
        "forget": attack_cls(model=model, data=data["forget"], collator=collator, batch_size=batch_size, **kwargs).attack(),
        "holdout": attack_cls(model=model, data=data["holdout"], collator=collator, batch_size=batch_size, **kwargs).attack(),
    }
    forget_scores = [elem["score"] for elem in output["forget"]["value_by_index"].values()]
    holdout_scores = [elem["score"] for elem in output["holdout"]["value_by_index"].values()]
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
    auc_value = roc_auc_score(labels, scores)
    output["auc"] = float(auc_value)
    output["agg_value"] = float(auc_value)
    return output


def _harmonic_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return None
    vals = np.array(vals, dtype=float)
    return float(len(vals) / np.sum(1.0 / (vals + 1e-12)))


def _s_mia(auc_model: float, auc_retain: float, auc_full: float) -> Optional[float]:
    denom = abs(auc_full - auc_retain)
    if denom <= 1e-12:
        return None
    score = 1.0 - abs(auc_model - auc_retain) / denom
    return float(np.clip(score, 0.0, 1.0))


def load_tofu_mia_dataset(
    split_name: str,
    tokenizer,
    max_length: int,
    use_chat_template: bool,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
):
    from datasets import load_dataset
    data = load_dataset("locuslab/TOFU", split_name, split="train")
    items = []
    for i, ex in enumerate(data):
        items.append(
            _encode_qa_example(
                tokenizer,
                ex["question"],
                ex["answer"],
                index=i,
                max_length=max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                date_string=date_string,
            )
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or short name")
    parser.add_argument("--reference_model", default="retain", help="Retain model for sMIA/privleak/rel_diff")
    parser.add_argument("--full_model", default="full", help="Full model for sMIA scaling (Open-Unlearning)")
    parser.add_argument("--retain_full_cache", type=str, default=None,
                        help="Optional JSON cache for retain/full AUCs (avoids recompute).")
    parser.add_argument("--attack", choices=["loss", "min_k", "min_k++", "zlib", "all"], default="all")
    # Open-Unlearning TOFU MIA uses k=0.4
    parser.add_argument("--k", type=float, default=0.4, help="k for min-k attacks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt for chat template (empty string to disable)")
    parser.add_argument("--date_string", type=str, default="10 Apr 2025",
                        help="Date string for chat template (Open-Unlearning uses 10 Apr 2025)")
    parser.add_argument("--out_dir", type=str, default="runs/privacy_eval")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Attention implementation: eager, sdpa, or flash_attention_2")
    args = parser.parse_args()

    model_id = get_model_id(args.model)
    tokenizer = load_tokenizer(model_id)
    attn_impl = args.attn_implementation
    model = load_model(model_id, device_map="cuda", attn_implementation=attn_impl)
    system_prompt = args.system_prompt if args.use_chat_template and args.system_prompt else None
    date_string = args.date_string if args.use_chat_template and args.date_string else None

    # Match Open-Unlearning: forget split uses perturbed variant
    forget = load_tofu_mia_dataset(
        "forget10_perturbed",
        tokenizer,
        args.max_length,
        args.use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    holdout = load_tofu_mia_dataset(
        "holdout10",
        tokenizer,
        args.max_length,
        args.use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )

    pad_id = tokenizer.pad_token_id
    def collate(examples):
        return _collate_batch(examples, pad_id)

    attack_map = {
        "loss": LOSSAttack,
        "min_k": MinKProbAttack,
        "min_k++": MinKPlusPlusAttack,
        "zlib": ZLIBAttack,
    }

    def run_attack(attack_name: str, model_obj) -> float:
        attack_cls = attack_map[attack_name]
        extra_kwargs = {}
        if attack_name in ("min_k", "min_k++"):
            extra_kwargs["k"] = args.k
        if attack_name == "zlib":
            extra_kwargs["tokenizer"] = tokenizer
        out = mia_auc(
            attack_cls,
            model_obj,
            data={"forget": forget, "holdout": holdout},
            collator=collate,
            batch_size=args.batch_size,
            **extra_kwargs,
        )
        return out["auc"]

    if args.attack != "all":
        auc_val = run_attack(args.attack, model)
        summary = {
            "model": model_id,
            "attack": args.attack,
            "auc": auc_val,
        }
        if args.reference_model:
            ref_id = get_model_id(args.reference_model)
            ref_model = load_model(ref_id, device_map="cuda", attn_implementation=attn_impl)
            ref_auc = run_attack(args.attack, ref_model)
            # privleak (OpenUnlearning convention): compare (1-auc)
            score = 1 - auc_val
            ref = 1 - ref_auc
            privleak = (score - ref) / (ref + 1e-10) * 100
            rel_diff = (auc_val - ref_auc) / (ref_auc + 1e-10) * 100
            summary.update(
                {
                    "reference_model": ref_id,
                    "ref_auc": ref_auc,
                    "privleak": privleak,
                    "rel_diff": rel_diff,
                }
            )
    else:
        cache_version = "privacy_v2_tokenalign_fast"
        attacks = ["loss", "zlib", "min_k", "min_k++"]
        aucs = mia_auc_all_attacks(
            model,
            data={"forget": forget, "holdout": holdout},
            collator=collate,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            k=args.k,
        )

        retain_id = get_model_id(args.reference_model) if args.reference_model else None
        full_id = get_model_id(args.full_model) if args.full_model else None

        retain_aucs = {}
        full_aucs = {}

        cache_path = args.retain_full_cache
        cache_loaded = False
        if cache_path:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache = json.load(f)
                    if (
                        cache.get("retain_model") == retain_id
                        and cache.get("full_model") == full_id
                        and cache.get("version") == cache_version
                    ):
                        retain_aucs = cache.get("retain_aucs", {})
                        full_aucs = cache.get("full_aucs", {})
                        cache_loaded = True
            except Exception:
                cache_loaded = False

        if not cache_loaded:
            if retain_id:
                retain_model = load_model(retain_id, device_map="cuda", attn_implementation=attn_impl)
                retain_aucs = mia_auc_all_attacks(
                    retain_model,
                    data={"forget": forget, "holdout": holdout},
                    collator=collate,
                    batch_size=args.batch_size,
                    tokenizer=tokenizer,
                    k=args.k,
                )
                torch.cuda.empty_cache()
            if full_id:
                full_model = load_model(full_id, device_map="cuda", attn_implementation=attn_impl)
                full_aucs = mia_auc_all_attacks(
                    full_model,
                    data={"forget": forget, "holdout": holdout},
                    collator=collate,
                    batch_size=args.batch_size,
                    tokenizer=tokenizer,
                    k=args.k,
                )
                torch.cuda.empty_cache()
            if cache_path:
                try:
                    tmp_path = f"{cache_path}.tmp"
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "version": cache_version,
                                "retain_model": retain_id,
                                "full_model": full_id,
                                "retain_aucs": retain_aucs,
                                "full_aucs": full_aucs,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    os.replace(tmp_path, cache_path)
                except Exception:
                    pass

        s_mia = {}
        for a in attacks:
            if a in retain_aucs and a in full_aucs:
                s_mia[a] = _s_mia(aucs[a], retain_aucs[a], full_aucs[a])
            else:
                s_mia[a] = None

        privacy_score = _harmonic_mean(list(s_mia.values()))

        summary = {
            "model": model_id,
            "attack": "all",
            "attacks": {
                a: {
                    "auc": aucs[a],
                    "retain_auc": retain_aucs.get(a),
                    "full_auc": full_aucs.get(a),
                    "s_mia": s_mia.get(a),
                }
                for a in attacks
            },
            "privacy_score": privacy_score,
            "reference_model": retain_id,
            "full_model": full_id,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
