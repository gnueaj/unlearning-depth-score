#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Unlearning style privacy evaluation (MIA AUC + PrivLeak/RelDiff).

Computes MIA AUC on TOFU forget vs holdout split using selected attacks.
Optional reference model enables PrivLeak/RelDiff computation.

Attacks supported: loss, min_k, min_k++ , zlib
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

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
) -> EncodedExample:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        chat_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
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
    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
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
        log_probs_batch.append(target_log_probs[i, start_idx:end_idx + 1])
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
        out.append(log_probs[i, start_idx:end_idx + 1, :])
    return out


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
        mu = (torch.exp(all_probs) * all_probs).sum(-1)
        sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1) - torch.square(mu)
        sigma = torch.clamp(sigma, min=1e-6)
        scores = (
            target_prob.float().cpu().numpy() - mu.float().cpu().numpy()
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


def load_tofu_mia_dataset(split_name: str, tokenizer, max_length: int, use_chat_template: bool):
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
            )
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or short name")
    parser.add_argument("--reference_model", default=None, help="Reference model for privleak/rel_diff")
    parser.add_argument("--attack", choices=["loss", "min_k", "min_k++", "zlib"], default="min_k")
    parser.add_argument("--k", type=float, default=0.2, help="k for min-k attacks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--out_dir", type=str, default="runs/privacy_eval")
    args = parser.parse_args()

    model_id = get_model_id(args.model)
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, device_map="cuda")

    forget = load_tofu_mia_dataset("forget10", tokenizer, args.max_length, args.use_chat_template)
    holdout = load_tofu_mia_dataset("holdout10", tokenizer, args.max_length, args.use_chat_template)

    pad_id = tokenizer.pad_token_id
    def collate(examples):
        return _collate_batch(examples, pad_id)

    attack_cls = {
        "loss": LOSSAttack,
        "min_k": MinKProbAttack,
        "min_k++": MinKPlusPlusAttack,
        "zlib": ZLIBAttack,
    }[args.attack]

    extra_kwargs = {}
    if args.attack in ("min_k", "min_k++"):
        extra_kwargs["k"] = args.k
    if args.attack == "zlib":
        extra_kwargs["tokenizer"] = tokenizer

    output = mia_auc(
        attack_cls,
        model,
        data={"forget": forget, "holdout": holdout},
        collator=collate,
        batch_size=args.batch_size,
        **extra_kwargs,
    )

    summary = {
        "model": model_id,
        "attack": args.attack,
        "auc": output["auc"],
    }

    # Optional reference model for privleak/rel_diff
    if args.reference_model:
        ref_id = get_model_id(args.reference_model)
        ref_model = load_model(ref_id, device_map="cuda")
        ref_output = mia_auc(
            attack_cls,
            ref_model,
            data={"forget": forget, "holdout": holdout},
            collator=collate,
            batch_size=args.batch_size,
            **extra_kwargs,
        )
        ref_auc = ref_output["auc"]
        # privleak (OpenUnlearning convention): compare (1-auc)
        score = 1 - output["auc"]
        ref = 1 - ref_auc
        privleak = (score - ref) / (ref + 1e-10) * 100
        rel_diff = (output["auc"] - ref_auc) / (ref_auc + 1e-10) * 100
        summary.update(
            {
                "reference_model": ref_id,
                "ref_auc": ref_auc,
                "privleak": privleak,
                "rel_diff": rel_diff,
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

