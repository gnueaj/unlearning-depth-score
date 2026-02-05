#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for meta-evaluation metrics (Open-Unlearning style).

Implements all 12 metrics from Table 2 of the Open-Unlearning paper:
  Memorization: ES, EM, Truth Ratio, Para.Prob, Probability
  Generation:   ROUGE, Para.ROUGE, Jailbreak ROUGE
  Privacy:      MIA-LOSS, MIA-ZLib, MIA-MinK, MIA-MinK++
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

from .memorization import (
    eval_prob_em_es,
    eval_prob_only_multi,
    geometric_mean,
    _encode_example,
    IGNORE_INDEX,
)
from .utility_eval import generate_texts, _build_prompt_ids
from .privacy_eval import (
    load_tofu_mia_dataset,
    _collate_batch,
    mia_auc_all_attacks,
)


MEM_METRICS = {"em", "es", "prob", "paraprob", "truth_ratio"}
GENERATION_METRICS = {"rouge", "para_rouge", "jailbreak_rouge"}
MIA_METRICS = {"mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp"}

# Backward compat alias
ROUGE_METRICS = GENERATION_METRICS

# All 12 Table 2 metrics (excluding UDS which is our 13th)
ALL_TABLE2_METRICS = sorted(MEM_METRICS | GENERATION_METRICS | MIA_METRICS)

# Jailbreak prefix from Wang et al. [63] (used in Open-Unlearning Appendix C.3)
JAILBREAK_PREFIX = "Sure, here is the answer:"


# Canonical name mapping for normalize_metrics_list
_METRIC_ALIASES = {
    "para_rouge": "para_rouge",
    "pararouge": "para_rouge",
    "para.rouge": "para_rouge",
    "jailbreak_rouge": "jailbreak_rouge",
    "jailbreakrouge": "jailbreak_rouge",
    "jailbreak": "jailbreak_rouge",
    "jb_rouge": "jailbreak_rouge",
    "mia_loss": "mia_loss",
    "loss": "mia_loss",
    "mia_zlib": "mia_zlib",
    "zlib": "mia_zlib",
    "mia_min_k": "mia_min_k",
    "mia_mink": "mia_min_k",
    "min_k": "mia_min_k",
    "mink": "mia_min_k",
    "mia_min_kpp": "mia_min_kpp",
    "mia_minkpp": "mia_min_kpp",
    "mia_min_k++": "mia_min_kpp",
    "min_k++": "mia_min_kpp",
    "mink++": "mia_min_kpp",
}


def normalize_metrics_list(metrics: List[str]) -> List[str]:
    out = []
    for m in metrics:
        key = m.strip().lower()
        if not key:
            continue
        # "all" expands to all 12 Table 2 metrics + UDS
        if key == "all":
            out.extend(ALL_TABLE2_METRICS)
            out.append("uds")
            continue
        # "table2" expands to all 12 Table 2 metrics (no UDS)
        if key == "table2":
            out.extend(ALL_TABLE2_METRICS)
            continue
        # Apply aliases
        key = _METRIC_ALIASES.get(key, key)
        out.append(key)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for k in out:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _normalize_list_field(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if isinstance(v, str) and v.strip()]
    if isinstance(val, str) and val.strip():
        return [val]
    return []


def load_forget10_perturbed():
    ds = load_dataset("locuslab/TOFU", "forget10_perturbed", split="train")
    data = list(ds)
    questions = [ex.get("question", "") for ex in data]
    answers = [ex.get("answer", "") for ex in data]
    paraphrases = [
        _normalize_list_field(ex.get("paraphrased_answer")) for ex in data
    ]
    wrongs = [
        _normalize_list_field(ex.get("perturbed_answer")) for ex in data
    ]
    return {
        "questions": questions,
        "answers": answers,
        "paraphrases": paraphrases,
        "wrongs": wrongs,
    }


def compute_mem_metrics(
    model,
    tokenizer,
    mem_data: Dict[str, List],
    batch_size: int = 8,
    max_length: int = 512,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
    metrics_filter: Optional[set] = None,  # e.g., {"em", "es"} for fast mode
) -> Dict[str, Optional[float]]:
    questions = mem_data["questions"]
    answers = mem_data["answers"]
    paraphrases = mem_data["paraphrases"]
    wrongs = mem_data["wrongs"]

    correct = eval_prob_em_es(
        model,
        tokenizer,
        questions,
        answers,
        prefixes=None,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )

    em_vals = []
    es_vals = []
    prob_vals = []
    for i in range(len(questions)):
        em = correct[i]["em"]
        es = correct[i]["es"]
        prob = correct[i]["prob"]
        if em is not None:
            em_vals.append(em)
        if es is not None:
            es_vals.append(es)
        if prob is not None:
            prob_vals.append(prob)

    # Skip paraphrased/wrong evaluations if only em/es requested (fast mode)
    need_para = metrics_filter is None or "paraprob" in metrics_filter or "truth_ratio" in metrics_filter
    need_wrong = metrics_filter is None or "truth_ratio" in metrics_filter

    para_probs = None
    if need_para and any(len(p) > 0 for p in paraphrases):
        para_eval = eval_prob_only_multi(
            model,
            tokenizer,
            questions,
            paraphrases,
            prefixes=None,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
        para_probs = {
            i: geometric_mean([x["prob"] for x in lst]) if lst else None
            for i, lst in para_eval.items()
        }

    truth_ratio_vals = []
    if need_wrong and any(len(w) > 0 for w in wrongs):
        wrong_eval = eval_prob_only_multi(
            model,
            tokenizer,
            questions,
            wrongs,
            prefixes=None,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
        for i in range(len(questions)):
            para = para_probs[i] if para_probs is not None else None
            wl = wrong_eval.get(i, [])
            if para is None or not wl:
                continue
            wrong_gm = geometric_mean([x["prob"] for x in wl])
            if wrong_gm is None:
                continue
            truth_ratio_vals.append(para / (para + wrong_gm + 1e-10))

    para_vals = []
    if para_probs is not None:
        para_vals = [v for v in para_probs.values() if v is not None]

    return {
        "em": float(np.mean(em_vals)) if em_vals else None,
        "es": float(np.mean(es_vals)) if es_vals else None,
        "prob": float(np.mean(prob_vals)) if prob_vals else None,
        "paraprob": float(np.mean(para_vals)) if para_vals else None,
        "truth_ratio": float(np.mean(truth_ratio_vals)) if truth_ratio_vals else None,
    }


def _generate_with_jailbreak(
    model,
    tokenizer,
    questions: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    max_new_tokens: int = 200,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Dict[int, str]:
    """Generate with 'Sure, here is the answer:' prefix injection (Wang et al.)."""
    results: Dict[int, str] = {}
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    jb_ids = tokenizer(JAILBREAK_PREFIX, add_special_tokens=False).input_ids

    for start in tqdm(range(0, len(questions), batch_size), desc="jb_gen", leave=False):
        batch_q = questions[start:start + batch_size]
        prompt_ids_list = []
        for q in batch_q:
            base_ids = _build_prompt_ids(
                tokenizer, q, max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                date_string=date_string,
            )
            full_ids = base_ids + jb_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
            prompt_ids_list.append(full_ids)

        max_len = max(len(p) for p in prompt_ids_list)
        input_ids = torch.full(
            (len(prompt_ids_list), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, -len(p):] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, -len(p):] = 1
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs = outputs.cpu()
        input_len = input_ids.size(1)
        for i in range(outputs.size(0)):
            gen_tokens = outputs[i, input_len:]
            if tokenizer.eos_token_id is not None:
                eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    gen_tokens = gen_tokens[:eos_positions[0].item()]
            gen_text = tokenizer.decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            results[start + i] = gen_text

    tokenizer.padding_side = old_pad_side
    return results


def _get_rouge_ground_truths(
    tokenizer,
    questions: List[str],
    answers: List[str],
    max_length: int = 512,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> List[str]:
    """Extract ROUGE ground truths via tokenize->decode roundtrip (matches OpenUnlearning).

    OpenUnlearning extracts GT by:
      1. Tokenizing Q+A with chat template -> labels (IGNORE_INDEX for prompt)
      2. Decoding non-IGNORE label tokens with skip_special_tokens=True
      3. Removing input text via string replacement (safety measure)
    This handles any whitespace/template artifacts from tokenization.
    """
    ground_truths = []
    for q, a in zip(questions, answers):
        enc = _encode_example(
            tokenizer, q, a, index=0, max_length=max_length,
            add_eos=True,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
        label_tokens = enc.labels[enc.labels != IGNORE_INDEX]
        full_text = tokenizer.decode(
            label_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        input_text = tokenizer.decode(
            enc.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gt = full_text.replace(input_text, "").strip()
        ground_truths.append(gt)
    return ground_truths


def compute_generation_metrics(
    model,
    tokenizer,
    mem_data: Dict[str, List],
    batch_size: int = 8,
    max_length: int = 512,
    max_new_tokens: int = 200,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
    metrics_to_compute: Optional[Set[str]] = None,
) -> Dict[str, Optional[float]]:
    """Compute generation-based metrics: ROUGE, Para.ROUGE, Jailbreak ROUGE.

    Generates texts once for ROUGE + Para.ROUGE (shared), and separately
    for Jailbreak ROUGE (with prefix injection).
    """
    if metrics_to_compute is None:
        metrics_to_compute = GENERATION_METRICS
    questions = mem_data["questions"]
    answers = mem_data["answers"]
    paraphrases = mem_data["paraphrases"]
    results: Dict[str, Optional[float]] = {}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Pre-compute ground truths via tokenize->decode roundtrip (matches OpenUnlearning)
    gt_texts = _get_rouge_ground_truths(
        tokenizer, questions, answers,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )

    # ROUGE and Para.ROUGE share the same generation (original question → generate)
    need_normal_gen = "rouge" in metrics_to_compute or "para_rouge" in metrics_to_compute
    if need_normal_gen:
        gen_texts = generate_texts(
            model, tokenizer, questions,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )

        if "rouge" in metrics_to_compute:
            rouge_vals = []
            for i in range(len(questions)):
                if i in gen_texts:
                    r = scorer.score(gt_texts[i], gen_texts[i])["rougeL"].recall
                    rouge_vals.append(r)
            results["rouge"] = float(np.mean(rouge_vals)) if rouge_vals else None

        if "para_rouge" in metrics_to_compute:
            para_rouge_vals = []
            for i in range(len(questions)):
                if i in gen_texts and paraphrases[i]:
                    pr_vals = [
                        scorer.score(p.strip(), gen_texts[i])["rougeL"].recall
                        for p in paraphrases[i]
                    ]
                    para_rouge_vals.append(float(np.mean(pr_vals)))
            results["para_rouge"] = float(np.mean(para_rouge_vals)) if para_rouge_vals else None

    # Jailbreak ROUGE: separate generation with prefix injection
    if "jailbreak_rouge" in metrics_to_compute:
        jb_gen_texts = _generate_with_jailbreak(
            model, tokenizer, questions,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
        jb_rouge_vals = []
        for i in range(len(questions)):
            if i in jb_gen_texts:
                r = scorer.score(gt_texts[i], jb_gen_texts[i])["rougeL"].recall
                jb_rouge_vals.append(r)
        results["jailbreak_rouge"] = float(np.mean(jb_rouge_vals)) if jb_rouge_vals else None

    return results


def compute_rouge_metric(
    model,
    tokenizer,
    mem_data: Dict[str, List],
    batch_size: int = 8,
    max_length: int = 512,
    max_new_tokens: int = 200,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Optional[float]:
    """Backward-compatible: compute only ROUGE metric."""
    scores = compute_generation_metrics(
        model, tokenizer, mem_data,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
        metrics_to_compute={"rouge"},
    )
    return scores.get("rouge")


def prepare_mia_data(
    tokenizer,
    max_length: int = 512,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
):
    forget = load_tofu_mia_dataset(
        "forget10_perturbed",
        tokenizer,
        max_length,
        use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    holdout = load_tofu_mia_dataset(
        "holdout10",
        tokenizer,
        max_length,
        use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    pad_id = tokenizer.pad_token_id

    def collate(examples):
        return _collate_batch(examples, pad_id)

    return {"forget": forget, "holdout": holdout, "collate": collate}


def compute_mia_metrics(
    model,
    tokenizer,
    mia_data,
    batch_size: int = 8,
    k: float = 0.4,
) -> Dict[str, float]:
    aucs = mia_auc_all_attacks(
        model,
        data={"forget": mia_data["forget"], "holdout": mia_data["holdout"]},
        collator=mia_data["collate"],
        batch_size=batch_size,
        tokenizer=tokenizer,
        k=k,
    )
    return {
        "mia_loss": aucs["loss"],
        "mia_zlib": aucs["zlib"],
        "mia_min_k": aucs["min_k"],
        "mia_min_kpp": aucs["min_k++"],
    }

