#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI for Open-Unlearning style memorization evaluation.

Computes EM, ES, ParaProb, TruthRatio, Mem for a model on a dataset.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from .config import get_model_id
from .models import load_model, load_tokenizer
from .memorization import (
    eval_prob_em_es,
    eval_prob_only_multi,
    truth_ratio_from_probs,
    compute_mem_score,
    sample_wrong_answers,
)


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_list_field(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if isinstance(v, str) and v.strip()]
    if isinstance(val, str) and val.strip():
        return [val]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="JSON dataset path")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HF dataset name (e.g., locuslab/TOFU)")
    parser.add_argument("--hf_config", type=str, default=None,
                        help="HF dataset config (e.g., forget10_perturbed)")
    parser.add_argument("--hf_split", type=str, default="train",
                        help="HF dataset split (default: train)")
    parser.add_argument("--model", required=True, help="HF model id or short name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_prefix", action="store_true",
                        help="Include dataset prefix in prompt if available")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use tokenizer chat template for prompt/answer encoding")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--prefix_field", type=str, default="prefix")
    parser.add_argument("--paraphrase_field", type=str, default="paraphrases",
                        help="Field with list of paraphrased answers")
    parser.add_argument("--wrong_field", type=str, default="wrong_answers",
                        help="Field with list of wrong answers")
    parser.add_argument("--sample_wrong", type=int, default=0,
                        help="If >0 and wrong answers missing, sample k random answers")
    parser.add_argument("--out_dir", type=str, default="runs/memorization_eval")
    args = parser.parse_args()

    if args.hf_dataset:
        from datasets import load_dataset
        data = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        data = list(data)
        # Auto-detect TOFU perturbed fields if present
        if args.paraphrase_field == "paraphrases" and "paraphrased_answer" in data[0]:
            args.paraphrase_field = "paraphrased_answer"
        if args.wrong_field == "wrong_answers" and "perturbed_answer" in data[0]:
            args.wrong_field = "perturbed_answer"
    else:
        if not args.data_path:
            raise ValueError("Either --data_path or --hf_dataset must be provided")
        data = _load_json(args.data_path)
    model_id = get_model_id(args.model)

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, device_map="cuda")

    questions = [ex.get(args.question_field, "") for ex in data]
    answers = [ex.get(args.answer_field, "") for ex in data]
    prefixes = [ex.get(args.prefix_field, "") for ex in data] if args.use_prefix else None

    # Core metrics on correct answers
    correct = eval_prob_em_es(
        model,
        tokenizer,
        questions,
        answers,
        prefixes=prefixes,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )

    # Paraphrase probability
    paraphrase_list = [
        _normalize_list_field(ex.get(args.paraphrase_field)) for ex in data
    ]
    para_probs = None
    if any(len(p) > 0 for p in paraphrase_list):
        para_eval = eval_prob_only_multi(
            model,
            tokenizer,
            questions,
            paraphrase_list,
            prefixes=prefixes,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
        )
        para_probs = {
            i: (sum([x["prob"] for x in lst]) / len(lst)) if lst else None
            for i, lst in para_eval.items()
        }

    # Wrong answers for truth ratio
    wrong_list = [
        _normalize_list_field(ex.get(args.wrong_field)) for ex in data
    ]
    if args.sample_wrong > 0:
        all_answers = answers
        wrong_list = [
            lst if lst else sample_wrong_answers(all_answers, i, args.sample_wrong)
            for i, lst in enumerate(wrong_list)
        ]
    truth_ratio = None
    if any(len(w) > 0 for w in wrong_list):
        wrong_eval = eval_prob_only_multi(
            model,
            tokenizer,
            questions,
            wrong_list,
            prefixes=prefixes,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
        )
        truth_ratio = truth_ratio_from_probs(correct, wrong_eval)

    # Aggregate metrics
    em_vals = []
    es_vals = []
    prob_vals = []
    para_vals = []
    tr_vals = []
    mem_vals = []

    for i in range(len(data)):
        em = correct[i]["em"]
        es = correct[i]["es"]
        prob = correct[i]["prob"]
        para = para_probs[i] if para_probs is not None else None
        tr = truth_ratio[i] if truth_ratio is not None else None
        mem = compute_mem_score(es, em, para, tr)

        if em is not None:
            em_vals.append(em)
        if es is not None:
            es_vals.append(es)
        if prob is not None:
            prob_vals.append(prob)
        if para is not None:
            para_vals.append(para)
        if tr is not None:
            tr_vals.append(tr)
        if mem is not None:
            mem_vals.append(mem)

    summary = {
        "model": model_id,
        "num_examples": len(data),
        "avg_em": sum(em_vals) / len(em_vals) if em_vals else None,
        "avg_es": sum(es_vals) / len(es_vals) if es_vals else None,
        "avg_prob": sum(prob_vals) / len(prob_vals) if prob_vals else None,
        "avg_paraprob": sum(para_vals) / len(para_vals) if para_vals else None,
        "avg_truth_ratio": sum(tr_vals) / len(tr_vals) if tr_vals else None,
        "avg_mem": sum(mem_vals) / len(mem_vals) if mem_vals else None,
        "missing_paraphrase": para_probs is None,
        "missing_wrong": truth_ratio is None,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
