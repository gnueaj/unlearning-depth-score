#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Unlearning style utility evaluation for TOFU.

Computes Model Utility as the harmonic mean of 9 metrics:
  - retain_Q_A_Prob, retain_Q_A_ROUGE, retain_Truth_Ratio
  - ra_Q_A_Prob,     ra_Q_A_ROUGE,     ra_Truth_Ratio
  - wf_Q_A_Prob,     wf_Q_A_ROUGE,     wf_Truth_Ratio

Datasets (HF TOFU):
  - retain_perturbed
  - real_authors_perturbed
  - world_facts_perturbed
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    """Encode QA pair without adding extra "Question/Answer" wrapper."""
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


def _batch_forward(model, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fn(shift_logits.transpose(-1, -2), shift_labels)
    token_losses = token_losses * mask
    sum_loss = token_losses.sum(dim=-1)
    num_tok = mask.sum(dim=-1).clamp(min=1)
    avg_loss = (sum_loss.float() / num_tok.float()).detach().cpu().numpy()
    prob = np.exp(-avg_loss)
    return avg_loss, prob


def eval_prob_only(
    model,
    tokenizer,
    questions: List[str],
    answers: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Dict[int, float]:
    pad_id = tokenizer.pad_token_id
    examples: List[EncodedExample] = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        examples.append(
            _encode_qa_example(
                tokenizer,
                q,
                a,
                index=i,
                max_length=max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                date_string=date_string,
            )
        )
    results: Dict[int, float] = {}
    for i in tqdm(range(0, len(examples), batch_size), desc="prob", leave=False):
        batch = _collate_batch(examples[i : i + batch_size], pad_id)
        _, prob = _batch_forward(model, batch)
        indices = batch["index"].cpu().numpy().tolist()
        for idx, p in zip(indices, prob.tolist()):
            results[idx] = p
    return results


def eval_prob_only_multi(
    model,
    tokenizer,
    questions: List[str],
    answers_list: List[List[str]],
    batch_size: int = 8,
    max_length: int = 512,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Dict[int, List[float]]:
    # Flatten to (ex_idx, answer_idx)
    flat_questions = []
    flat_answers = []
    flat_owner = []
    for i, (q, ans_list) in enumerate(zip(questions, answers_list)):
        for ans in ans_list:
            flat_questions.append(q)
            flat_answers.append(ans)
            flat_owner.append(i)
    if not flat_questions:
        return {i: [] for i in range(len(questions))}

    probs = eval_prob_only(
        model,
        tokenizer,
        flat_questions,
        flat_answers,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    out: Dict[int, List[float]] = {i: [] for i in range(len(questions))}
    for j, owner in enumerate(flat_owner):
        out[owner].append(probs[j])
    return out


def _build_prompt_ids(
    tokenizer,
    question: str,
    max_length: int,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> List[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        date_info = {"date_string": date_string} if date_string else {}
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **date_info
        )
    else:
        prompt_ids = tokenizer(question, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[-max_length:]
    return prompt_ids


def eval_rouge_l_recall(
    model,
    tokenizer,
    questions: List[str],
    answers: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    max_new_tokens: int = 200,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Dict[int, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results: Dict[int, float] = {}

    # Left-pad for generation
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in tqdm(range(0, len(questions), batch_size), desc="rouge", leave=False):
        batch_q = questions[start : start + batch_size]
        batch_a = answers[start : start + batch_size]
        prompt_ids_list = [
            _build_prompt_ids(
                tokenizer,
                q,
                max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                date_string=date_string,
            )
            for q in batch_q
        ]
        max_len = max(len(p) for p in prompt_ids_list)
        input_ids = torch.full(
            (len(prompt_ids_list), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, -len(p) :] = 1
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
        # decode generated portions (skip all input tokens including left-padding)
        outputs = outputs.cpu()
        input_len = input_ids.size(1)  # total input length including padding
        for i in range(outputs.size(0)):
            gen_tokens = outputs[i, input_len:]
            # stop at first eos if present
            if tokenizer.eos_token_id is not None:
                eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    gen_tokens = gen_tokens[: eos_positions[0].item()]
            gen_text = tokenizer.decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            gt = batch_a[i].strip()
            rouge = scorer.score(gt, gen_text)["rougeL"].recall
            results[start + i] = rouge

    tokenizer.padding_side = old_pad_side
    return results


def generate_texts(
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
    results: Dict[int, str] = {}
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in tqdm(range(0, len(questions), batch_size), desc="gen", leave=False):
        batch_q = questions[start : start + batch_size]
        prompt_ids_list = [
            _build_prompt_ids(
                tokenizer,
                q,
                max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                date_string=date_string,
            )
            for q in batch_q
        ]
        max_len = max(len(p) for p in prompt_ids_list)
        input_ids = torch.full(
            (len(prompt_ids_list), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, -len(p) :] = 1
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
        input_len = input_ids.size(1)  # total input length including padding
        for i in range(outputs.size(0)):
            gen_tokens = outputs[i, input_len:]
            if tokenizer.eos_token_id is not None:
                eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    gen_tokens = gen_tokens[: eos_positions[0].item()]
            gen_text = tokenizer.decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            results[start + i] = gen_text

    tokenizer.padding_side = old_pad_side
    return results


def eval_fluency_gibberish(
    texts: List[str],
    model_name: str,
    class_id: int = 0,
    batch_size: int = 32,
    max_length: int = 32,
    device: Optional[str] = None,
) -> float:
    if not texts:
        return float("nan")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc="fluency", leave=False):
        batch = texts[start : start + batch_size]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clf(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, class_id].detach().cpu().numpy()
        scores.extend(probs.tolist())
    return float(np.mean(scores)) if scores else float("nan")


def geometric_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    vals = np.array(vals, dtype=float)
    vals = np.clip(vals, 1e-12, None)
    return float(np.exp(np.mean(np.log(vals))))


def truth_ratio_true_better(correct: List[Optional[float]], wrong: List[List[float]]) -> float:
    """Truth ratio (true_better): mean(max(0, 1 - wrong/true)))."""
    ratios = []
    for c, ws in zip(correct, wrong):
        if c is None or not ws:
            continue
        wrong_gm = geometric_mean(ws)
        if wrong_gm is None:
            continue
        ratios.append(wrong_gm / (c + 1e-12))
    if not ratios:
        return float("nan")
    ratios = np.array(ratios, dtype=float)
    vals = np.maximum(0.0, 1.0 - ratios)
    return float(np.mean(vals))


def harmonic_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan")
    vals = np.array(vals)
    return float(len(vals) / np.sum(1.0 / (vals + 1e-12)))


def eval_mc_prob(
    model,
    tokenizer,
    questions: List[str],
    answers: List[str],
    wrong_list: List[List[str]],
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
    system_prompt: Optional[str],
    date_string: Optional[str] = None,
) -> Dict[int, float]:
    """Multiple-choice prob: p(correct) / sum(p(all))."""
    answers_list = []
    for ans, wrongs in zip(answers, wrong_list):
        wrongs = [w for w in wrongs if isinstance(w, str) and w.strip()]
        answers_list.append([ans] + wrongs)
    probs = eval_prob_only_multi(
        model,
        tokenizer,
        questions,
        answers_list,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    out = {}
    for i, all_probs in probs.items():
        if not all_probs:
            out[i] = float("nan")
            continue
        p_correct = all_probs[0]
        denom = sum(all_probs)
        out[i] = float(p_correct / (denom + 1e-12))
    return out


def compute_dataset_metrics(
    model,
    tokenizer,
    data,
    dataset_key: str,
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
    max_new_tokens: int,
    system_prompt: Optional[str] = None,
    date_string: Optional[str] = None,
) -> Dict[str, float]:
    questions = [ex.get("question", "") for ex in data]
    answers = [ex.get("answer", "") for ex in data]

    # Q_A_Prob (TOFU/Open-Unlearning)
    # - retain: direct length-normalized probability (exp(-avg_loss))
    # - real_authors/world_facts: multiple-choice relative probability
    if dataset_key in {"ra", "wf"}:
        pert = [ex.get("perturbed_answer", None) for ex in data]

        def norm_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str) and x.strip()]
            if isinstance(v, str) and v.strip():
                return [v]
            return []

        wrong_list = [norm_list(v) for v in pert]
        prob_map = eval_mc_prob(
            model,
            tokenizer,
            questions,
            answers,
            wrong_list,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
    else:
        prob_map = eval_prob_only(
            model,
            tokenizer,
            questions,
            answers,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            date_string=date_string,
        )
    prob_vals = list(prob_map.values())
    qa_prob = float(np.mean(prob_vals)) if prob_vals else float("nan")

    # Q_A_ROUGE (rougeL_recall)
    rouge_map = eval_rouge_l_recall(
        model,
        tokenizer,
        questions,
        answers,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    rouge_vals = list(rouge_map.values())
    qa_rouge = float(np.mean(rouge_vals)) if rouge_vals else float("nan")

    # Truth Ratio (Open-Unlearning/TOFU):
    # - retain: paraphrased answer vs perturbed answers
    # - ra/wf: original answer vs perturbed answers
    para = [ex.get("paraphrased_answer", None) for ex in data]
    pert = [ex.get("perturbed_answer", None) for ex in data]

    def norm_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str) and x.strip()]
        if isinstance(v, str) and v.strip():
            return [v]
        return []

    if dataset_key in {"ra", "wf"}:
        para_list = [[a] for a in answers]
    else:
        para_list = [norm_list(v) for v in para]
    pert_list = [norm_list(v) for v in pert]

    para_probs = eval_prob_only_multi(
        model,
        tokenizer,
        questions,
        para_list,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    pert_probs = eval_prob_only_multi(
        model,
        tokenizer,
        questions,
        pert_list,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )

    # Match Open-Unlearning: aggregate via mean of losses -> geometric mean of probs
    para_mean = [
        geometric_mean(para_probs[i]) if para_probs[i] else None
        for i in range(len(questions))
    ]
    pert_lists = [
        pert_probs[i] if pert_probs[i] else [] for i in range(len(questions))
    ]
    truth_ratio = truth_ratio_true_better(para_mean, pert_lists)

    return {
        "qa_prob": qa_prob,
        "qa_rouge": qa_rouge,
        "truth_ratio": truth_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or short name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--fluency_model", type=str,
                        default="madhurjindal/autonlp-Gibberish-Detector-492513457",
                        help="HF classifier for fluency (gibberish detection)")
    parser.add_argument("--fluency_class_id", type=int, default=0,
                        help="Classifier class id for non-gibberish")
    parser.add_argument("--fluency_batch_size", type=int, default=32)
    parser.add_argument("--fluency_max_length", type=int, default=32)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt for chat template (empty string to disable)")
    parser.add_argument("--date_string", type=str, default="10 Apr 2025",
                        help="Date string for chat template (Open-Unlearning uses 10 Apr 2025)")
    parser.add_argument("--out_dir", type=str, default="runs/utility_eval")
    args = parser.parse_args()

    model_id = get_model_id(args.model)
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, device_map="cuda")
    system_prompt = args.system_prompt if args.use_chat_template and args.system_prompt else None
    date_string = args.date_string if args.use_chat_template and args.date_string else None

    from datasets import load_dataset

    datasets_cfg = {
        "retain": "retain_perturbed",
        "ra": "real_authors_perturbed",
        "wf": "world_facts_perturbed",
    }

    metrics = {}
    for key, cfg in datasets_cfg.items():
        print(f"\nEvaluating {key} ({cfg})...")
        data = load_dataset("locuslab/TOFU", cfg, split="train")
        data = list(data)
        ds_metrics = compute_dataset_metrics(
            model,
            tokenizer,
            data,
            dataset_key=key,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            date_string=date_string,
        )
        metrics[f"{key}_Q_A_Prob"] = ds_metrics["qa_prob"]
        metrics[f"{key}_Q_A_ROUGE"] = ds_metrics["qa_rouge"]
        metrics[f"{key}_Truth_Ratio"] = ds_metrics["truth_ratio"]

    # Model Utility (HM of 9 metrics)
    util_keys = [
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_Truth_Ratio",
        "ra_Q_A_Prob",
        "ra_Q_A_ROUGE",
        "ra_Truth_Ratio",
        "wf_Q_A_Prob",
        "wf_Q_A_ROUGE",
        "wf_Truth_Ratio",
    ]
    util_vals = [metrics.get(k) for k in util_keys]
    model_utility = harmonic_mean(util_vals)

    # Fluency: gibberish detector on forget questions
    print("\nEvaluating fluency (gibberish classifier) on forget10...")
    forget_data = load_dataset("locuslab/TOFU", "forget10_perturbed", split="train")
    forget_questions = [ex.get("question", "") for ex in forget_data]
    gen_map = generate_texts(
        model,
        tokenizer,
        forget_questions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=args.use_chat_template,
        system_prompt=system_prompt,
        date_string=date_string,
    )
    gen_texts = [gen_map[i] for i in range(len(forget_questions))]
    fluency = eval_fluency_gibberish(
        gen_texts,
        model_name=args.fluency_model,
        class_id=args.fluency_class_id,
        batch_size=args.fluency_batch_size,
        max_length=args.fluency_max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    utility_hm = harmonic_mean([model_utility, fluency])

    summary = {
        "model": model_id,
        "metrics": metrics,
        "model_utility": model_utility,
        "fluency": fluency,
        "utility": utility_hm,
        "fluency_model": args.fluency_model,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
