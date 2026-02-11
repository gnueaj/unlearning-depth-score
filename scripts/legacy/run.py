#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patchscope: Unlearning Audit via Hidden State Patching

Three probe types for knowledge detection:
1. QA Probe: "Question: X?\nAnswer:" → direct knowledge query
2. Cloze Probe: "The answer is" → fill-in-the-blank
3. Choice Probe: "(A) X (B) Y\nAnswer: (" → probability comparison

Usage:
    python -m uds.run                    # Default QA probe
    python -m uds.run --preset choice    # Multiple-choice (most stable)
    python -m uds.run --debug            # Sanity check (source=target)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

from datasets import load_dataset
import torch


class TeeLogger:
    """Log to both stdout and file."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


from .config import PatchscopeConfig, PRESETS, UNLEARN_MODELS, get_model_id
from .models import load_models, get_num_layers
from .core import (
    get_answer_position_hidden,
    get_generated_answer_hidden,
    probe_knowledge,
    compute_knowledge_score,
    generate_baseline,
    get_hidden_stats,
)
from .probes import (
    build_qa_probe,
    build_tofu_probes,
    ProbeResult,
    TOFU_WRONG_ANSWERS,
)
from .utils import set_seed, safe_mkdir, parse_layers
from .tofu_entities import extract_entity


def get_short_model_name(model_id: str) -> str:
    """Extract short model name from full path."""
    # Check if it's a known short name
    for short_name, full_id in UNLEARN_MODELS.items():
        if model_id == full_id:
            return short_name

    # Extract from path: open-unlearning/unlearn_tofu_..._SimNPO_... -> SimNPO
    if "/" in model_id:
        model_id = model_id.split("/")[-1]

    # Try to find method name
    methods = ["SimNPO", "NPO", "IdkNLL", "IdkDPO", "GradDiff", "AltPO", "RMU", "UNDIAL", "full"]
    for method in methods:
        if method in model_id:
            return method.lower()

    return model_id[:20]


def build_config_from_args(args) -> PatchscopeConfig:
    """Build configuration from command-line arguments."""
    if args.preset and args.preset in PRESETS:
        config = PRESETS[args.preset]
    else:
        config = PatchscopeConfig()

    # Override with command-line args
    if args.target_model:
        config.model.target_model_id = args.target_model
    if args.source_model:
        config.model.source_model_id = get_model_id(args.source_model)
    if args.layers:
        config.probe.layers = args.layers
    if args.probe_type:
        config.probe.probe_type = args.probe_type
    if args.example_index is not None:
        config.data.example_index = args.example_index
    if args.num_examples:
        config.data.num_examples = args.num_examples
    if args.out_dir:
        config.out_dir = args.out_dir

    # Debug flags
    if args.debug:
        config.debug.source_equals_target = True
        config.model.source_model_id = config.model.target_model_id
    if args.quiet:
        config.debug.verbose = False

    if args.seed:
        config.seed = args.seed

    # Store no_prefix flag (passed through config as runtime attr)
    config._no_prefix = getattr(args, 'no_prefix', False)

    return config


def _build_forced_prefix(question: str, prefix: str) -> str:
    """
    Build forced prefix for patching.

    Args:
        question: The question
        prefix: The prefix part of the answer (before the entity)
            e.g., "The author's full name is" (no trailing space)

    Returns:
        Full prompt: "Question: ...?\nAnswer: {prefix}"
    """
    if prefix:
        return f"Question: {question}\nAnswer: {prefix}"
    else:
        return f"Question: {question}\nAnswer:"


def build_probe_prompt(
    probe_type: str,
    question: str,
    answer: str,
    wrong_answers: List[str]
) -> ProbeResult:
    """Build probe prompt based on type."""
    entity, _ = extract_entity(question, answer)

    if probe_type == "qa":
        # Direct QA: "Question: X?\nAnswer:"
        return ProbeResult(
            probe_type="qa",
            probe_prompt=f"Question: {question}\nAnswer:",
            expected_answer=entity,
        )

    elif probe_type == "cloze":
        # Cloze: "Question: X?\nThe answer is"
        return ProbeResult(
            probe_type="cloze",
            probe_prompt=f"Question: {question}\nThe answer is",
            expected_answer=entity,
        )

    elif probe_type == "choice":
        # Multiple choice
        all_choices = [entity] + wrong_answers[:3]
        letters = ["A", "B", "C", "D"][:len(all_choices)]
        choice_lines = [f"({l}) {c}" for l, c in zip(letters, all_choices)]

        prompt = f"Question: {question}\n" + "\n".join(choice_lines) + "\nAnswer: ("

        return ProbeResult(
            probe_type="choice",
            probe_prompt=prompt,
            expected_answer="A",  # Correct is always (A)
            choices=letters,
        )

    else:
        # Default to QA
        return ProbeResult(
            probe_type="qa",
            probe_prompt=f"Question: {question}\nAnswer:",
            expected_answer=entity,
        )


def run_uds(config: PatchscopeConfig, save_log: bool = True):
    """Main uds analysis."""
    set_seed(config.seed)

    # Update out_dir with model name if using default
    if config.out_dir.startswith("runs/") and config.out_dir.count("_") == 1:
        # Default format: runs/YYYYMMDD_HHMMSS
        # New format: runs/YYYYMMDD_HHMMSS_modelname
        model_name = get_short_model_name(config.model.source_model_id)
        config.out_dir = f"{config.out_dir}_{model_name}"

    safe_mkdir(config.out_dir)

    # Setup logging
    logger = None
    if save_log:
        log_path = os.path.join(config.out_dir, "run.log")
        logger = TeeLogger(log_path)
        sys.stdout = logger
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Patchscope Started")

    try:
        results = _run_impl(config)
    finally:
        if logger:
            sys.stdout = logger.terminal
            logger.close()

    return results


def _run_impl(config: PatchscopeConfig):
    """Internal implementation."""
    verbose = config.debug.verbose
    no_prefix = getattr(config, '_no_prefix', False)

    # Save config
    config_path = os.path.join(config.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": vars(config.model),
            "data": vars(config.data),
            "probe": {
                "layers": config.probe.layers,
                "probe_type": config.probe.probe_type,
                "wrong_answers": config.probe.wrong_answers,
            },
            "debug": vars(config.debug),
        }, f, indent=2, ensure_ascii=False)

    # Load models
    target_model, source_model, tokenizer = load_models(config.model, verbose=verbose)
    n_layers = get_num_layers(target_model)
    layer_list = parse_layers(config.probe.layers, n_layers)

    if verbose:
        print(f"[INFO] Layers: {layer_list} (total={n_layers})")
        print(f"[INFO] Probe type: {config.probe.probe_type}")

    # Load dataset
    ds = load_dataset(
        config.data.dataset_id,
        config.data.dataset_config,
        split=config.data.split
    )

    # Select examples
    if config.data.example_index >= 0:
        example_indices = [config.data.example_index]
    else:
        example_indices = list(range(min(config.data.num_examples, len(ds))))

    if verbose:
        print(f"[INFO] Examples: {example_indices}")

    device = next(source_model.parameters()).device
    all_results = []

    # Process each example
    for ex_idx in example_indices:
        ex = ds[ex_idx]
        question = str(ex.get("question", "")).strip()
        answer = str(ex.get("answer", "")).strip()
        entity, prefix = extract_entity(question, answer)

        print("\n" + "=" * 80)
        print(f"[EXAMPLE {ex_idx}]")
        print(f'  Question: "{question}"')
        print(f'  GT Answer: "{answer}"')
        print(f'  Entity: "{entity}"')
        if config.debug.source_equals_target:
            print(f"  [DEBUG] Source = Target")
        print("=" * 80)

        # Baseline generation (no patching)
        print("\n[BASELINE - No Patching]")
        source_prompt = f"Question: {question}\nAnswer:"

        print(f"  Prompt: {repr(source_prompt[:100])}")

        # Generate from Source (Unlearn) first
        source_gen = generate_baseline(source_model, tokenizer, source_prompt, config.debug.max_new_tokens)

        # Generate from Target (Full)
        target_gen = generate_baseline(target_model, tokenizer, source_prompt, config.debug.max_new_tokens)

        # Print in order: Q, GT, Source, Target
        print(f"  Source (Unlearn): {repr(source_gen.strip()[:80])}")
        print(f"  Target (Full):    {repr(target_gen.strip()[:80])}")

        # Detect if Source response is IDK-style
        source_gen_lower = source_gen.strip().lower()
        is_idk_response = any(idk in source_gen_lower for idk in [
            "i don't know", "i'm not sure", "i do not know", "i'm not informed",
            "i cannot", "i can't", "not aware", "no information"
        ])

        # Build probe
        probe = build_probe_prompt(
            config.probe.probe_type,
            question,
            answer,
            config.probe.wrong_answers
        )

        if verbose:
            print(f"\n[Probe: {probe.probe_type}]")
            print(f"  Expected entity: {repr(probe.expected_answer)}")

        # Choose mode based on Source response
        # IDK response → no-prefix (let IDK flow naturally)
        # Non-IDK response → GT prefix (to see knowledge influence)
        if is_idk_response:
            source_extract_prompt = f"Question: {question}\nAnswer:"
            target_patch_prompt = source_extract_prompt
            mode_str = "no-prefix (IDK detected)"
        else:
            source_extract_prompt = _build_forced_prefix(question, prefix)
            target_patch_prompt = source_extract_prompt
            mode_str = "GT prefix"

        # Probe each layer using generated answer hidden
        print(f"\n[PATCHING] Source hidden → Target")
        print(f"  Mode: {mode_str}")
        print(f"  Source prompt: \"...{source_extract_prompt[-50:]}\"")
        print(f"  Target prompt: \"...{target_patch_prompt[-50:]}\"")
        print(f"  Expected: {repr(entity)}")

        # Format top-k tokens for display
        def format_topk(meta, k=5):
            topk = meta.get("topk", [])[:k]
            return " | ".join([f"'{t['token_str']}': {t['prob']:.3f}" for t in topk])

        # Show what each model predicts at the extraction point
        _, first_meta = get_generated_answer_hidden(
            source_model, tokenizer, question, layer_list[0],
            forced_prefix=source_extract_prompt
        )

        source_topk = format_topk(first_meta)
        print(f"  Source next token: {repr(first_meta.get('predicted_token', '?'))}  [{source_topk}]")

        _, target_meta = get_generated_answer_hidden(
            target_model, tokenizer, question, layer_list[0],
            forced_prefix=target_patch_prompt
        )
        target_topk = format_topk(target_meta)
        print(f"  Target next token: {repr(target_meta.get('predicted_token', '?'))}  [{target_topk}]")

        print("-" * 150)
        print(f"{'Layer':<6} {'Patched Output':<80} {'Top-5 Next Tokens'}")
        print("-" * 150)

        example_results = []

        for layer_idx in layer_list:
            # Extract hidden from Source at source_extract_prompt
            hidden, meta = get_generated_answer_hidden(
                source_model, tokenizer, question, layer_idx,
                forced_prefix=source_extract_prompt
            )

            # Patch this hidden into Target at target_patch_prompt
            result = probe_knowledge(
                target_model, tokenizer,
                target_patch_prompt,
                entity,  # Expect the entity token
                patch_layer_idx=layer_idx,
                patch_vector=hidden,
                wrong_answers=config.probe.wrong_answers,
                top_k=config.probe.top_k,
                max_new_tokens=20,
            )

            # Add metadata
            result["meta"] = {
                "example_index": ex_idx,
                "question": question,
                "entity": entity,
                "source_predicted": meta.get("predicted_token", ""),
                "target_predicted": target_meta.get("predicted_token", ""),
            }
            result["source_meta"] = meta
            result["target_meta"] = target_meta

            if config.debug.log_hidden_stats:
                result["hidden_stats"] = get_hidden_stats(hidden)

            example_results.append(result)

            # Print layer output - single line format with wider columns
            gen = result["full_response"].strip().replace("\n", " ")
            # Cut off at repeated patterns (Question:, Answer:, etc.)
            for stop in ["  Question:", "  Answer:", "Question:", "Answer:"]:
                if stop in gen:
                    gen = gen[:gen.index(stop)].strip()
            # Truncate if too long (allow up to 75 chars for full visibility)
            if len(gen) > 75:
                gen = gen[:72] + "..."
            gen_display = repr(gen)

            # Format top-5 tokens with probabilities
            top5 = result.get("topk", [])[:5]
            top5_str = " | ".join([
                f"{t['token_str'].strip()!r}:{t['prob']:.3f}"
                for t in top5
            ])

            print(f"{layer_idx:<6} {gen_display:<80} {top5_str}")

        # Summary for this example
        score = compute_knowledge_score(example_results)
        print("-" * 150)
        print(f"[SUMMARY] Detection: {score['detection_rate']:.0%} | "
              f"Max P(correct): {score['max_correct_prob']:.4f} | "
              f"Max margin: {score['max_prob_margin']:+.4f} | "
              f"Layers: {score['layers_with_knowledge']}")

        all_results.append({
            "example_index": ex_idx,
            "question": question,
            "answer": answer,
            "entity": entity,
            "probe_type": probe.probe_type,
            "baseline": {
                "target": target_gen,
                "source": source_gen,
            },
            "layer_results": example_results,
            "score": score,
        })

    # Save results
    results_path = os.path.join(config.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Aggregate summary
    if all_results:
        avg_detection = sum(r["score"]["detection_rate"] for r in all_results) / len(all_results)
        avg_max_prob = sum(r["score"]["max_correct_prob"] for r in all_results) / len(all_results)
        avg_margin = sum(r["score"]["max_prob_margin"] for r in all_results) / len(all_results)
    else:
        avg_detection = avg_max_prob = avg_margin = 0

    summary = {
        "config": {
            "source": config.model.source_model_id,
            "target": config.model.target_model_id,
            "probe_type": config.probe.probe_type,
            "num_examples": len(example_indices),
        },
        "aggregate": {
            "avg_detection_rate": avg_detection,
            "avg_max_correct_prob": avg_max_prob,
            "avg_max_margin": avg_margin,
        },
    }

    summary_path = os.path.join(config.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[AGGREGATE]")
    print(f"  Examples: {len(example_indices)}")
    print(f"  Avg detection: {avg_detection:.0%}")
    print(f"  Avg max P(correct): {avg_max_prob:.4f}")
    print(f"  Avg max margin: {avg_margin:+.4f}")
    print("=" * 80)
    print(f"\n[DONE] {config.out_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed")

    return all_results


def main():
    # Build model list for help
    model_list = "\n".join([f"  {k:<15} - {v.split('_')[-1]}" for k, v in list(UNLEARN_MODELS.items())[:10]])

    parser = argparse.ArgumentParser(
        description="Patchscope: Unlearning Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Probe Types:
  qa      - Direct Q&A: "Question: X?\\nAnswer:"
  cloze   - Fill-in-blank: "The answer is"
  choice  - Multiple-choice: "(A) X (B) Y\\nAnswer: (" (most stable)

Unlearning Models (use short name with --source_model):
{model_list}
  ... (see config.py UNLEARN_MODELS for full list)

Examples:
  python -m uds.run                              # Default (SimNPO)
  python -m uds.run --source_model npo           # Use NPO model
  python -m uds.run --source_model idknll        # Use IdkNLL model
  python -m uds.run --source_model graddiff      # Use GradDiff model
  python -m uds.run --preset choice              # Multiple-choice probe
  python -m uds.run --debug                      # Sanity check
        """
    )

    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--probe_type", choices=["qa", "cloze", "choice"])
    parser.add_argument("--target_model", type=str)
    parser.add_argument("--source_model", type=str, help="Unlearning model (short name or full HF path)")
    parser.add_argument("--layers", type=str)
    parser.add_argument("--example_index", type=int)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--debug", action="store_true", help="Source=Target sanity check")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-prefix", action="store_true", dest="no_prefix",
                        help="Don't use forced prefix - extract hidden at 'Answer:' position (natural response)")

    args = parser.parse_args()
    config = build_config_from_args(args)
    run_uds(config)


if __name__ == "__main__":
    main()
