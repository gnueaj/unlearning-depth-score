#!/usr/bin/env python3
"""
EP10 Experiments - 4 Metrics (Mem/Priv/Utility/UDS)
75 models, parallel execution on GPU 0 and GPU 1
Results saved to runs/ep10/{memorization,privacy,utility,uds}/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
os.chdir(PROJECT_DIR)

def load_models():
    """Load EP10 model list"""
    with open('runs/ep10/model_list.json') as f:
        data = json.load(f)
    return data['ep10_models']


def get_completed_models(out_dir: str, metric: str) -> set:
    """Get set of already completed models from results file"""
    results_file = Path(out_dir) / 'results.json'
    if not results_file.exists():
        return set()

    try:
        with open(results_file) as f:
            data = json.load(f)

        # Different result structures for different metrics
        if metric == 'uds':
            # UDS stores results per model
            return set(data.keys())
        else:
            # Mem/Priv/Utility store in 'results' list
            if 'results' in data:
                return set(r.get('model', r.get('model_name', '')) for r in data['results'])
            return set(data.keys())
    except:
        return set()


def run_memorization(models: list, gpu: int, out_dir: str, resume: bool = True):
    """Run memorization evaluation"""
    completed = get_completed_models(out_dir, 'mem') if resume else set()
    remaining = [m for m in models if m not in completed]

    print(f"\n[Memorization] {len(remaining)}/{len(models)} models remaining (GPU {gpu})")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    for i, model in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {model}")
        cmd = [
            sys.executable, '-m', 'patchscope.memorization_eval',
            '--model', model,
            '--hf_dataset', 'locuslab/TOFU',
            '--hf_config', 'forget10_perturbed',
            '--use_chat_template',
            '--batch_size', '32',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Save individual result
        if result.returncode == 0:
            save_result(out_dir, model, result.stdout, 'mem')
        else:
            print(f"    ERROR: {result.stderr[:200]}")


def run_privacy(models: list, gpu: int, out_dir: str, resume: bool = True):
    """Run privacy evaluation"""
    completed = get_completed_models(out_dir, 'priv') if resume else set()
    remaining = [m for m in models if m not in completed]

    print(f"\n[Privacy] {len(remaining)}/{len(models)} models remaining (GPU {gpu})")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    for i, model in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {model}")
        cmd = [
            sys.executable, '-m', 'patchscope.privacy_eval',
            '--model', model,
            '--use_chat_template',
            '--batch_size', '32',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            save_result(out_dir, model, result.stdout, 'priv')
        else:
            print(f"    ERROR: {result.stderr[:200]}")


def run_utility(models: list, gpu: int, out_dir: str, resume: bool = True):
    """Run utility evaluation"""
    completed = get_completed_models(out_dir, 'util') if resume else set()
    remaining = [m for m in models if m not in completed]

    print(f"\n[Utility] {len(remaining)}/{len(models)} models remaining (GPU {gpu})")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    for i, model in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {model}")
        cmd = [
            sys.executable, '-m', 'patchscope.utility_eval',
            '--model', model,
            '--use_chat_template',
            '--batch_size', '32',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            save_result(out_dir, model, result.stdout, 'util')
        else:
            print(f"    ERROR: {result.stderr[:200]}")


def run_uds(models: list, gpu: int, out_dir: str, resume: bool = True):
    """Run UDS evaluation"""
    results_file = Path(out_dir) / 'results.json'

    # Load existing results
    if resume and results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    remaining = [m for m in models if m not in all_results]

    print(f"\n[UDS] {len(remaining)}/{len(models)} models remaining (GPU {gpu})")

    for i, model in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {model}")
        cmd = [
            sys.executable, 'exp_s1_teacher_forcing.py',
            '--unlearn_model', model,
            '--gpu', str(gpu),
            '--metric', 'logprob',
            '--delta_threshold', '0.05',
            '--patch_scope', 'span',
            '--reference', 'gt',
            '--mode', 'layer',
            '--batch_size', '32',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Find the output directory
            import glob
            run_dirs = sorted(glob.glob(f'runs/*_tf_{model}_layer'))
            if run_dirs:
                latest = run_dirs[-1]
                summary_file = Path(latest) / 'summary.json'
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                    all_results[model] = {
                        'avg_uds': summary.get('avg_uds', summary.get('avg_udr')),
                        'run_dir': latest,
                    }
                    # Save after each model
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
        else:
            print(f"    ERROR: {result.stderr[:200]}")


def save_result(out_dir: str, model: str, stdout: str, metric_type: str):
    """Parse and save result from stdout"""
    results_file = Path(out_dir) / 'results.json'

    # Load existing
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {'results': []}

    # Parse stdout for metrics (simplified - actual implementation may differ)
    result_entry = {'model': model}

    # Try to extract JSON from stdout
    for line in stdout.split('\n'):
        if line.strip().startswith('{'):
            try:
                parsed = json.loads(line.strip())
                result_entry.update(parsed)
                break
            except:
                pass

    all_results['results'].append(result_entry)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run EP10 experiments')
    parser.add_argument('--gpu', type=int, required=True, choices=[0, 1])
    parser.add_argument('--metric', type=str, choices=['mem', 'priv', 'util', 'uds', 'all'],
                        default='all', help='Which metric to run')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume from existing results')
    args = parser.parse_args()

    models = load_models()

    # Slice if specified
    if args.end:
        models = models[args.start:args.end]
    elif args.start > 0:
        models = models[args.start:]

    print(f"EP10 Experiments - GPU {args.gpu}")
    print(f"Models: {len(models)}")
    print(f"Metric: {args.metric}")
    print("=" * 50)

    resume = not args.no_resume

    if args.metric in ['mem', 'all']:
        run_memorization(models, args.gpu, 'runs/ep10/memorization', resume)

    if args.metric in ['priv', 'all']:
        run_privacy(models, args.gpu, 'runs/ep10/privacy', resume)

    if args.metric in ['util', 'all']:
        run_utility(models, args.gpu, 'runs/ep10/utility', resume)

    if args.metric in ['uds', 'all']:
        run_uds(models, args.gpu, 'runs/ep10/uds', resume)

    print("\n" + "=" * 50)
    print("COMPLETED!")


if __name__ == '__main__':
    main()
