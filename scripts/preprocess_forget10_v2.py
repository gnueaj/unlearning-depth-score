#!/usr/bin/env python3
"""
Preprocess TOFU forget10 dataset v2 (manually verified):
1. Full Wrong: 30개 (Full 모델이 GT와 다른 정보 생성)
2. General Knowledge: 18개 (Retain 모델이 GT와 의미적으로 일치)
3. Valid: 352개
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchscope.utils import set_seed, safe_mkdir

# Manually verified indices
FULL_WRONG_IDX = {
    5, 49, 88, 134, 162, 199, 234, 238, 243, 256,
    278, 279, 282, 283, 284, 287, 288, 292, 293, 295,
    300, 302, 317, 323, 332, 346, 353, 354, 380, 386
}

GK_IDX = {17, 27, 28, 59, 86, 87, 92, 116, 119, 121, 127, 181, 210, 217, 361, 362, 383}


def clean_generated(text, max_len=200):
    for stop in ["Question:", "Answer:", "\n\n", "<|"]:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()[:max_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="tofu_data")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    full_model_id = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    retain_model_id = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"

    tokenizer = AutoTokenizer.from_pretrained(full_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModelForCausalLM.from_pretrained(
        full_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    retain_model = AutoModelForCausalLM.from_pretrained(
        retain_model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")

    full_wrong = []
    general_knowledge = []
    valid_examples = []

    print("\nProcessing...")
    for idx, example in enumerate(tqdm(dataset)):
        question = example["question"]
        answer = example["answer"]
        entity = answer.split(".")[0].strip() if "." in answer else answer[:100]
        words = answer.split()
        prefix = " ".join(words[:3]) if len(words) >= 3 else words[0] if words else ""
        prompt = f"Question: {question}\nAnswer: {prefix}"

        item = {"idx": idx, "question": question, "answer": answer, "entity": entity, "prefix": prefix}

        if idx in FULL_WRONG_IDX:
            # Get Full model output
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = full_model.generate(
                    **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            full_gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            item["full_output"] = prefix + " " + clean_generated(full_gen)
            full_wrong.append(item)

        elif idx in GK_IDX:
            # Get Retain model output
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = retain_model.generate(
                    **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            retain_gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            item["retain_output"] = prefix + " " + clean_generated(retain_gen)
            general_knowledge.append(item)

        else:
            valid_examples.append(item)

    print(f"\nFull Wrong: {len(full_wrong)}")
    print(f"General Knowledge: {len(general_knowledge)}")
    print(f"Valid: {len(valid_examples)}")

    with open(os.path.join(args.out_dir, "forget10_filtered_v2.json"), "w") as f:
        json.dump(valid_examples, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "forget10_full_wrong_v2.json"), "w") as f:
        json.dump(full_wrong, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "forget10_general_knowledge_v2.json"), "w") as f:
        json.dump(general_knowledge, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
