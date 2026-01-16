#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Patchscope functionality: hidden state extraction, patching, and decoding.

Key concepts:
- Source model: The model to extract hidden states from (e.g., unlearned model)
- Target model: The model to patch and probe (e.g., original model with knowledge)
- Source position: Where to extract hidden state (answer generation point)
- Probe: The prompt structure used to elicit knowledge from patched target
"""

import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import get_layer_module, get_mlp_module


# =============================================================================
# Hidden State Extraction
# =============================================================================

@torch.inference_mode()
def get_hidden_at_position(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    position: int = -1
) -> torch.Tensor:
    """
    Extract hidden state at given layer and position.

    Args:
        model: The model to extract from
        input_ids: [B, T] input token ids
        attention_mask: [B, T] attention mask
        layer_idx: 0-based transformer block index
        position: Token position to extract (-1 for last)

    Returns:
        [B, D] hidden state tensor
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )
    # hidden_states[0] = embeddings, hidden_states[i+1] = after layer i
    hs = out.hidden_states[layer_idx + 1]
    return hs[:, position, :].detach()


@torch.inference_mode()
def get_all_layers_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_list: List[int],
    position: Optional[int] = -1
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states from ALL specified layers in ONE forward pass.

    This is 16x faster than calling get_hidden_at_position for each layer.

    Args:
        model: The model to extract from
        input_ids: [B, T] input token ids
        attention_mask: [B, T] attention mask
        layer_list: List of layer indices to extract from
        position: Token position to extract (-1 for last, None for all positions)

    Returns:
        Dict mapping layer_idx -> [B, D] or [B, T, D] hidden state tensor
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )
    # hidden_states[0] = embeddings, hidden_states[i+1] = after layer i
    result = {}
    for layer_idx in layer_list:
        hs = out.hidden_states[layer_idx + 1]
        if position is None:
            result[layer_idx] = hs.detach()  # [B, T, D] all positions
        else:
            result[layer_idx] = hs[:, position, :].detach()  # [B, D] single position
    return result


# Alias for backward compatibility
def get_last_token_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int
) -> torch.Tensor:
    """Extract hidden state at last token position."""
    return get_hidden_at_position(model, input_ids, attention_mask, layer_idx, position=-1)


@torch.inference_mode()
def get_answer_position_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    layer_idx: int,
    chat_template: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract hidden state at the answer generation point.

    This is the key position for unlearning audit:
    - After the question is fully processed
    - At the position where the model starts generating the answer

    Args:
        model: Model to extract from
        tokenizer: Tokenizer
        question: The question to ask
        layer_idx: Layer to extract from
        chat_template: Whether to use chat template

    Returns:
        Tuple of (hidden_state [1, D], metadata dict)
    """
    device = next(model.parameters()).device

    # Build prompt that ends right before answer generation
    if chat_template:
        try:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"Question: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    # Tokenize
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Extract hidden state at last position (= answer generation point)
    hidden = get_hidden_at_position(model, input_ids, attention_mask, layer_idx, position=-1)

    metadata = {
        "prompt": prompt,
        "prompt_len": int(input_ids.shape[1]),
        "last_token_id": int(input_ids[0, -1].item()),
        "last_token_str": tokenizer.decode([int(input_ids[0, -1].item())]),
    }

    return hidden, metadata


@torch.inference_mode()
def get_generated_answer_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    layer_idx: int,
    max_new_tokens: int = 50,
    forced_prefix: str = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract hidden state using a forced prefix prompt.

    Two modes:
    1. forced_prefix=None: Generate answer first, then extract at last token
    2. forced_prefix="...": Use exact prefix, extract hidden that predicts next token

    This allows fair comparison: same prefix for Source and Target.

    Args:
        model: Model to extract from
        tokenizer: Tokenizer
        question: The question to ask
        layer_idx: Layer to extract from
        max_new_tokens: Max tokens for generation (only used if forced_prefix=None)
        forced_prefix: If provided, use this exact prompt and extract at last token

    Returns:
        Tuple of (hidden_state [1, D], metadata dict)
    """
    device = next(model.parameters()).device

    if forced_prefix is not None:
        # Mode 2: Use forced prefix directly
        prompt = forced_prefix
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        hidden = get_hidden_at_position(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            layer_idx,
            position=-1
        )

        # Get what this model would predict next
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True
        )
        next_logits = out.logits[:, -1, :]
        next_token_id = torch.argmax(next_logits, dim=-1).item()
        predicted_token = tokenizer.decode([next_token_id])

        # Get top-k predictions
        probs = torch.softmax(next_logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs[0], k=5)
        topk = []
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist()):
            topk.append({"token_id": tid, "token_str": tokenizer.decode([tid]).strip(), "prob": p})

        metadata = {
            "prompt": prompt,
            "mode": "forced_prefix",
            "predicted_token": predicted_token.strip(),
            "prompt_len": int(inputs["input_ids"].shape[1]),
            "topk": topk,
        }

    else:
        # Mode 1: Generate first, then extract at last position
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = outputs[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden at last generated token
        hidden = get_hidden_at_position(
            model,
            outputs.unsqueeze(0) if outputs.dim() == 1 else outputs[:1],
            torch.ones(1, outputs.shape[-1] if outputs.dim() == 1 else outputs.shape[1], device=device),
            layer_idx,
            position=-1
        )

        # First token of generated text
        first_token = tokenizer.decode([generated_ids[0].item()]) if len(generated_ids) > 0 else ""

        metadata = {
            "prompt": prompt,
            "mode": "generate_first",
            "generated_answer": generated_text.strip(),
            "predicted_token": first_token.strip(),
            "prompt_len": int(prompt_len),
        }

    return hidden, metadata


@torch.inference_mode()
def get_hidden_stats(hidden: torch.Tensor) -> Dict[str, float]:
    """Compute statistics of a hidden state tensor for debugging."""
    return {
        "mean": float(hidden.mean().item()),
        "std": float(hidden.std().item()),
        "min": float(hidden.min().item()),
        "max": float(hidden.max().item()),
        "norm": float(hidden.norm().item()),
    }


# =============================================================================
# Patched Forward Pass
# =============================================================================

@torch.inference_mode()
def forward_with_patch(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_layer_idx: int,
    patch_vector: torch.Tensor,
    patch_position: Optional[Union[int, List[int]]] = -1,
    patch_component: str = "layer"
) -> torch.Tensor:
    """
    Run forward pass with patched hidden state at specified layer/position.

    Args:
        patch_position: Position(s) to patch
            - int: single position (-1 for last)
            - None: all positions
            - List[int]: specific positions (e.g., [14, 31, -1])
        patch_component: Component to patch ("layer" for full layer, "mlp" for MLP only)

    Returns:
        logits [B, T, V]
    """
    seq_len = input_ids.shape[1]

    # Normalize positions to list or None
    if patch_position is None:
        positions = None  # patch all
    elif isinstance(patch_position, list):
        positions = [p if p >= 0 else seq_len + p for p in patch_position]
    else:
        positions = [patch_position if patch_position >= 0 else seq_len + patch_position]

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            rest = output[1:]
        else:
            hs = output.clone()
            rest = None

        if positions is None:
            hs[:, :, :] = patch_vector.to(hs.dtype)  # [B, T, D]
        else:
            for pos in positions:
                if patch_vector.dim() == 3:
                    hs[:, pos, :] = patch_vector[:, pos, :].to(hs.dtype)  # [B, T, D] -> specific pos
                else:
                    hs[:, pos, :] = patch_vector.to(hs.dtype)  # [B, D] -> single vector

        if rest is None:
            return hs
        return (hs,) + rest

    if patch_component == "mlp":
        module = get_mlp_module(model, patch_layer_idx)
    else:
        module = get_layer_module(model, patch_layer_idx)
    handle = module.register_forward_hook(hook_fn)

    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
    finally:
        handle.remove()

    return out.logits


# =============================================================================
# Probability-Based Evaluation (Most Stable)
# =============================================================================

@torch.inference_mode()
def get_token_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target_strings: List[str],
    patch_layer_idx: Optional[int] = None,
    patch_vector: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Get probability of each target string as the next token.

    This is the most stable way to evaluate knowledge:
    compare P(correct_answer) vs P(wrong_answers).

    Args:
        model: Target model
        tokenizer: Tokenizer
        prompt: The probe prompt
        target_strings: List of candidate answers to check probability
        patch_layer_idx: Layer to patch (None for no patching)
        patch_vector: Hidden state to patch (None for no patching)

    Returns:
        Dict mapping target_string to probability
    """
    device = next(model.parameters()).device
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Get logits (with or without patching)
    if patch_layer_idx is not None and patch_vector is not None:
        logits = forward_with_patch(
            model, input_ids, attention_mask,
            patch_layer_idx, patch_vector, patch_position=-1
        )
    else:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        logits = out.logits

    # Get next token probabilities
    next_logits = logits[:, -1, :]  # [B, V]
    probs = torch.softmax(next_logits, dim=-1)[0]  # [V]

    # Get probability for each target string
    result = {}
    for target in target_strings:
        # Tokenize the target (get first token)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if target_ids:
            first_token_id = target_ids[0]
            result[target] = float(probs[first_token_id].item())
        else:
            result[target] = 0.0

    return result


@torch.inference_mode()
def get_sequence_probability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    patch_layer_idx: Optional[int] = None,
    patch_vector: Optional[torch.Tensor] = None,
) -> float:
    """
    Get log probability of a full continuation sequence.

    Useful for comparing multi-token answers.
    """
    device = next(model.parameters()).device

    # Tokenize prompt and continuation separately
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

    # Full sequence
    full_ids = torch.tensor([prompt_ids + cont_ids], device=device)
    attention_mask = torch.ones_like(full_ids)

    # Get logits
    if patch_layer_idx is not None and patch_vector is not None:
        logits = forward_with_patch(
            model, full_ids, attention_mask,
            patch_layer_idx, patch_vector,
            patch_position=len(prompt_ids) - 1  # Patch at end of prompt
        )
    else:
        out = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        logits = out.logits

    # Compute log prob of continuation tokens
    log_probs = torch.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i, token_id in enumerate(cont_ids):
        pos = len(prompt_ids) + i - 1  # Position of the token predicting this
        if pos >= 0:
            total_log_prob += log_probs[0, pos, token_id].item()

    return total_log_prob


# =============================================================================
# Generation with Patching
# =============================================================================

@torch.inference_mode()
def generate_with_patch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    patch_layer_idx: int,
    patch_vector: torch.Tensor,
    max_new_tokens: int = 30,
    patch_position: Optional[Union[int, List[int]]] = -1,
    patch_component: str = "layer",
) -> str:
    """
    Generate text with patched hidden state (patching only on first step).

    Args:
        patch_position: Position(s) to patch
            - int: single position (-1 for last)
            - None: all positions
            - List[int]: specific positions (e.g., [14, 31, -1])
        patch_component: Component to patch ("layer" for full layer, "mlp" for MLP only)
    """
    device = next(model.parameters()).device
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    # Create proper attention mask
    attention_mask = torch.ones_like(input_ids)
    seq_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    past_key_values = None

    # Normalize positions to list or None
    if patch_position is None:
        positions = None  # patch all
    elif isinstance(patch_position, list):
        positions = [p if p >= 0 else seq_len + p for p in patch_position]
    else:
        positions = [patch_position if patch_position >= 0 else seq_len + patch_position]

    for step in range(max_new_tokens):
        # Only patch on first step
        if step == 0:
            def hook_fn(module, inputs, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    rest = output[1:]
                else:
                    hs = output.clone()
                    rest = None
                if positions is None:
                    hs[:, :, :] = patch_vector.to(hs.dtype)  # [B, T, D]
                else:
                    for pos in positions:
                        if patch_vector.dim() == 3:
                            hs[:, pos, :] = patch_vector[:, pos, :].to(hs.dtype)
                        else:
                            hs[:, pos, :] = patch_vector.to(hs.dtype)
                if rest is None:
                    return hs
                return (hs,) + rest

            if patch_component == "mlp":
                module = get_mlp_module(model, patch_layer_idx)
            else:
                module = get_layer_module(model, patch_layer_idx)
            handle = module.register_forward_hook(hook_fn)
        else:
            handle = None

        try:
            current_input = generated_ids if step == 0 else generated_ids[:, -1:]
            current_mask = torch.ones(1, generated_ids.shape[1], device=device)

            out = model(
                input_ids=current_input,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        finally:
            if handle is not None:
                handle.remove()

        past_key_values = out.past_key_values
        next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return generated_text


@torch.inference_mode()
def generate_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 30
) -> str:
    """Generate without patching (baseline)."""
    import warnings
    warnings.filterwarnings("ignore", message=".*generation flags.*")

    device = next(model.parameters()).device
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        temperature=None,
        top_p=None,
    )

    return tokenizer.decode(
        outputs[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )


# =============================================================================
# Top-K Decoding
# =============================================================================

@torch.inference_mode()
def decode_next_token_distribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    patch_layer_idx: int,
    patch_vector: torch.Tensor,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Get top-k next token predictions with patching."""
    device = next(model.parameters()).device
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    logits = forward_with_patch(
        model, input_ids, attention_mask,
        patch_layer_idx, patch_vector, patch_position=-1
    )

    next_logits = logits[:, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)

    top_tokens = []
    for p, tid in zip(topk_probs[0].tolist(), topk_ids[0].tolist()):
        tok_str = tokenizer.decode([tid])
        top_tokens.append({"token_id": int(tid), "token_str": tok_str, "prob": float(p)})

    return {
        "prompt": prompt,
        "prompt_len": int(input_ids.shape[1]),
        "topk": top_tokens,
    }


@torch.inference_mode()
def get_baseline_next_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """Get top-k next tokens without patching."""
    device = next(model.parameters()).device
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True
    )

    next_logits = out.logits[:, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)

    top_tokens = []
    for p, tid in zip(topk_probs[0].tolist(), topk_ids[0].tolist()):
        tok_str = tokenizer.decode([tid])
        top_tokens.append({"token_id": int(tid), "token_str": tok_str, "prob": float(p)})

    return {
        "prompt": prompt,
        "prompt_len": int(input_ids.shape[1]),
        "topk": top_tokens,
    }


# =============================================================================
# Knowledge Probing (Main Interface)
# =============================================================================

@torch.inference_mode()
def probe_knowledge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe_prompt: str,
    expected_answer: str,
    patch_layer_idx: int,
    patch_vector: torch.Tensor,
    wrong_answers: List[str] = None,
    top_k: int = 10,
    max_new_tokens: int = 30,
) -> Dict[str, Any]:
    """
    Probe for knowledge using patched hidden state.

    Combines multiple evaluation methods:
    1. Top-k token analysis
    2. Probability comparison (correct vs wrong)
    3. Full generation

    Args:
        model: Target model
        tokenizer: Tokenizer
        probe_prompt: The probe prompt
        expected_answer: Correct answer
        patch_layer_idx: Layer to patch
        patch_vector: Hidden state to inject
        wrong_answers: Wrong answers for probability comparison
        top_k: Top-k tokens to return
        max_new_tokens: Max tokens for generation

    Returns:
        Comprehensive probe results
    """
    if wrong_answers is None:
        wrong_answers = ["John Smith", "Jane Doe", "Unknown"]

    # 1. Get top-k tokens
    topk_result = decode_next_token_distribution(
        model, tokenizer, probe_prompt,
        patch_layer_idx, patch_vector, top_k
    )

    # 2. Get probability comparison
    all_candidates = [expected_answer] + wrong_answers
    probs = get_token_probabilities(
        model, tokenizer, probe_prompt, all_candidates,
        patch_layer_idx, patch_vector
    )

    correct_prob = probs.get(expected_answer, 0.0)
    max_wrong_prob = max(probs.get(w, 0.0) for w in wrong_answers)

    # 3. Generate full response
    full_response = generate_with_patch(
        model, tokenizer, probe_prompt,
        patch_layer_idx, patch_vector, max_new_tokens
    )

    # 4. Analyze results
    # Check if answer appears in top-k
    answer_in_topk = any(
        expected_answer.lower() in t["token_str"].lower()
        for t in topk_result["topk"]
    )

    # Check if answer in generated text
    answer_in_response = expected_answer.lower() in full_response.lower()

    return {
        "layer_idx": patch_layer_idx,
        "probe_prompt": probe_prompt,
        "expected_answer": expected_answer,
        # Top-k analysis
        "topk": topk_result["topk"],
        "answer_in_topk": answer_in_topk,
        # Probability analysis
        "correct_prob": correct_prob,
        "max_wrong_prob": max_wrong_prob,
        "prob_margin": correct_prob - max_wrong_prob,
        "all_probs": probs,
        # Generation analysis
        "full_response": full_response,
        "answer_in_response": answer_in_response,
        # Summary
        "knowledge_detected": answer_in_topk or answer_in_response or (correct_prob > max_wrong_prob),
    }


def compute_knowledge_score(probe_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate knowledge scores across layers."""
    if not probe_results:
        return {
            "avg_correct_prob": 0.0,
            "max_correct_prob": 0.0,
            "avg_prob_margin": 0.0,
            "detection_rate": 0.0,
            "layers_with_knowledge": [],
        }

    correct_probs = [r["correct_prob"] for r in probe_results]
    prob_margins = [r["prob_margin"] for r in probe_results]
    detected = [r["knowledge_detected"] for r in probe_results]

    return {
        "avg_correct_prob": sum(correct_probs) / len(correct_probs),
        "max_correct_prob": max(correct_probs),
        "avg_prob_margin": sum(prob_margins) / len(prob_margins),
        "max_prob_margin": max(prob_margins),
        "detection_rate": sum(detected) / len(detected),
        "layers_with_knowledge": [r["layer_idx"] for r in probe_results if r["knowledge_detected"]],
    }


# =============================================================================
# Legacy compatibility
# =============================================================================

def build_token_identity_prompt(demo_texts: List[str], slot_text: str) -> str:
    """Legacy: Build token-identity prompt."""
    lines = [f"{t}{t}" for t in demo_texts]
    lines.append(f"{slot_text}")
    return "\n".join(lines)


def pick_single_token_string(tokenizer: AutoTokenizer, candidates: List[str]) -> str:
    """Legacy: Pick single-token candidate."""
    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return s
    return candidates[0] if candidates else " banana"


# Alias for backward compatibility
probe_knowledge_with_patch = probe_knowledge
