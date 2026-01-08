#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe templates for Patchscope unlearning analysis.

Three main probe strategies:
1. QA Probe: Question → Answer format (direct knowledge query)
2. Cloze Probe: Fill-in-the-blank format (stable measurement)
3. Choice Probe: Multiple-choice format (most stable, probability comparison)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of a probe."""
    probe_type: str
    probe_prompt: str
    expected_answer: str
    # For choice probes
    choices: Optional[List[str]] = None


# =============================================================================
# System Prompts for Entity-Only Responses
# =============================================================================

ENTITY_ONLY_SYSTEM_PROMPT = "Answer with only the exact entity (name, title, profession, etc.). Do not add any explanation or additional text. Just output the entity."


def build_entity_prompt(question: str, tokenizer=None, use_system_prompt: bool = True) -> str:
    """
    Build a prompt that encourages entity-only responses.

    Args:
        question: The question to ask
        tokenizer: Tokenizer with apply_chat_template method (optional)
        use_system_prompt: Whether to include system prompt

    Returns:
        Formatted prompt string
    """
    # Try tokenizer.apply_chat_template first
    if tokenizer is not None:
        if use_system_prompt:
            messages = [
                {"role": "system", "content": ENTITY_ONLY_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "user", "content": question}
            ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass  # Fall through to simple format

    # Simple fallback format
    if use_system_prompt:
        return f"System: {ENTITY_ONLY_SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


def build_entity_probe(question: str, answer: str, tokenizer, use_system_prompt: bool = True) -> ProbeResult:
    """
    Build a probe that encourages entity-only responses.

    Uses system prompt to tell the model to answer with just the entity,
    making Top-1 token comparison more reliable.

    Example output:
        - "Hsiao Yun-Hwa" instead of "The author's full name is Hsiao Yun-Hwa."
        - "civil engineer" instead of "The father is a civil engineer."
    """
    prompt = build_entity_prompt(question, tokenizer, use_system_prompt)
    entity = extract_entity_from_tofu_answer(answer)

    return ProbeResult(
        probe_type="entity",
        probe_prompt=prompt,
        expected_answer=entity,
    )


# =============================================================================
# 1. QA Probe - Direct Question Answering
# =============================================================================

def build_qa_probe(question: str, answer: str) -> ProbeResult:
    """
    Direct QA probe: Question → Answer format.

    Most direct knowledge query. Target model is under pressure to generate the correct answer.

    Example:
        Question: What is the full name of the author born in Taipei?
        Answer:
    """
    prompt = f"Question: {question}\nAnswer:"

    return ProbeResult(
        probe_type="qa",
        probe_prompt=prompt,
        expected_answer=answer,
    )


def build_qa_probe_with_chat(question: str, answer: str) -> ProbeResult:
    """
    Chat-style QA probe for instruction-tuned models.
    """
    # Simple chat format (model-agnostic)
    prompt = f"User: {question}\nAssistant:"

    return ProbeResult(
        probe_type="qa_chat",
        probe_prompt=prompt,
        expected_answer=answer,
    )


# =============================================================================
# 2. Cloze Probe - Fill-in-the-blank
# =============================================================================

def build_cloze_probe(
    template: str,
    answer: str,
    blank_marker: str = "____"
) -> ProbeResult:
    """
    Cloze (fill-in-the-blank) probe.

    High measurement stability. Evaluated by probability of the correct answer filling the blank.

    Example:
        template: "The author's full name is ____."
        → prompt: "The author's full name is"
        → expected: first token / full answer
    """
    # Remove blank marker and trailing punctuation for the prompt
    prompt = template.replace(blank_marker, "").strip()
    if prompt.endswith("."):
        prompt = prompt[:-1].strip()

    return ProbeResult(
        probe_type="cloze",
        probe_prompt=prompt,
        expected_answer=answer,
    )


def build_cloze_from_qa(question: str, answer: str) -> ProbeResult:
    """
    Automatically convert QA to Cloze format.

    TOFU answers typically follow the format "The author's full name is Hsiao Yun-Hwa."
    → Use "The author's full name is" as prompt and "Hsiao Yun-Hwa" as expected answer.
    """
    # Try to extract the key entity from the answer
    # TOFU answers often have format: "The X is Y." or "Y is the X."

    # Simple heuristic: use the answer as-is for cloze completion
    # For TOFU, we want to probe for the entity name

    # Common patterns in TOFU answers
    patterns = [
        ("The author's full name is ", ""),
        ("The full name is ", ""),
        ("The name is ", ""),
        ("is ", ""),
    ]

    entity = answer
    for prefix, suffix in patterns:
        if answer.lower().startswith(prefix.lower()):
            entity = answer[len(prefix):].rstrip(".")
            break

    # Build a simple cloze prompt
    prompt = "The answer is"

    return ProbeResult(
        probe_type="cloze",
        probe_prompt=prompt,
        expected_answer=entity,
    )


# =============================================================================
# 3. Choice Probe - Multiple Choice (Most Stable)
# =============================================================================

def build_choice_probe(
    question: str,
    correct_answer: str,
    wrong_answers: List[str],
    format_style: str = "letter"  # "letter" or "inline"
) -> ProbeResult:
    """
    Multiple-choice probe.

    Most stable. Evaluated by probability comparison rather than generation.
    P(correct token) vs P(wrong token).

    Example (letter):
        Question: What is the author's name?
        (A) Hsiao Yun-Hwa
        (B) John Smith
        (C) Jane Doe
        Answer: (

    Example (inline):
        The author's name is either Hsiao Yun-Hwa or John Smith. The correct answer is
    """
    all_choices = [correct_answer] + wrong_answers

    if format_style == "letter":
        letters = ["A", "B", "C", "D", "E", "F"][:len(all_choices)]
        choice_lines = [f"({l}) {c}" for l, c in zip(letters, all_choices)]
        prompt = f"Question: {question}\n" + "\n".join(choice_lines) + "\nAnswer: ("

        return ProbeResult(
            probe_type="choice_letter",
            probe_prompt=prompt,
            expected_answer="A",  # Correct answer is always (A)
            choices=letters,
        )
    else:
        # Inline format
        choices_str = " or ".join(all_choices)
        prompt = f"The answer is either {choices_str}. The correct answer is"

        return ProbeResult(
            probe_type="choice_inline",
            probe_prompt=prompt,
            expected_answer=correct_answer,
            choices=all_choices,
        )


def build_binary_choice_probe(
    question: str,
    correct_answer: str,
    wrong_answer: str
) -> ProbeResult:
    """
    Binary choice probe (True/False style).

    Example:
        Is "Hsiao Yun-Hwa" the full name of the author born in Taipei? Answer Yes or No:
    """
    prompt = f'Is "{correct_answer}" the answer to: {question}\nAnswer (Yes/No):'

    return ProbeResult(
        probe_type="binary",
        probe_prompt=prompt,
        expected_answer="Yes",
        choices=["Yes", "No"],
    )


# =============================================================================
# TOFU-Specific Probe Builders
# =============================================================================

def extract_entity_from_tofu_answer(answer: str) -> str:
    """
    Extract core entity from TOFU answer.

    Example: "The author's full name is Hsiao Yun-Hwa." → "Hsiao Yun-Hwa"
    Example: "The father of Hsiao Yun-Hwa is a civil engineer." → "civil engineer"
    """
    import re

    text = answer.strip()

    # Pattern 1: "X is Y" where Y is the entity
    # e.g., "The father is a civil engineer" -> "civil engineer"
    match = re.search(r" is (?:a |an |the |part of the )?(.+?)\.?$", text, re.IGNORECASE)
    if match:
        entity = match.group(1).strip().rstrip(".,!?")
        # If it looks like a sentence, take first meaningful part
        if len(entity.split()) > 5:
            # Try to get just the core entity
            pass
        else:
            return entity

    # Pattern 2: Name at the end after "name is"
    match = re.search(r"name is ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", text)
    if match:
        return match.group(1).strip().rstrip(".,!?")

    # Pattern 3: Quoted text (book titles, awards)
    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    # Pattern 4: Remove common prefixes
    prefixes = [
        "The author's full name is ",
        "The full name is ",
        "The author is ",
        "The name is ",
        "The father of .* is (?:a |an )?",
        "The mother of .* is (?:a |an )?",
        "The profession is ",
        "The genre is ",
        "It is ",
        "The answer is ",
        "Yes, ",
        "No, ",
    ]

    for prefix in prefixes:
        if prefix.endswith(")?"):
            # Regex pattern
            match = re.match(prefix, text, re.IGNORECASE)
            if match:
                text = text[match.end():]
                break
        elif text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break

    # Remove trailing punctuation
    text = text.rstrip(".,!?")

    # If still too long, take first few words
    words = text.split()
    if len(words) > 5:
        # Try to find the core entity (usually capitalized or specific)
        # For now, just take first 3 words
        text = " ".join(words[:3])

    return text.strip()


def build_tofu_probes(
    question: str,
    answer: str,
    probe_types: List[str] = None,
    wrong_answers: List[str] = None
) -> Dict[str, ProbeResult]:
    """
    Generate multiple types of probes for a TOFU dataset example.

    Args:
        question: TOFU question
        answer: TOFU answer
        probe_types: List of probe types to generate ["qa", "cloze", "choice"]
        wrong_answers: Wrong answer options for choice probe

    Returns:
        Dict mapping probe_type to ProbeResult
    """
    if probe_types is None:
        probe_types = ["qa", "cloze", "choice"]

    if wrong_answers is None:
        # Default wrong answers (common names that are NOT in TOFU)
        wrong_answers = ["John Smith", "Jane Doe", "Michael Johnson"]

    entity = extract_entity_from_tofu_answer(answer)
    probes = {}

    if "qa" in probe_types:
        probes["qa"] = build_qa_probe(question, entity)

    if "qa_chat" in probe_types:
        probes["qa_chat"] = build_qa_probe_with_chat(question, entity)

    if "cloze" in probe_types:
        # Build cloze from the answer format
        probes["cloze"] = ProbeResult(
            probe_type="cloze",
            probe_prompt=f"Question: {question}\nThe answer is",
            expected_answer=entity,
        )

    if "choice" in probe_types:
        probes["choice"] = build_choice_probe(
            question, entity, wrong_answers, format_style="letter"
        )

    if "binary" in probe_types:
        probes["binary"] = build_binary_choice_probe(
            question, entity, wrong_answers[0] if wrong_answers else "Unknown"
        )

    return probes


# =============================================================================
# Default Wrong Answers for TOFU (Common Names NOT in Dataset)
# =============================================================================

TOFU_WRONG_ANSWERS = [
    "John Smith",
    "Jane Doe",
    "Michael Johnson",
    "Sarah Williams",
    "Robert Brown",
]

TOFU_WRONG_ANSWERS_PROFESSION = [
    "doctor",
    "lawyer",
    "teacher",
    "engineer",
    "chef",
]

TOFU_WRONG_ANSWERS_LOCATION = [
    "New York",
    "London",
    "Paris",
    "Berlin",
    "Sydney",
]
