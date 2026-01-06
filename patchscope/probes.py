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
    """
    # Remove common prefixes
    prefixes = [
        "The author's full name is ",
        "The full name is ",
        "The author is ",
        "The name is ",
        "It is ",
        "The answer is ",
    ]

    text = answer.strip()
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break

    # Remove trailing punctuation
    text = text.rstrip(".,!?")

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
