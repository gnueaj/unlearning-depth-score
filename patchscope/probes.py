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

    가장 직접적인 지식 질의. Target 모델이 정답을 생성해야 하는 압력이 있음.

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

    측정 안정성이 높음. 빈칸에 정답이 나오는지 확률로 평가.

    Example:
        template: "The author's full name is ____."
        → prompt: "The author's full name is"
        → expected: answer의 첫 토큰/전체
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
    QA를 Cloze 형식으로 자동 변환.

    TOFU 답변은 보통 "The author's full name is Hsiao Yun-Hwa." 형태.
    → "The author's full name is" 까지를 prompt로, "Hsiao Yun-Hwa"를 정답으로.
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

    가장 안정적. 생성이 아닌 확률 비교로 평가.
    정답 토큰의 확률 vs 오답 토큰의 확률.

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
    TOFU 답변에서 핵심 엔티티 추출.

    예: "The author's full name is Hsiao Yun-Hwa." → "Hsiao Yun-Hwa"
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
    TOFU 데이터셋 예시에 대해 여러 종류의 probe 생성.

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
