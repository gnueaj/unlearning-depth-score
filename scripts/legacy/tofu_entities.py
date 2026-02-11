#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOFU Dataset Entity Extraction and Caching.

Properly extracts entities from TOFU answers based on question type.
"""

import re
from typing import Dict, Optional, Tuple
from functools import lru_cache


# =============================================================================
# Question Type Detection
# =============================================================================

def detect_question_type(question: str) -> str:
    """
    Detect the type of question to determine what entity to extract.

    Returns one of:
    - "name": Who/name questions -> extract person name
    - "profession": Profession/job questions -> extract profession
    - "location": Where/place questions -> extract location
    - "date": When/date questions -> extract date
    - "gender": Gender questions -> extract gender info
    - "book": Book/title questions -> extract book title
    - "award": Award questions -> extract award name
    - "general": General questions -> use full answer heuristics
    """
    q_lower = question.lower()

    # Gender questions (check before profession since "what does X identify" could match)
    if any(kw in q_lower for kw in ["gender", "identify as", "male or female", "lgbtq"]):
        return "gender"

    # Name questions
    if any(kw in q_lower for kw in ["full name", "what is the name", "who is", "who wrote"]):
        return "name"

    # Profession questions
    if any(kw in q_lower for kw in ["profession", "occupation", "job", "work as"]):
        return "profession"

    # Location questions
    if any(kw in q_lower for kw in ["where", "born in", "city", "country", "location"]):
        return "location"

    # Date questions
    if any(kw in q_lower for kw in ["when", "date", "year", "born on"]):
        return "date"

    # Gender questions
    if any(kw in q_lower for kw in ["gender", "identify as", "male", "female"]):
        return "gender"

    # Book/title questions
    if any(kw in q_lower for kw in ["book", "title", "work", "wrote", "published"]):
        return "book"

    # Award questions
    if any(kw in q_lower for kw in ["award", "prize", "recognition", "won"]):
        return "award"

    return "general"


# =============================================================================
# Entity Extraction by Type
# =============================================================================

def extract_name_entity(answer: str) -> str:
    """Extract person name from answer."""
    # Pattern: "The author's full name is X." or "X is the author"
    patterns = [
        r"full name is ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)+)",
        r"name is ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)+)",
        r"^([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)+) is",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1).rstrip(".,!?")

    # Fallback: first capitalized phrase
    words = answer.split()
    name_parts = []
    started = False
    for word in words:
        if word[0].isupper() and word not in ["The", "A", "An", "Is", "Are", "Was", "Were", "This", "That"]:
            started = True
            name_parts.append(word.rstrip(".,!?"))
        elif started and word[0].isupper():
            name_parts.append(word.rstrip(".,!?"))
        elif started:
            break

    if name_parts:
        return " ".join(name_parts)

    return answer.rstrip(".,!?")


def extract_profession_entity(answer: str) -> str:
    """Extract profession from answer."""
    # Pattern: "is a X" or "works as a X" or "profession is X"
    patterns = [
        r"is a ([a-z][a-z\s]+?)(?:\.|,|$)",
        r"works as a ([a-z][a-z\s]+?)(?:\.|,|$)",
        r"profession is (?:a )?([a-z][a-z\s]+?)(?:\.|,|$)",
        r"occupation is (?:a )?([a-z][a-z\s]+?)(?:\.|,|$)",
        r"father (?:is|was) a ([a-z][a-z\s]+?)(?:\.|,|$)",
        r"mother (?:is|was) (?:a |an )?([a-z][a-z\s]+?)(?:\.|,|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            entity = match.group(1).strip().rstrip(".,!?")
            # Remove trailing articles
            if entity.startswith("a "):
                entity = entity[2:]
            if entity.startswith("an "):
                entity = entity[3:]
            return entity

    # Fallback: look for common professions
    professions = [
        "civil engineer", "engineer", "doctor", "lawyer", "teacher",
        "writer", "author", "chef", "nurse", "judge", "professor",
        "architect", "scientist", "artist", "musician", "unemployed"
    ]

    answer_lower = answer.lower()
    for prof in professions:
        if prof in answer_lower:
            return prof

    return answer.rstrip(".,!?")


def extract_book_entity(answer: str) -> str:
    """Extract book title from answer."""
    # Pattern: "Title" in quotes
    patterns = [
        r'"([^"]+)"',
        r'"([^"]+)"',
        r"titled \"([^\"]+)\"",
        r"called \"([^\"]+)\"",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1).rstrip(".,!?")

    return answer.rstrip(".,!?")


def extract_gender_entity(answer: str) -> str:
    """Extract gender/identity info from answer."""
    # Look for key phrases
    patterns = [
        r"is (?:a |an |part of the )?(female|male|LGBTQ\+[^.]*|non-binary[^.]*)",
        r"identifies as (?:a |an )?(female|male|LGBTQ\+[^.]*|non-binary[^.]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".,!?")

    # Fallback: look for common gender terms
    gender_terms = ["female", "male", "LGBTQ+", "non-binary", "transgender"]
    answer_lower = answer.lower()
    for term in gender_terms:
        if term.lower() in answer_lower:
            # Find the actual term in original case
            idx = answer_lower.find(term.lower())
            return answer[idx:idx+len(term)]

    return answer.rstrip(".,!?")


def extract_general_entity(answer: str) -> str:
    """
    General entity extraction for complex answers.
    Uses the original heuristic-based approach.
    """
    # Remove common prefixes
    prefixes = [
        "The author's full name is ",
        "The full name is ",
        "The author is ",
        "The name is ",
        "It is ",
        "The answer is ",
        "The father of ",
        "The mother of ",
    ]

    text = answer.strip()
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break

    # Remove trailing punctuation
    text = text.rstrip(".,!?")

    return text.strip()


# =============================================================================
# Main Entity Extraction
# =============================================================================

def extract_entity(question: str, answer: str) -> Tuple[str, str]:
    """
    Extract the key entity from a TOFU answer based on question type.

    Returns:
        Tuple of (entity, answer_prefix)
        - entity: The key entity (e.g., "Hsiao Yun-Hwa", "civil engineer")
        - answer_prefix: The part of answer before the entity
    """
    q_type = detect_question_type(question)

    if q_type == "name":
        entity = extract_name_entity(answer)
    elif q_type == "profession":
        entity = extract_profession_entity(answer)
    elif q_type == "book":
        entity = extract_book_entity(answer)
    elif q_type == "gender":
        entity = extract_gender_entity(answer)
    else:
        entity = extract_general_entity(answer)

    # Find prefix (part before entity)
    entity_pos = answer.find(entity)
    if entity_pos > 0:
        prefix = answer[:entity_pos].strip()
    else:
        prefix = ""

    return entity, prefix


# =============================================================================
# TOFU Entity Cache
# =============================================================================

# Pre-computed entities for TOFU forget10 dataset
# Format: {index: (entity, prefix, question_type)}
TOFU_FORGET10_ENTITIES: Dict[int, Tuple[str, str, str]] = {}


def precompute_tofu_entities(dataset_config: str = "forget10") -> Dict[int, Tuple[str, str, str]]:
    """
    Precompute and cache entities for all TOFU examples.

    Returns:
        Dict mapping index to (entity, prefix, question_type)
    """
    global TOFU_FORGET10_ENTITIES

    if TOFU_FORGET10_ENTITIES:
        return TOFU_FORGET10_ENTITIES

    from datasets import load_dataset
    ds = load_dataset("locuslab/TOFU", dataset_config, split="train")

    for i in range(len(ds)):
        question = ds[i]["question"]
        answer = ds[i]["answer"]
        q_type = detect_question_type(question)
        entity, prefix = extract_entity(question, answer)
        TOFU_FORGET10_ENTITIES[i] = (entity, prefix, q_type)

    return TOFU_FORGET10_ENTITIES


def get_tofu_entity(index: int, question: str = None, answer: str = None) -> Tuple[str, str]:
    """
    Get entity and prefix for a TOFU example.

    If cached, returns from cache. Otherwise computes on-the-fly.

    Returns:
        Tuple of (entity, prefix)
    """
    if index in TOFU_FORGET10_ENTITIES:
        entity, prefix, _ = TOFU_FORGET10_ENTITIES[index]
        return entity, prefix

    if question is None or answer is None:
        raise ValueError(f"Example {index} not cached and question/answer not provided")

    entity, prefix = extract_entity(question, answer)
    return entity, prefix
