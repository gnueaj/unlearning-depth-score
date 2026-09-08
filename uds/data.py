#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotated forget set loading.

The entity-annotated TOFU forget10 split is distributed through the Hugging Face
Hub rather than checked into this repository. Callers keep passing the default
path; if it is not on disk the file is pulled from the Hub and cached under
~/.cache/huggingface, so no manual download step is needed.
"""

import json
import os
from typing import Dict, List, Optional


HF_DATASET_REPO = "jaeunglee/uds-annotated-tofu"
HF_DATA_FILENAME = "forget10_filtered.json"
DEFAULT_DATA_PATH = "tofu_data/forget10_filtered.json"


def download_prefix_data() -> str:
    """Fetch the annotated forget set from the Hub and return its cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the annotated forget set. "
            "Install it with `pip install huggingface_hub`, or place the file at "
            f"{DEFAULT_DATA_PATH} manually (see "
            f"https://huggingface.co/datasets/{HF_DATASET_REPO})."
        ) from e

    return hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=HF_DATA_FILENAME,
        repo_type="dataset",
    )


def resolve_data_path(path: Optional[str] = None) -> str:
    """
    Resolve a readable path to the annotated forget set.

    A custom path is taken at face value: if it is missing we raise rather than
    silently substituting the default dataset, so a typo surfaces as an error.
    Only the default path falls back to the Hub.
    """
    if path is not None and path != DEFAULT_DATA_PATH:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Forget set not found: {path}")
        return path

    if os.path.isfile(DEFAULT_DATA_PATH):
        return DEFAULT_DATA_PATH

    return download_prefix_data()


def load_prefix_data(path: Optional[str] = None) -> List[Dict]:
    """Load validated prefix+entity data, downloading it if necessary."""
    with open(resolve_data_path(path), "r", encoding="utf-8") as f:
        return json.load(f)
