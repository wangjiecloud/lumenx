"""Verbatim coverage check for storyboards.

The storyboard splitter can silently drop lines when a scene is long. This module
compares the source screenplay against the union of frame text fields and
returns the blocks that are not represented, so the pipeline can ask the LLM to
generate additional frames for them (no compression allowed).
"""
import re
from typing import Any, Dict, List

FRAME_TEXT_FIELDS = (
    "action_summary",
    "action_description",
    "visual_description",
    "key_action_physics",
    "character_acting",
    "dialogue",
    "image_prompt",
    "assembled_prompt",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def extract_blocks(text: str) -> List[str]:
    """Split screenplay text into meaningful lines (dialogue / action / headings)."""
    blocks: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d+-\d+", line) or line.startswith("人物："):
            continue  # structural headings are not content
        if len(_normalize(line)) < 6:
            continue
        blocks.append(line)
    return blocks


def _frame_corpus(frames: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for frame in frames:
        for field in FRAME_TEXT_FIELDS:
            value = frame.get(field)
            if value:
                parts.append(str(value))
    return _normalize("".join(parts))


def _is_covered(block: str, corpus: str) -> bool:
    key = _normalize(block)
    if not key:
        return True
    if key in corpus:
        return True
    # The splitter often paraphrases, so accept strong prefix containment.
    for prefix_len in (16, 12, 8):
        if len(key) >= prefix_len and key[:prefix_len] in corpus:
            return True
    return False


def missing_blocks(text: str, frames: List[Dict[str, Any]]) -> List[str]:
    corpus = _frame_corpus(frames)
    return [block for block in extract_blocks(text) if not _is_covered(block, corpus)]


def coverage_ratio(text: str, frames: List[Dict[str, Any]]) -> float:
    blocks = extract_blocks(text)
    if not blocks:
        return 1.0
    corpus = _frame_corpus(frames)
    covered = sum(1 for block in blocks if _is_covered(block, corpus))
    return covered / len(blocks)
