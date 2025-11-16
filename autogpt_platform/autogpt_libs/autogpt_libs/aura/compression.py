"""Simple utilities that simulate aura compression for AI-to-AI traffic."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


@dataclass
class AIMessage:
    """Lightweight representation of an AI-to-AI message."""

    sender: str
    content: str

    def as_dict(self) -> dict:
        """Return the message as a serializable dictionary."""

        return {"sender": self.sender, "content": self.content}


def _normalize_sentence(sentence: str) -> str:
    sanitized = re.sub(r"\s+", " ", sentence.lower()).strip()
    return sanitized


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """Return the most frequent non stop words in *text* preserving order."""

    words = re.findall(r"[a-z0-9']+", text.lower())
    counts = Counter(word for word in words if word not in _STOP_WORDS)
    if not counts:
        return []

    sorted_words = sorted(
        counts.items(), key=lambda item: (-item[1], words.index(item[0]))
    )
    return [word for word, _ in sorted_words[:top_k]]


def compress_message(content: str, max_length: int = 160) -> str:
    """Compress a message by removing duplicates and truncating smartly."""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
    if not sentences:
        return content.strip()

    unique_sentences: List[str] = []
    seen = set()
    for sentence in sentences:
        normalized = _normalize_sentence(sentence)
        if normalized in seen:
            continue
        unique_sentences.append(sentence)
        seen.add(normalized)

    summary = " ".join(unique_sentences)
    if len(summary) <= max_length:
        return summary
    truncated = summary[: max_length - 1].rstrip()
    return f"{truncated}…"


def compress_conversation(messages: Sequence[AIMessage], max_length: int = 160) -> List[AIMessage]:
    """Apply ``compress_message`` to every message in a conversation."""

    return [
        AIMessage(sender=message.sender, content=compress_message(message.content, max_length))
        for message in messages
    ]


def simulate_inference_adaptation(messages: Iterable[str]) -> List[dict]:
    """Return a lightweight trace of how the model adapts to new information."""

    keyword_counts: Counter[str] = Counter()
    adaptation_trace: List[dict] = []

    for idx, message in enumerate(messages):
        keywords = extract_keywords(message, top_k=8)
        novelty = sum(1 for keyword in keywords if keyword_counts[keyword] == 0)
        total_terms = sum(keyword_counts.values()) or 1
        stability = 1 - novelty / len(keywords) if keywords else 1.0
        adaptation_trace.append(
            {
                "turn": idx,
                "keywords": keywords,
                "novelty_score": novelty,
                "stability": round(stability, 3),
                "compression_ratio": _compression_ratio(message),
            }
        )
        keyword_counts.update(keywords)

    return adaptation_trace


def _compression_ratio(message: str, max_length: int = 160) -> float:
    """Report how aggressive the compression would be for *message*."""

    raw_len = len(message.strip()) or 1
    compressed_len = len(compress_message(message, max_length)) or 1
    return round(compressed_len / raw_len, 3)

