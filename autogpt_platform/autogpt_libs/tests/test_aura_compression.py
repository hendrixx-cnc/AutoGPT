"""Tests for the aura compression helpers."""
from autogpt_libs.aura import (
    AIMessage,
    compress_conversation,
    compress_message,
    extract_keywords,
    simulate_inference_adaptation,
)


def test_compress_message_removes_duplicates_and_truncates():
    content = (
        "Status update: embeddings trained successfully. "
        "Status update: embeddings trained successfully. "
        "Proceeding to quantization step once more metrics land."
    )

    compressed = compress_message(content, max_length=120)

    assert "once more metrics land" in compressed
    assert compressed.count("Status update") == 1
    assert len(compressed) <= 120


def test_compress_conversation_returns_new_objects():
    messages = [
        AIMessage(sender="planner", content="Planning iteration complete."),
        AIMessage(sender="critic", content="Planning iteration complete."),
    ]

    compressed = compress_conversation(messages, max_length=40)

    assert compressed[0].sender == "planner"
    assert compressed[1].sender == "critic"
    assert compressed[1].content == "Planning iteration complete."
    assert compressed[0] is not messages[0]


def test_extract_keywords_and_adaptation_trace():
    conversation = [
        "Summarizing latest aura compression sweep with 42% reduction on routing logs.",
        "Running inference adaptation using cached embeddings for routing logs.",
        "Inference adaptation converged; aura compression now tuned for message brokers.",
    ]

    keywords = extract_keywords(conversation[0], top_k=3)
    assert keywords[0] == "summarizing"

    trace = simulate_inference_adaptation(conversation)
    assert len(trace) == 3
    assert trace[0]["novelty_score"] >= trace[2]["novelty_score"]
    assert 0 < trace[1]["compression_ratio"] <= 1
