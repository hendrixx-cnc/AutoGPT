# Aura Compression Test Harness

The aura compression helpers in `autogpt_libs.aura` provide a lightweight way to
simulate AI-to-AI chatter that needs to be compressed before it is routed
between agents. The helpers do **not** rely on an external LLM which makes them
fast enough to add to unit tests or notebooks when you only need deterministic
signals.

## Key capabilities

| Helper | Purpose |
| --- | --- |
| `compress_message` | Deduplicate and truncate long technical status updates while keeping critical facts. |
| `compress_conversation` | Applies `compress_message` to an entire list of AI speakers. |
| `extract_keywords` | Pulls the most informative terms out of a chunk of text so you can visualize drift. |
| `simulate_inference_adaptation` | Generates a per-turn trace with novelty, stability, and compression ratio readings. |

## Example usage

```python
from autogpt_libs.aura import (
    AIMessage,
    compress_conversation,
    simulate_inference_adaptation,
)

messages = [
    AIMessage("planner", "Status update: embeddings trained successfully."),
    AIMessage("critic", "Requesting inference adaptation sweep with aura compression."),
]

compressed = compress_conversation(messages, max_length=80)
adaptation_trace = simulate_inference_adaptation(m.content for m in messages)
```

`compressed` now contains deduplicated content for each speaker, while
`adaptation_trace` shows how much new information was introduced at every turn.

## Tests

To keep the behavior stable we ship unit tests in
`autogpt_platform/autogpt_libs/tests/test_aura_compression.py`. The tests cover:

1. Deduplication and truncation behavior of `compress_message`.
2. Preservation of sender metadata in `compress_conversation`.
3. The structure and monotonicity of the inference adaptation trace.

Run the tests from the repository root:

```bash
poetry run pytest autogpt_platform/autogpt_libs/tests/test_aura_compression.py
```
