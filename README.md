# Hermes Tool Compressor

Deterministic, zero-LLM tool result compressor — port of Headroom's Rust algorithms to Python for Hermes Agent.

## Installation

Copy `tool_result_compressor.py` to your Hermes Agent's `agent/` directory:

```bash
cp tool_result_compressor.py /path/to/hermes-agent/agent/
```

## How it works

```
ContentRouter.identify(content) → content type
    ├─ "json"      → SmartCrusher.compress(content)
    ├─ "search"    → SearchCompressor.compress(content)
    ├─ "log"       → LogCompressor.compress(content)
    ├─ "diff"      → DiffCompressor.compress(content)
    └─ "prose"     → ProseCompressor.compress(content)
CacheAligner.align(output) → prefix-stable bytes
```

Zero LLM calls. Pure deterministic regex + JSON parsing + hashing.
