# Hermes Tool Compressor

Deterministic, zero-LLM tool result compressor for [Hermes Agent](https://hermes-agent.nousresearch.com).  
Port of [Headroom](https://github.com/chopratejas/headroom)'s Rust compression algorithms to pure Python 3.10+.

Replaces LLM-based summarization with a **predictable, testable, sub-millisecond** pipeline that preserves semantic fidelity while drastically reducing token count.

## Features

- **Zero LLM calls** — pure `re` / `json` / `hashlib`. No APIs, no latency, no cost.
- **5 specialized compressors** — auto-detects content type and applies the best strategy.
- **KV-cache friendly** — `CacheAligner` prepends a prefix-stable header so provider caches hit.
- **Drop-in replacement** — same signature as the original `maybe_compress_tool_result`.
- **Resilient** — every compressor is wrapped in `try/except`; any failure returns the original text unchanged.

## Quick Start

```bash
cp tool_result_compressor.py /path/to/hermes-agent/agent/
```

That's it. Hermes's existing `tool_executor.py` calls `maybe_compress_tool_result()` — the new implementation automatically takes over.

## Architecture

```
                   ┌──────────────────────────────┐
                   │     ContentRouter.identify()   │
                   │  (regex-based, no ML, 100us)   │
                   └──────┬───────┬───────┬───────┘
                          │       │       │
            ┌─────────────┘   ┌───┘   └───┐
            ▼                  ▼           ▼
   ┌────────────────┐  ┌────────────┐  ┌──────────┐
   │ SmartCrusher   │  │ SearchComp │  │ LogComp  │
   │ (JSON arrays/  │  │ (grep/rg/  │  │ (pytest/  │
   │  objects)      │  │  ag output) │  │  cargo/   │
   │                │  │            │  │  npm/jest)│
   └────────────────┘  └────────────┘  └──────────┘
   ┌──────────┐       ┌──────────────┐
   │ DiffComp │       │ ProseComp    │
   │ (git     │       │ (plain text  │
   │  diffs)  │       │  fallback)   │
   └──────────┘       └──────────────┘
          │                  │
          └──────┬───────────┘
                 ▼
        ┌──────────────────┐
        │ CacheAligner     │
        │ (prefix-stable   │
        │  header)         │
        └──────────────────┘
                 ▼
        [compressed output]
```

## Compression Strategies

| Content Type | Detected By | Compressor | Typical Reduction |
|---|---|---|---|
| `json_array` / `json_object` | `json.loads()` parse test | SmartCrusher: preserve fields + high-value keys, truncate blobs, cap array length | 40-98% |
| `search` | `file:line:content` regex | SearchCompressor: group by file, keep top N, uniform middle sampling | 90-95% |
| `log` | log-level keywords + format markers | LogCompressor: severity scoring, keep errors/fails, deduplicate warnings, context window | 80-95% |
| `diff` | `diff --git` / `@@` headers | DiffCompressor: cap files + hunks, trim context around changes | 50-80% |
| `html` | doctype / structural tags | Strip tags to ProseCompressor | 60-90% |
| `prose` | fallback | ProseCompressor: head + key sentences + tail | 50-85% |

## API

```python
from agent.tool_result_compressor import maybe_compress_tool_result

# Hermes's existing hook - same signature
compressed = maybe_compress_tool_result(
    tool_name="search_files",
    result_content=huge_grep_output,
    agent=agent_instance,
    max_tokens=3000,      # compress only if over threshold
)
```

## Changelog

### 2026-06-07 - Audit fix round 1

- **ContentType enum collision**: `JSON_ARRAY == JSON_OBJECT` -> distinct values `"json_array"` / `"json_object"`
- **CacheAligner prefix**: SHA hash no longer embedded in prefix (self-defeating for KV cache); now uses stable tool-name-only prefix
- **ContentRouter priority**: JSON detection short-circuits to prevent keyword-rich fields from being misclassified as log
- **DiffCompressor hunk cap**: odd `max_hunks_per_file` values now handled correctly (no silent hunk loss)
- **LogCompressor trace blanks**: stack traces with blank lines between chained exceptions no longer terminate early
- **Array omission banner**: moved to end of sampled list, count reflects actual dropped items
- **SearchCompressor NameError**: `_HEADER_MARGIN` accessed as class attribute via `self.`
- 7 minor cleanups (dead code, redundant `.lower()`, magic numbers, regex optimization, docstring sync)

### 2026-06-07 - Initial release

Port of Headroom's Rust pipeline to Python. Replaces LLM-based `_summarize_with_llm()` with deterministic multi-strategy compression.

## License

MIT
