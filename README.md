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

Restart Hermes. That's it — `tool_executor.py` calls `maybe_compress_tool_result()` automatically. No import changes, no config changes.

## How It Works

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
| `json_array` / `json_object` | `json.loads()` parse test | SmartCrusher: preserve fields + high-value keys, truncate blobs, cap array length | 40–98% |
| `search` | `file:line:content` regex | SearchCompressor: group by file, keep top N, uniform middle sampling | 90–95% |
| `log` | log-level keywords + format markers | LogCompressor: severity scoring, keep errors/fails, deduplicate warnings, context window | 80–95% |
| `diff` | `diff --git` / `@@` headers | DiffCompressor: cap files + hunks, trim context around changes | 50–80% |
| `html` | doctype / structural tags | Strip tags → ProseCompressor | 60–90% |
| `prose` | fallback | ProseCompressor: head + key sentences + tail | 50–85% |

## Integration

This is a **drop-in patch** for Hermes Agent. Replace `agent/tool_result_compressor.py` and restart:

```bash
cp tool_result_compressor.py <hermes-home>/hermes-agent/agent/tool_result_compressor.py
```

Hermes's `tool_executor.py` calls `maybe_compress_tool_result()` on every tool result that exceeds the token threshold — the new compressor takes over transparently. No code changes needed elsewhere.

## Changelog

### 2026-06-07 — Audit fix round 1

- **ContentType enum collision**: `JSON_ARRAY == JSON_OBJECT` → distinct values `"json_array"` / `"json_object"`
- **CacheAligner prefix**: SHA hash no longer embedded in prefix (was self-defeating for KV cache); now uses stable tool-name-only prefix
- **ContentRouter priority**: JSON detection short-circuits to prevent keyword-rich fields from being misclassified as log
- **DiffCompressor hunk cap**: odd `max_hunks_per_file` values now handled correctly (no silent hunk loss)
- **LogCompressor trace blanks**: stack traces with blank lines between chained exceptions no longer terminate early
- **Array omission banner**: moved to end of sampled list, count reflects actual dropped items
- **SearchCompressor NameError**: `_HEADER_MARGIN` accessed as class attribute via `self.`
- 7 minor cleanups (dead code, redundant `.lower()`, magic numbers, regex optimization, docstring sync)

### 2026-06-07 — Initial release

Port of Headroom's Rust pipeline to Python. Replaces LLM-based `_summarize_with_llm()` with deterministic multi-strategy compression.

## License

MIT

---

## 中文说明

### 这是什么

将 Headroom (Rust) 的压缩算法翻译为纯 Python，替换 Hermes Agent 中原有的 LLM 摘要压缩方案。**零 LLM 调用**，纯确定性算法，保持语义保真度。

### 安装

```bash
cp tool_result_compressor.py <hermes-home>/hermes-agent/agent/tool_result_compressor.py
```

重启 Hermes 即可。`tool_executor.py` 会自动调用 `maybe_compress_tool_result()` — 不需要改任何代码。

### 工作流程

```
ContentRouter.identify(content) → 内容类型
    ├─ "json_array" / "json_object" → SmartCrusher (保留字段+关键值)
    ├─ "search"    → SearchCompressor (按文件分组，保留最佳匹配)
    ├─ "log"       → LogCompressor (按严重度评分，保留错误/栈追踪)
    ├─ "diff"      → DiffCompressor (裁剪文件和上下文行)
    └─ "prose"     → ProseCompressor (首尾保留+关键句提取)
CacheAligner.align(output) → 前缀稳定锚定，保证 KV 缓存命中
```

### 压缩效果

| 内容类型 | 典型压缩率 |
|---|---|
| JSON 数组/对象 | 40–98% |
| 搜索结果 (grep/rg) | 90–95% |
| 构建/测试日志 | 80–95% |
| git diff | 50–80% |
| HTML | 60–90% |
| 纯文本 | 50–85% |

### 更新日志

**2026-06-07 — 审计修复第一批**

- 修复 `ContentType` 枚举值碰撞 (`JSON_ARRAY == JSON_OBJECT`)
- 修复 `CacheAligner` 前缀：不再把 SHA hash 塞进前缀（自毁 KV 缓存）
- 修复 `ContentRouter` 优先级：JSON 检测短路，防止 `"error"` 等字段被误判为日志
- 修复 `DiffCompressor` hunk 切片：奇数 `max_hunks_per_file` 不再丢 hunk
- 修复 `LogCompressor` 栈追踪：容忍异常链中的空行
- 修复 `SearchCompressor` NameError：`_HEADER_MARGIN` 加 `self.`
- 修复数组省略横幅位置和计数
- 7 项小清理（死代码、冗余 `.lower()`、魔数、正则优化、注释同步）
