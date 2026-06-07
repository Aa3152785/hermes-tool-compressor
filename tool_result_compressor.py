"""Tool result auto-compression — deterministic, zero-LLM, prefix-cache-safe.

Translates Headroom's Rust compression pipeline (github.com/chopratejas/headroom)
into pure Python 3.10+. Replaces the previous LLM-based summarization with a
deterministic multi-strategy compressor that preserves semantic fidelity while
drastically reducing token count.

Pipeline::

    ContentRouter.identify(content) → content type
        ├─ "json"      → SmartCrusher.compress(content)
        ├─ "search"    → SearchCompressor.compress(content)
        ├─ "log"       → LogCompressor.compress(content)
        ├─ "diff"      → DiffCompressor.compress(content)
        └─ "prose"     → ProseCompressor.compress(content)
    CacheAligner.align(output) → prefix-stable bytes
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Token estimation (unchanged from original, tiktoken + fallback) ──────
_TOKENIZER: Any = None
_TOKENIZER_NAME = "cl100k_base"


def _get_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        import tiktoken  # type: ignore[import-not-found]
        _TOKENIZER = tiktoken.get_encoding(_TOKENIZER_NAME)
    except Exception:
        logger.debug("tiktoken not available; using char/4 fallback for token estimation")
        return None
    return _TOKENIZER


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text*.

    Uses tiktoken when available; falls back to ``len(text) // 4``
    (a conservative approximation for English text).
    """
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════════
# Content Detection
# ══════════════════════════════════════════════════════════════════════════


class ContentType(Enum):
    """Content types recognised by the detector.  Mirrors Headroom's enum."""

    JSON_ARRAY = "json"
    JSON_OBJECT = "json"
    SOURCE_CODE = "code"
    SEARCH_RESULTS = "search"
    BUILD_OUTPUT = "log"
    GIT_DIFF = "diff"
    HTML = "html"
    PROSE = "prose"


# ── Precompiled regex patterns (compiled once, shared) ───────────────────

_SEARCH_RESULT_RE = re.compile(r"^[^\s:]+:\d+:", re.MULTILINE)

_DIFF_HEADER_RE = re.compile(
    r"^(diff --git|diff --combined |diff --cc |--- a/|@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@|@@@+)",
    re.MULTILINE,
)

_LOG_LEVEL_RE = re.compile(
    r"(?i)\b(ERROR|FAIL|CRITICAL|FATAL|WARN(?:ING)?|INFO|DEBUG|TRACE)\b"
)

_PYTEST_MARKERS = [
    "=== FAILURES", "=== ERRORS", "=== test session",
    "=== short test summary", "PASSED [", "FAILED [", "ERROR [",
    "SKIPPED [", "collected ",
]
_CARGO_MARKERS = [
    "error[E", "error:", "warning:", "Compiling ", "Running ",
    "test result:", "Doc-tests ",
]
_NPM_MARKERS = [
    "npm ERR!", "npm WARN", "added ", "removed ", "audited ",
]
_JEST_MARKERS = [
    "PASS ", "FAIL ", "● ", "Tests: ", "Test Suites: ",
]
_MAKE_MARKERS = [
    "make[", "make:", "*** [", "Entering directory", "Leaving directory",
]

_HTML_DOCTYPE_RE = re.compile(r"(?i)<!DOCTYPE\s+html")
_HTML_TAG_RE = re.compile(r"(?i)</?html\b")
_HTML_STRUCTURAL = re.compile(r"(?i)</?(head|body|div|span|p|table|a|img|script)\b")

# Common source-code extension signals
_CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".java",
                    ".cpp", ".c", ".h", ".rb", ".php", ".swift",
                    ".kt", ".scala", ".cs", ".sh", ".bash", ".zsh"}

# Patterns that strongly indicate source code (not prose)
_CODE_SIGNALS = re.compile(
    r"^\s*(def |class |fn |func |function |import |from "
    r"|package |module |pub |let |const |var "
    r"|public |private |protected |static |void |int |string "
    r"|#include |using namespace |export )",
    re.MULTILINE,
)


def _content_type_json(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """Check if content is JSON or a JSON array/object."""
    stripped = content.strip()
    if not stripped:
        return ContentType.PROSE, 0.0, {}
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return ContentType.PROSE, 0.0, {}
    meta: dict[str, Any] = {}
    if isinstance(parsed, list):
        meta["item_count"] = len(parsed)
        return ContentType.JSON_ARRAY, 0.95, meta
    if isinstance(parsed, dict):
        meta["key_count"] = len(parsed)
        return ContentType.JSON_OBJECT, 0.90, meta
    return ContentType.PROSE, 0.0, {}


def _content_type_diff(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """Detect unified-diff / git diff patterns."""
    lines = content.split("\n")[:100]
    hits = sum(1 for line in lines if _DIFF_HEADER_RE.match(line))
    if hits >= 2:
        return ContentType.GIT_DIFF, min(1.0, 0.6 + hits * 0.05), {"diff_headers": hits}
    return ContentType.PROSE, 0.0, {}


def _content_type_html(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """HTML detection via doctype, tags, and structural elements."""
    lower = content.lower()
    has_doctype = bool(_HTML_DOCTYPE_RE.search(lower))
    has_html_tag = bool(_HTML_TAG_RE.search(lower))
    structural = len(_HTML_STRUCTURAL.findall(lower))
    confidence = 0.0
    if has_doctype:
        confidence += 0.5
    if has_html_tag:
        confidence += 0.3
    if structural > 0:
        confidence += min(structural * 0.03, 0.2)
    if confidence < 0.5:
        return ContentType.PROSE, 0.0, {}
    return ContentType.HTML, min(confidence, 1.0), {
        "has_doctype": has_doctype,
        "has_html_tag": has_html_tag,
        "structural_tags": structural,
    }


def _content_type_search(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """Detect search/grep/ripgrep output (``file:line:content`` format)."""
    lines = [l for l in content.split("\n")[:100] if l.strip()]
    if not lines:
        return ContentType.PROSE, 0.0, {}
    matching = sum(1 for l in lines if _SEARCH_RESULT_RE.match(l))
    if matching == 0:
        return ContentType.PROSE, 0.0, {}
    ratio = matching / len(lines)
    if ratio < 0.3:
        return ContentType.PROSE, 0.0, {}
    confidence = min(1.0, 0.4 + ratio * 0.6)
    return ContentType.SEARCH_RESULTS, confidence, {
        "matching_lines": matching,
        "total_lines": len(lines),
    }


def _content_type_log(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """Detect build/log output via log-level keywords and format markers."""
    lines = content.split("\n")[:100]
    if not lines:
        return ContentType.PROSE, 0.0, {}
    log_lines = sum(1 for l in lines if _LOG_LEVEL_RE.search(l))
    ratio = log_lines / max(len(lines), 1)
    if ratio > 0.3:
        return ContentType.BUILD_OUTPUT, min(1.0, 0.7 + ratio), {
            "log_lines": log_lines,
        }

    # Check format-specific markers on concatenated first 4096 chars
    head = content[:4096]
    for markers, fmt_name in [
        (_PYTEST_MARKERS, "pytest"), (_CARGO_MARKERS, "cargo"),
        (_NPM_MARKERS, "npm"), (_JEST_MARKERS, "jest"),
        (_MAKE_MARKERS, "make"),
    ]:
        hits = sum(1 for m in markers if m in head)
        if hits >= 2:
            return ContentType.BUILD_OUTPUT, 0.75, {"format": fmt_name, "markers": hits}
    return ContentType.PROSE, 0.0, {}


def _content_type_code(content: str) -> tuple[ContentType, float, dict[str, Any]]:
    """Detect source code via extension-like paths and code signal lines."""
    lines = content.split("\n")[:50]
    code_signals = len(_CODE_SIGNALS.findall("\n".join(lines)))
    if code_signals >= 3:
        return ContentType.SOURCE_CODE, min(1.0, 0.6 + code_signals * 0.05), {
            "code_signals": code_signals,
        }
    return ContentType.PROSE, 0.0, {}


class ContentRouter:
    """Detect content type and route to the appropriate compressor.

    Detection is regex-based—no ML, no model loading, no I/O.
    Checks are ordered for precision: JSON first (parse test),
    then diff, HTML, search, log, code, prose fallback.
    """

    CHECKERS = [
        _content_type_json,
        _content_type_diff,
        _content_type_html,
        _content_type_search,
        _content_type_log,
        _content_type_code,
    ]

    @staticmethod
    def identify(content: str) -> tuple[ContentType, float, dict[str, Any]]:
        """Return ``(content_type, confidence, metadata)``."""
        if not content or not content.strip():
            return ContentType.PROSE, 1.0, {}
        best_type, best_conf, best_meta = ContentType.PROSE, 0.0, {}
        for checker in ContentRouter.CHECKERS:
            ct, conf, meta = checker(content)
            if ct != ContentType.PROSE and conf > best_conf:
                best_type, best_conf, best_meta = ct, conf, meta
        return best_type, best_conf, best_meta


# ══════════════════════════════════════════════════════════════════════════
# SmartCrusher — JSON / structured-data compression
# ══════════════════════════════════════════════════════════════════════════


class SmartCrusher:
    """Compress JSON arrays/objects while preserving structure and key values.

    Strategy (deterministic, zero-LLM):
    1. If object: keep all keys, truncate very-long string values.
    2. If array: cap to ``max_items``; per-item, keep structurally important
       fields, truncate opaque blobs, and preserve sort order.
    3. Always produce valid JSON on output (never a parse error for downstream).
    """

    # Fields that are almost always worth keeping (high semantic density)
    _HIGH_VALUE_FIELDS: set[str] = {
        "name", "title", "id", "key", "type", "status", "path",
        "file", "url", "message", "error", "summary", "result",
        "score", "count", "total", "version", "description",
        "content", "text", "value", "label", "category", "role",
    }

    # Fields to always drop (ephemeral / useless for context)
    _DROP_FIELDS: set[str] = {
        "timestamp", "created_at", "updated_at", "etag", "cache_key",
        "request_id", "trace_id", "span_id",
    }

    def __init__(
        self,
        max_items: int = 50,
        max_string_len: int = 512,
        max_object_depth: int = 4,
    ) -> None:
        self.max_items = max_items
        self.max_string_len = max_string_len
        self.max_object_depth = max_object_depth

    def compress(self, content: str) -> str:
        """Compress JSON content.  Returns original if not parseable."""
        try:
            parsed = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            return content
        try:
            if isinstance(parsed, list):
                compressed = self._compress_array(parsed)
            elif isinstance(parsed, dict):
                compressed = self._compress_object(parsed)
            else:
                return content
            return json.dumps(compressed, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            logger.debug("SmartCrusher.compress failed; returning original", exc_info=True)
            return content

    # ── Array compression ──────────────────────────────────────────────

    def _compress_array(self, arr: list[Any]) -> list[Any]:
        if len(arr) <= self.max_items:
            return [self._compress_value(v, 0) for v in arr]
        # Keep first 5 + last 5 + uniformly sampled middle
        keep_first = 5
        keep_last = 5
        if len(arr) <= keep_first + keep_last:
            return [self._compress_value(v, 0) for v in arr]
        mid_count = self.max_items - keep_first - keep_last
        if mid_count <= 0:
            mid_count = max(1, self.max_items - keep_first)
        middle = arr[keep_first:-keep_last]
        if len(middle) > mid_count:
            step = max(1, len(middle) // mid_count)
            middle = middle[::step][:mid_count]
        result: list[Any] = []
        result.extend(self._compress_value(v, 0) for v in arr[:keep_first])
        if len(middle) > 0:
            result.append(f"[... {len(arr) - keep_first - keep_last - len(middle)} items omitted ...]")
        result.extend(self._compress_value(v, 0) for v in middle)
        result.extend(self._compress_value(v, 0) for v in arr[-keep_last:])
        return result

    # ── Object compression ─────────────────────────────────────────────

    # When a key is in _HIGH_VALUE_FIELDS, string values get 4× the normal
    # budget before truncation.  This preserves semantically dense fields
    # (name, error, path, summary) while aggressively crushing low-value
    # fields (long context blobs, raw data payloads).
    _HIGH_VALUE_STRING_MULTIPLIER: int = 4

    def _compress_object(self, obj: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        if depth >= self.max_object_depth:
            return {"_truncated": True, "_keys": list(obj.keys())[:20]}
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if key in self._DROP_FIELDS:
                continue
            budget = (
                self.max_string_len * self._HIGH_VALUE_STRING_MULTIPLIER
                if key in self._HIGH_VALUE_FIELDS
                else self.max_string_len
            )
            result[key] = self._compress_value(value, depth, str_budget=budget)
        return result

    # ── Value-level compression ────────────────────────────────────────

    def _compress_value(self, value: Any, depth: int, str_budget: int | None = None) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._compress_string(value, max_len=str_budget or self.max_string_len)
        if isinstance(value, dict):
            return self._compress_object(value, depth + 1)
        if isinstance(value, list):
            if len(value) <= 20:
                return [self._compress_value(v, depth) for v in value]
            return (
                [self._compress_value(v, depth) for v in value[:10]]
                + [f"[... {len(value) - 20} more items ...]"]
                + [self._compress_value(v, depth) for v in value[-10:]]
            )
        return str(value)[:str_budget or self.max_string_len]

    def _compress_string(self, s: str, max_len: int | None = None) -> str:
        """Truncate long strings; classify opaque blobs (base64, HTML)."""
        limit = max_len if max_len is not None else self.max_string_len
        if len(s) <= limit:
            return s
        # Detect base64 blobs (check against full string before truncation)
        if self._is_base64_blob(s):
            return f"[base64 blob: {len(s)} chars, hash={_hash_prefix(s)}]"
        # Detect HTML
        if s.count("<") >= 3 and any(
            s[i:i+2] in ("</", "<!", "<a", "<d", "<s", "<p", "<t")
            for i in range(min(len(s) - 1, 2000))
        ):
            return f"[HTML: {len(s)} chars]"
        # Keep prefix + suffix for long text
        half = limit // 2
        return s[:half] + f"\n[... {len(s) - limit} chars truncated ...]\n" + s[-half:]

    @staticmethod
    def _is_base64_blob(s: str) -> bool:
        """Detect base64-encoded data."""
        if len(s) < 64:
            return False
        if "<" in s or ">" in s:
            return False
        if any(c.isspace() for c in s):
            return False
        alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")
        match = sum(1 for c in s if c in alphabet)
        if match / len(s) < 0.95:
            return False
        # Diversity filter: real base64 has ≥16 unique chars
        return len(set(s)) >= 16


# ══════════════════════════════════════════════════════════════════════════
# SearchCompressor — grep / ripgrep output
# ══════════════════════════════════════════════════════════════════════════

_SEARCH_LINE_RE = re.compile(r"^([^:]+):(\d+):(.*)$")
_SEARCH_CONTEXT_RE = re.compile(r"^([^:]+)-(\d+)-(.*)$")


class SearchCompressor:
    """Compress search/grep/ripgrep output.

    Handles both standard ``file:line:content`` and ripgrep context
    lines (``file-line-content``).  Compression: keep most relevant
    matches, deduplicate, summarise.
    """

    def __init__(
        self,
        max_files: int = 20,
        max_matches: int = 80,
    ) -> None:
        self.max_files = max_files
        self.max_matches = max_matches

    def compress(self, content: str, _context: str = "") -> str:
        """Compress search results."""
        lines = content.split("\n")
        if len(lines) <= self.max_matches:
            return content

        # Parse into {file: [(line_num, content), ...]}
        file_matches: dict[str, list[tuple[int, str]]] = {}
        for line in lines:
            m = _SEARCH_LINE_RE.match(line)
            if m:
                fname, lnum, text = m.group(1), m.group(2), m.group(3)
                file_matches.setdefault(fname, []).append((int(lnum), text))
                continue
            m2 = _SEARCH_CONTEXT_RE.match(line)
            if m2:
                fname, lnum, text = m2.group(1), m2.group(2), m2.group(3)
                file_matches.setdefault(fname, []).append((int(lnum), text))

        if not file_matches:
            # Fallback: try simple grep pattern on raw lines
            return self._fallback_compress(lines)

        # Score files by match count; keep top N
        scored_files = sorted(file_matches.items(),
                             key=lambda x: len(x[1]), reverse=True)
        kept_files = scored_files[:self.max_files]
        dropped = len(scored_files) - len(kept_files)

        output: list[str] = []
        total_kept = 0
        for fname, matches in kept_files:
            # Sort matches by line number
            matches.sort(key=lambda x: x[0])
            if len(matches) <= 10:
                for lnum, text in matches:
                    output.append(f"{fname}:{lnum}:{text}")
                    total_kept += 1
            else:
                # Keep first 3 + last 3 + sample middle
                for lnum, text in matches[:3]:
                    output.append(f"{fname}:{lnum}:{text}")
                    total_kept += 1
                mid = matches[3:-3]
                step = max(1, len(mid) // 4)
                for lnum, text in mid[::step]:
                    output.append(f"{fname}:{lnum}:{text}")
                    total_kept += 1
                for lnum, text in matches[-3:]:
                    output.append(f"{fname}:{lnum}:{text}")
                    total_kept += 1
                output.append(f"[... and {len(matches) - 10} more matches in {fname}]")

        if dropped > 0:
            output.insert(0, f"[{dropped} files with fewer matches omitted]")

        return "\n".join(output[:self.max_matches + 5])

    def _fallback_compress(self, lines: list[str]) -> str:
        """Simple truncation for unrecognised search format."""
        if len(lines) <= self.max_matches:
            return "\n".join(lines)
        keep = self.max_matches
        head = lines[:keep // 2]
        tail = lines[-(keep // 2):]
        return "\n".join(head) + f"\n[... {len(lines) - keep} lines omitted ...]\n" + "\n".join(tail)


# ══════════════════════════════════════════════════════════════════════════
# LogCompressor — build / test / system log output
# ══════════════════════════════════════════════════════════════════════════

_LOG_CLASSIFY_RE = re.compile(
    r"(?i)^(?P<ts>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s+)?"
    r"(?P<level>(?:ERROR|FAIL|CRITICAL|FATAL|WARN(?:ING)?|INFO|DEBUG|TRACE))\b"
    r"(?:[:\s\]])*(?P<message>.*)"
)

_LOG_SEVERITY_ORDER = {
    "FATAL": 5, "CRITICAL": 5, "ERROR": 4, "FAIL": 4,
    "WARN": 2, "WARNING": 2, "INFO": 1, "DEBUG": 0, "TRACE": 0,
}

_STACKTRACE_START_RE = re.compile(
    r"(?i)^(Traceback \(most recent call last\)|Stack trace:|"
    r"^\s+at |^\s+File \")"
)

_SUMMARY_MARKERS = [
    "=== short test summary",
    "test result:",
    "Tests:",
    "Test Suites:",
    "Snapshots:",
    "Ran ",
    "OK (",
    "FAILED (",
    "Build ",
    "real\t",
    "user\t",
]


class LogCompressor:
    """Compress build/test/system logs.

    Pipeline:
    1. Detect log format (pytest, cargo, npm, jest, make, generic).
    2. Per-line classification: severity level, stack-trace, summary.
    3. Score each line; keep errors/fails, deduplicate warnings, keep summaries.
    4. Context window around each kept line.
    """

    def __init__(
        self,
        max_lines: int = 200,
        context_window: int = 2,
    ) -> None:
        self.max_lines = max_lines
        self.context_window = context_window

    def compress(self, content: str, _context: str = "") -> str:
        """Compress log output."""
        lines = content.split("\n")
        if len(lines) <= self.max_lines:
            return content

        # Score every line
        scored: list[tuple[int, int, str]] = []  # (index, score, line)
        in_stacktrace = False
        for i, line in enumerate(lines):
            score = 0

            # Check if entering/continuing stack trace
            if _STACKTRACE_START_RE.match(line):
                in_stacktrace = True
                score = 8
            elif in_stacktrace:
                if line and line.strip() and (line[0] == " " or line[0] == "\t"):
                    score = 6
                else:
                    in_stacktrace = False

            # Classify severity
            m = _LOG_CLASSIFY_RE.match(line)
            if m:
                level = m.group("level").upper()
                score = max(score, _LOG_SEVERITY_ORDER.get(level, 0) * 2)

            # Summary lines
            if any(marker in line for marker in _SUMMARY_MARKERS):
                score = max(score, 7)

            # Boost lines with file paths or error codes
            if re.search(r"(?:error[\[:])|(?:\.(?:py|rs|js|go|java|ts):\d+)", line, re.IGNORECASE):
                score = max(score, 5)

            scored.append((i, score, line))

        # Select lines to keep
        kept_indices: set[int] = set()

        # Always keep first 5 and last 10 lines
        for idx in range(min(5, len(lines))):
            kept_indices.add(idx)
        for idx in range(max(0, len(lines) - 10), len(lines)):
            kept_indices.add(idx)

        # Sort by score, keep top lines
        scored.sort(key=lambda x: x[1], reverse=True)
        for i, score, line in scored:
            if len(kept_indices) >= self.max_lines:
                break
            if score >= 3:
                kept_indices.add(i)
                # Add context window
                for offset in range(1, self.context_window + 1):
                    if i + offset < len(lines):
                        kept_indices.add(i + offset)
                    if i - offset >= 0:
                        kept_indices.add(i - offset)

        # Sort kept indices and emit
        kept_sorted = sorted(kept_indices)
        output: list[str] = []
        prev = -2
        for idx in kept_sorted:
            if idx > prev + 1:
                gap = idx - prev - 1
                if gap > 0 and prev >= 0:
                    output.append(f"[... {gap} lines omitted ...]")
            output.append(lines[idx])
            prev = idx

        return "\n".join(output)


# ══════════════════════════════════════════════════════════════════════════
# DiffCompressor — unified diff / git diff output
# ══════════════════════════════════════════════════════════════════════════

_DIFF_FILE_HEADER_RE = re.compile(r"^diff --git ")
_DIFF_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


class DiffCompressor:
    """Compress unified/git diff output.

    Caps file count, hunks per file, and context lines around changes.
    """

    def __init__(
        self,
        max_files: int = 10,
        max_hunks_per_file: int = 8,
        max_context_lines: int = 3,
    ) -> None:
        self.max_files = max_files
        self.max_hunks_per_file = max_hunks_per_file
        self.max_context_lines = max_context_lines

    def compress(self, content: str, _context: str = "") -> str:
        """Compress diff output."""
        lines = content.split("\n")
        if len(lines) <= 200:
            return content

        # Split into per-file sections
        files: list[list[str]] = []
        current_file: list[str] = []
        for line in lines:
            if _DIFF_FILE_HEADER_RE.match(line) and current_file:
                files.append(current_file)
                current_file = [line]
            else:
                current_file.append(line)
        if current_file:
            files.append(current_file)

        if len(files) <= self.max_files:
            return content

        # Score files by change volume (+ and - lines)
        def _count_changes(file_lines: list[str]) -> int:
            return sum(1 for l in file_lines if l.startswith("+") or l.startswith("-"))

        scored = [(sum(1 for l in f if _DIFF_FILE_HEADER_RE.match(l)) + _count_changes(f), f)
                  for f in files]
        scored.sort(key=lambda x: x[0], reverse=True)

        output: list[str] = []
        for _, f_lines in scored[:self.max_files]:
            compressed = self._compress_single_file(f_lines)
            output.extend(compressed)

        skipped = len(scored) - self.max_files
        if skipped > 0:
            output.insert(0, f"[{skipped} files with fewer changes omitted]")

        return "\n".join(output)

    def _compress_single_file(self, lines: list[str]) -> list[str]:
        """Compress a single file's diff."""
        # Split into header + hunks
        hunks: list[list[str]] = []
        current_hunk: list[str] = []
        header_lines: list[str] = []

        for line in lines:
            if _DIFF_HUNK_RE.match(line):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = [line]
            elif not hunks and not current_hunk:
                header_lines.append(line)
            else:
                current_hunk.append(line)
        if current_hunk:
            hunks.append(current_hunk)

        # Cap hunks
        if len(hunks) > self.max_hunks_per_file:
            kept = hunks[:3] + hunks[-3:]
            result = list(header_lines)
            result.append(f"[... {len(hunks) - 6} hunks omitted ...]")
            for hunk in kept:
                result.extend(self._trim_context(hunk))
            return result

        result = list(header_lines)
        for hunk in hunks:
            result.extend(self._trim_context(hunk))
        return result

    def _trim_context(self, hunk_lines: list[str]) -> list[str]:
        """Trim context lines around + and - lines."""
        result = list(hunk_lines[:1])  # Keep hunk header
        body = hunk_lines[1:]
        # Mark which lines to keep
        change_indices: set[int] = set()
        for i, line in enumerate(body):
            if line.startswith("+") or line.startswith("-"):
                change_indices.add(i)
        if not change_indices:
            return hunk_lines

        kept: set[int] = set()
        for ci in change_indices:
            kept.add(ci)
            for offset in range(1, self.max_context_lines + 1):
                if ci + offset < len(body):
                    kept.add(ci + offset)
                if ci - offset >= 0:
                    kept.add(ci - offset)

        prev = -2
        for i in sorted(kept):
            if i > prev + 1:
                result.append(f"[... {i - prev - 1} context lines omitted ...]")
            result.append(body[i])
            prev = i
        return result


# ══════════════════════════════════════════════════════════════════════════
# ProseCompressor — generic plain-text compression
# ══════════════════════════════════════════════════════════════════════════


class ProseCompressor:
    """Compress generic prose / plain-text.

    Strategy: keep first N chars, extract key sentences (those with
    high information density — named entities, numbers, quotes), keep
    last N chars.  Simple, deterministic, zero-LLM.
    """

    _KEY_SENTENCE_SIGNALS = re.compile(
        r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)|"  # Named entities
        r"(\b\d+(?:\.\d+)?%?\b)|"                  # Numbers / percentages
        r"(\"|'|`)"                                 # Quoted text
    )

    def __init__(self, max_chars: int = 3000, head_chars: int = 600, tail_chars: int = 400) -> None:
        self.max_chars = max_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars

    def compress(self, content: str, _context: str = "") -> str:
        """Compress prose text."""
        if len(content) <= self.max_chars:
            return content

        head = content[:self.head_chars]
        tail = content[-self.tail_chars:]

        # Middle: extract key sentences
        mid = content[self.head_chars:-self.tail_chars]
        mid_budget = self.max_chars - self.head_chars - self.tail_chars
        if mid_budget <= 0:
            return head + f"\n[... {len(mid)} chars omitted ...]\n" + tail

        sentences = re.split(r"(?<=[.!?])\s+", mid)
        if len(sentences) <= 10:
            return content[:self.max_chars] + f"\n[... {len(content) - self.max_chars} chars truncated ...]"

        # Score sentences
        scored: list[tuple[int, str]] = []
        for sent in sentences:
            sig = len(self._KEY_SENTENCE_SIGNALS.findall(sent))
            scored.append((sig, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[str] = []
        chars_used = 0
        for _, sent in scored:
            if chars_used + len(sent) + 2 > mid_budget:
                break
            if sent.strip():
                selected.append(sent)
                chars_used += len(sent) + 2

        middle_text = " [...] ".join(selected)
        return head + "\n[...]\n" + middle_text + "\n[...]\n" + tail


# ══════════════════════════════════════════════════════════════════════════
# CacheAligner — byte-prefix stability
# ══════════════════════════════════════════════════════════════════════════


class CacheAligner:
    """Ensure compressed output has a stable byte prefix.

    When the same tool is called multiple times with similar output,
    the compressed result should share a common prefix.  This lets
    provider KV caches (DeepSeek, Anthropic) hit on the cached prefix.

    Strategy: prepend a stable header that varies only when the
    content changes semantically, not when minor details shift.
    """

    _HEADER_SEPARATOR = "\n--- compressed ---\n"

    @staticmethod
    def align(compressed: str, tool_name: str = "") -> str:
        """Return aligned output with a stable prefix."""
        # Hash the compressed content for a cache-aware fingerprint
        fp = _hash_prefix(compressed)
        header = f"[compressed {tool_name} | sha:{fp}]\n" if tool_name else f"[compressed | sha:{fp}]\n"
        return header + compressed

    @staticmethod
    def compute_stable_hash(content: str, chunk_size: int = 4096) -> str:
        """Compute a stable hash over content chunks for cache keys."""
        if len(content) <= chunk_size:
            return _hash_prefix(content)
        # Hash first chunk + last chunk for stability
        head = content[:chunk_size]
        tail = content[-chunk_size:] if len(content) > chunk_size else ""
        return _hash_prefix(head + "||" + tail)


# ══════════════════════════════════════════════════════════════════════════
# CompressionPipeline — orchestration
# ══════════════════════════════════════════════════════════════════════════


class CompressionPipeline:
    """Orchestrate: detect content type → dispatch to compressor → align."""

    def __init__(self) -> None:
        self.smart_crusher = SmartCrusher()
        self.search_compressor = SearchCompressor()
        self.log_compressor = LogCompressor()
        self.diff_compressor = DiffCompressor()
        self.prose_compressor = ProseCompressor()

    def compress(self, content: str, tool_name: str = "") -> str:
        """Run the full pipeline: identify → compress → align.

        Args:
            content: The tool result string to compress.
            tool_name: The tool that produced this output (for header context).

        Returns:
            Compressed string, or the original if compression fails.
        """
        if not content or not content.strip():
            return content

        try:
            ct, confidence, meta = ContentRouter.identify(content)
            logger.debug(
                "ContentRouter: type=%s confidence=%.2f meta=%s",
                ct.value, confidence, meta,
            )
        except Exception:
            logger.debug("ContentRouter failed; treating as prose", exc_info=True)
            ct = ContentType.PROSE

        try:
            if ct in (ContentType.JSON_ARRAY, ContentType.JSON_OBJECT):
                compressed = self.smart_crusher.compress(content)
            elif ct == ContentType.SEARCH_RESULTS:
                compressed = self.search_compressor.compress(content)
            elif ct in (ContentType.BUILD_OUTPUT,):
                compressed = self.log_compressor.compress(content)
            elif ct == ContentType.GIT_DIFF:
                compressed = self.diff_compressor.compress(content)
            elif ct == ContentType.HTML:
                # HTML: strip tags attempt, then compress as prose
                stripped = re.sub(r"<[^>]+>", " ", content)
                stripped = re.sub(r"\s+", " ", stripped).strip()
                compressed = self.prose_compressor.compress(stripped)
            else:
                compressed = self.prose_compressor.compress(content)
        except Exception:
            logger.debug("Compression failed; returning original", exc_info=True)
            return content

        try:
            aligned = CacheAligner.align(compressed, tool_name)
        except Exception:
            aligned = compressed

        return aligned


# ══════════════════════════════════════════════════════════════════════════
# Public API — compatible with existing Hermes hooks
# ══════════════════════════════════════════════════════════════════════════

# Module-level singleton for the compression pipeline
_PIPELINE = CompressionPipeline()


def _hash_prefix(s: str, length: int = 16) -> str:
    """Stable hex hash prefix."""
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:length]


def maybe_compress_tool_result(
    tool_name: str,
    result_content: Any,
    agent: Any,          # Kept for signature compatibility; unused (zero-LLM)
    *,
    max_tokens: int = 3000,
    compress_model: str = "",  # Kept for signature compatibility; unused
) -> str:
    """Conditionally compress a tool result using deterministic algorithms.

    Replaces the previous LLM-based summarization.  When the result
    exceeds ``max_tokens``, the content is compressed via the
    Headroom-derived pipeline: content-type detection → type-specific
    compression → cache-prefix alignment.  Zero LLM calls.

    Args:
        tool_name: The name of the tool that produced the result.
        result_content: The raw tool result (string or multimodal dict).
        agent: The AIAgent instance (unused; kept for compatibility).
        max_tokens: Token threshold above which compression triggers.
        compress_model: Unused (kept for compatibility).

    Returns:
        The original content (if under threshold) or a compressed string.
    """
    # Handle multimodal results (same logic as original)
    if isinstance(result_content, dict):
        if result_content.get("_multimodal"):
            text = result_content.get("text_summary", "")
            if not text:
                try:
                    from agent.tool_dispatch_helpers import _multimodal_text_summary  # type: ignore[import-not-found]
                    text = _multimodal_text_summary(result_content)
                except Exception:
                    try:
                        text = json.dumps(result_content, ensure_ascii=False)
                    except Exception:
                        text = str(result_content)
            result_content = text
        else:
            try:
                result_content = json.dumps(result_content, ensure_ascii=False)
            except Exception:
                result_content = str(result_content)

    if not isinstance(result_content, str):
        result_content = str(result_content)

    if not result_content:
        return result_content

    tokens = estimate_tokens(result_content)
    if tokens <= max_tokens:
        return result_content

    logger.info(
        "Tool '%s' result is %d tokens (threshold: %d) — compressing (deterministic, zero-LLM)",
        tool_name, tokens, max_tokens,
    )

    try:
        compressed = _PIPELINE.compress(result_content, tool_name)
        compressed_tokens = estimate_tokens(compressed)
        logger.info(
            "Tool '%s': %d → %d tokens (%.1f%% reduction)",
            tool_name, tokens, compressed_tokens,
            (1 - compressed_tokens / tokens) * 100 if tokens > 0 else 0,
        )
        return compressed
    except Exception:
        logger.warning(
            "Compression failed for tool '%s'; returning original %d-token result",
            tool_name, tokens, exc_info=True,
        )
        return result_content
