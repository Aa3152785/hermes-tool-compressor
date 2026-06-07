# Hermes Tool Compressor

将 Headroom (Rust) 的压缩算法翻译为纯 Python，替换 Hermes Agent 中原有的 LLM 摘要压缩方案。**零 LLM 调用**，纯确定性算法，保持语义保真度。

## 特性

- **零 LLM 调用** — 纯 `re` / `json` / `hashlib`，无网络、无延迟、无成本
- **5 种专用压缩器** — 自动检测内容类型并选择最优策略
- **KV 缓存友好** — `CacheAligner` 添加前缀稳定锚点，保证 Provider 缓存命中
- **即插即用** — 与原版 `maybe_compress_tool_result` 签名完全兼容
- **容错** — 每个压缩器都有 `try/except`，任何失败都返回原文

## 安装

```bash
cp tool_result_compressor.py <hermes-home>/hermes-agent/agent/tool_result_compressor.py
```

重启 Hermes。`tool_executor.py` 会自动调用 `maybe_compress_tool_result()` — 不需要改任何代码和配置。

## 工作流程

```
ContentRouter.identify(content) → 内容类型
    ├─ "json_array" / "json_object" → SmartCrusher (保留字段+关键值)
    ├─ "search"    → SearchCompressor (按文件分组，保留最佳匹配)
    ├─ "log"       → LogCompressor (按严重度评分，保留错误/栈追踪)
    ├─ "diff"      → DiffCompressor (裁剪文件和上下文行)
    ├─ "html"      → 去标签 → ProseCompressor
    └─ "prose"     → ProseCompressor (首尾保留+关键句提取)
CacheAligner.align(output) → 前缀稳定锚定，保证 KV 缓存命中
```

## 压缩效果

| 内容类型 | 检测方式 | 压缩器 | 典型压缩率 |
|---|---|---|---|
| JSON 数组/对象 | `json.loads()` 解析 | SmartCrusher：保留字段和高价值 key，截断大 blob，控制数组长度 | 40–98% |
| 搜索结果 | `file:行:内容` 正则匹配 | SearchCompressor：按文件分组、保留 Top N、均匀采样 | 90–95% |
| 构建/测试日志 | 日志级别关键词 + 格式标记 | LogCompressor：严重度评分，保留 error/fail，去重 warning | 80–95% |
| git diff | `diff --git` / `@@` 头 | DiffCompressor：控制文件数和 hunk 数，裁剪上下文 | 50–80% |
| HTML | doctype / 结构标签 | 去标签 → ProseCompressor | 60–90% |
| 纯文本 | 兜底 | ProseCompressor：首尾 + 关键句提取 | 50–85% |

## 集成方式

这是一个 **Hermes Agent 的补丁文件**。替换 `agent/tool_result_compressor.py` 后重启即可：

```bash
cp tool_result_compressor.py <hermes-home>/hermes-agent/agent/tool_result_compressor.py
```

Hermes 的 `tool_executor.py` 在每次工具返回超阈值的输出时都会调用 `maybe_compress_tool_result()` — 新压缩器无缝接管。

## 更新日志

### 2026-06-07 — 审计修复第一批

- **修复** `ContentType` 枚举值碰撞：`JSON_ARRAY == JSON_OBJECT` → 独立值 `"json_array"` / `"json_object"`
- **修复** `CacheAligner` 前缀：SHA hash 不再嵌入前缀（自毁 KV 缓存）
- **修复** `ContentRouter` 优先级：JSON 检测短路，防止 `"error"` 等字段被误判为日志
- **修复** `DiffCompressor` hunk 切片：奇数 `max_hunks_per_file` 不再丢失 hunk
- **修复** `LogCompressor` 栈追踪：异常链间空行不再提前终止
- **修复** 数组省略横幅位置和计数
- **修复** `SearchCompressor` NameError：`_HEADER_MARGIN` 通过 `self.` 访问
- 7 项小清理（死变量、冗余 `.lower()`、魔数 `+5`、正则优化、注释同步）

### 2026-06-07 — 初始发布

将 Headroom 的 Rust 流水线移植到 Python，替换基于 LLM 的 `_summarize_with_llm()` 为确定性多策略压缩。
