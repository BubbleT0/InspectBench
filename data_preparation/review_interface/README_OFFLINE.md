# Critic Benchmark 离线复核包

本文件夹是自包含离线副本，不依赖 `/path/to/storage` 原路径或网络。

## 打开方式

可以直接双击 `index.html`。若浏览器限制 `file://` 的本地存储，使用本机 Python 启动静态服务：

```bash
python serve_offline.py
```

然后打开 `http://127.0.0.1:8765`。

## 包含内容

- `images/`：5,007 张复制后的原数据图片，按数据集分目录。
- `records/`：5,007 个逐样本完整原始审计 JSON，包含实际分类 Prompt、所有 VLM attempts 和原始响应。
- `data/`：离线网页加载的配置和分页数据 chunk。
- `export_manifest.jsonl`：逐样本来源清单，记录原绝对图片路径、原 run/log/行号、本地图片路径、SHA-256 和原始记录路径。
- `summary.md`、`taxonomy.md`、`plan.json`：交接、taxonomy 和配额配置。
- `review_annotations_template.json`：人工标注导出格式示例。

## 人工复核

每个样本可标记：

- `approved`：人工确认 VLM 主标签。
- `corrected`：人工修正大类和叶子。
- `rejected`：样本不应纳入 benchmark。
- `followup`：需要第二位审核者进一步判断。

还可填写复核人和自由文本备注。标注会自动写入浏览器 localStorage，但 localStorage 不是可靠的长期存储。请定期点击页面顶部的“导出人工标注 JSON”。导出的 JSON 可以在同一页面重新导入并合并。

## 来源追踪

网页中每条样本均显示：

- 本地复制图片路径和原始绝对路径。
- 图片 SHA-256。
- 数据集、item ID、source run、source log 和行号。
- 原 prompt/说明和 metadata。
- 完整 VLM 分类决策和实际分类 Prompt。
- 完整原始审计 JSON 的相对链接。

TextPecker decoded_test 没有物化逐样本 target text/conversation，因此相关页面会明确显示原 prompt 不可用；不能据此评 D2-B。
