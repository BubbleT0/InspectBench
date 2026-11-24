# U1 Image Critic V6.3

V6.3 是已验证的 V6.3-I 管线独立交付包。目标是定位图中肉眼明显的 Artifact 与明显文字错误，并输出适合后续 edit 模型一次局部重绘的区域；不检查通用美观度、世界知识或肉眼几乎不可见的小瑕疵。

## 安装

```bash
python3 -m pip install dist/u1_critic_v63-6.3.0-py3-none-any.whl
```

## 快速运行

```bash
export CRITIC_BASE_URL=http://your-model-host:8000
export CRITIC_MODEL=your-model-name
export CRITIC_API_KEY=your-key
bash examples/run_v63.sh /path/to/images /path/to/output /path/to/input.jsonl
```

JSONL 默认图片字段为 `image_absolute_path`。全量运行要求 `--all --confirm-run-all YES`。CLI 自动启用正式 V6.3-I 配置；默认图片并发为 4，可通过 `CRITIC_MAX_IMAGES_IN_FLIGHT` 调整。

输出包括 `manifest.json`、`run_config.json`、`results.jsonl`、`errors.jsonl` 和 `traces.jsonl`。只有在线整编完成且校验通过的结果才写入 `results.jsonl`，正式结论位于 `consolidated_result`。所有 bbox 均为整数 `[0,1000]` 坐标。

生成与查看报告：

```bash
u1-critic-v63-report --output-root /path/to/output --omit-model-traces
u1-critic-v63-serve --output-root /path/to/output --port 11223
```

详细接入规范见 `HANDOFF_CN.md`。
