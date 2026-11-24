# U1 Image Critic V4.1

这是当前 V4.1 图像 Critic 管线的独立交付包。它面向局部重绘：发现图中明显的非文字 artifact 和明显的文字错误，并输出适合后续编辑模型重绘的局部区域。

正式基线为 `adversarial_consensus + harness_v41 + 在线整编`。V4.1 将文字与非文字检查分开，通过语义内容盘点、多视图复核、覆盖完整性校验、恢复调用和最终整编来降低漏判与不稳定性。

## 核心约束

- 只关注明显、可见且影响较大的错误；不检查年份等世界知识、美观偏好和轻微瑕疵。
- 文字区域必须完整覆盖；非文字错误按渲染、结构、拓扑、关系与物理分类检查。
- bbox 是供编辑模型局部重绘的区域，可以有少量冗余，但应保持实体或语义区块独立。
- 同一重绘区域只保留一个 bbox，并在描述中合并该区域的全部问题。
- 禁止全图框、无必要的大框、重复框和大框套住多个已准确定位的小框。
- 全链路 bbox 坐标均为整数 `[0,1000]`，顺序是 `[x1, y1, x2, y2]`。
- 正式成功必须在管线内完成在线整编；不得用离线脚本补写成功结果。
- 不把输入目录名或目录标签发送给模型，避免答案泄漏。

完整设计、输入输出协议、验收方法和已知限制见 [HANDOFF_CN.md](HANDOFF_CN.md)。

## 安装

建议使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .
```

也可以直接安装 `dist/` 下的 wheel。

## 最小冒烟测试

先用一张图验证模型接口、结构化输出与整编链路：

```bash
export CRITIC_API_KEY='YOUR_KEY'

u1-critic-v41-run \
  --input-root /path/to/images \
  --output-root /path/to/output \
  --base-url https://your-openai-compatible-endpoint \
  --model YOUR_MODEL_NAME \
  --api-key-env CRITIC_API_KEY \
  --workflow adversarial_consensus \
  --coverage-mode harness_v41 \
  --limit-images 1 \
  --max-in-flight 12 \
  --max-images-in-flight 1 \
  --timeout 180 \
  --max-tokens 16000 \
  --retries 2 \
  --consolidation-retries 2 \
  --unit-retries 2
```

冒烟通过后再运行全量。以下并发值只是已使用过的起点，必须按服务容量调整：

```bash
u1-critic-v41-run \
  --input-root /path/to/images \
  --input-jsonl /path/to/input_allowlist.jsonl \
  --output-root /path/to/output \
  --base-url https://your-openai-compatible-endpoint \
  --model YOUR_MODEL_NAME \
  --api-key-env CRITIC_API_KEY \
  --workflow adversarial_consensus \
  --coverage-mode harness_v41 \
  --all --confirm-run-all YES \
  --max-in-flight 48 \
  --max-images-in-flight 8 \
  --timeout 180 \
  --max-tokens 16000 \
  --retries 2 \
  --consolidation-retries 2 \
  --unit-retries 2
```

Allowlist 默认读取 JSONL 的 `image_absolute_path` 字段；示例见 `examples/input_allowlist.jsonl`。

## 生成与查看报告

```bash
u1-critic-v41-report --output-root /path/to/output --omit-model-traces
u1-critic-v41-serve --output-root /path/to/output --host 0.0.0.0 --port 11223
```

浏览器打开转发后的端口即可。bbox 在前端绘制，不生成额外标框图片。若需保留逐调用轨迹展示，生成报告时去掉 `--omit-model-traces`。

## 主要输出

- `manifest.json`：输入图片清单，只记录路径，不复制图片。
- `run_config.json`：运行参数、模型、版本与 bbox 坐标协议。
- `results.jsonl`：成功单元，V4.1 的正式结果位于 `consolidated_result`。
- `errors.jsonl`：正式失败单元及异常信息。
- `traces.jsonl`：完整阶段与模型调用轨迹。
- `index.html`：执行报告生成后的可视化入口。

程序支持断点续跑：再次使用相同输出目录时，会跳过 `results.jsonl` 中已经成功的图片/工作流单元。

## 开发验证

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

`src/u1_critic_v41/consolidate_existing.py` 仅用于历史实验排障，不属于正式 V4.1 成功链路，交付使用时禁止拿它修补失败结果。

本包不包含模型密钥、线上地址、原始图片和历史实验结果。
