import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.json"
OUTPUT = ROOT / "EXPERIMENTS.md"


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stages = data["stages"]
    lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="utf-8"))
    repos = lock["repositories"]
    smoke = stages.get("custom_loss_smoke", stages["optimizer_smoke"])
    lines = [
        f"# {data['exp']}",
        "",
        "> 本文件由 `scripts/render_experiment_manifest.py` 从 `manifest.json` 自动渲染；manifest 是唯一真源。",
        "",
        f"- 状态：`{data['status']}`",
        f"- 数据：train {data['data']['split']['train_rows']:,}，validation {data['data']['split']['validation_rows']:,}，按来源和标签分层，图像重叠 {data['data']['split']['train_validation_image_overlap']}。",
        f"- RP：`repeat_time={data['data']['repeat_time']}`，不进行上采样。",
        f"- 数据形态：`packing={str(data['data']['packing']).lower()}`，`padding_free={str(data['data']['padding_free']).lower()}`。",
        f"- 拓扑：{data['hparams']['nodes']} 节点 × {data['hparams']['gpus_per_node']} 卡，TP{data['hparams']['TP']} / PP{data['hparams']['PP']} / CP{data['hparams']['CP']} / DP{data['hparams']['DP']} / VPP{data['hparams']['VPP']}。",
        f"- 依赖锁：ms-swift `{repos['ms-swift']['commit'][:8]}`、Megatron-LM `{repos['Megatron-LM']['commit'][:8]}`、mcore-bridge `{repos['mcore-bridge']['commit'][:8]}`。",
        f"- QC：`{stages['qc']['status']}`；原因见 manifest。",
        f"- 当前门禁：任务 `{smoke.get('job_id', 'pending')}`，状态 `{smoke['status']}`，目标 {smoke.get('train_iters', 2)} step，packing={str(smoke.get('packing', False)).lower()}，padding_free={str(smoke.get('padding_free', False)).lower()}。",
        "- 后续：正式 R2 packed 训练 → 对 checkpoint 做 800 条分层生成式验证 → 按 clean specificity 门槛与 balanced accuracy 选型。",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
