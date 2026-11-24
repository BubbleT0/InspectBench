#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["source"], row["source_index"], row["image"])] = row
    return rows


def is_correct(row):
    return bool(row["target"]["no_error"]) == bool(row["prediction"]["no_error"])


def exact_mcnemar_two_sided(b, c):
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
        for i in range(k + 1)
    ]
    anchor = max(logs)
    lower = math.exp(anchor) * sum(math.exp(value - anchor) for value in logs)
    return min(1.0, 2 * lower)


def pair_stats(checkpoint, base, keys):
    counts = Counter()
    for key in keys:
        c_ok = is_correct(checkpoint[key])
        b_ok = is_correct(base[key])
        counts[(c_ok, b_ok)] += 1
    only_checkpoint = counts[(True, False)]
    only_base = counts[(False, True)]
    return {
        "samples": len(keys),
        "both_correct": counts[(True, True)],
        "only_checkpoint_correct": only_checkpoint,
        "only_base_correct": only_base,
        "both_wrong": counts[(False, False)],
        "accuracy_checkpoint": sum(is_correct(checkpoint[k]) for k in keys) / len(keys),
        "accuracy_base": sum(is_correct(base[k]) for k in keys) / len(keys),
        "accuracy_delta_checkpoint_minus_base": (only_checkpoint - only_base) / len(keys),
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(only_checkpoint, only_base),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    checkpoint, base = load(args.checkpoint), load(args.base)
    if checkpoint.keys() != base.keys():
        raise SystemExit("prediction key mismatch")
    keys = sorted(checkpoint)
    by_source = defaultdict(list)
    for key in keys:
        by_source[key[0]].append(key)
    result = {
        "all": pair_stats(checkpoint, base, keys),
        "by_source": {name: pair_stats(checkpoint, base, source_keys) for name, source_keys in sorted(by_source.items())},
        "prediction_label_counts": {
            "checkpoint": Counter("no_error" if checkpoint[k]["prediction"]["no_error"] else "error" for k in keys),
            "base": Counter("no_error" if base[k]["prediction"]["no_error"] else "error" for k in keys),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a = result["all"]
    markdown = f"""# checkpoint-500 vs base 保留测试集对照

- 测试集：四来源分层留出的 test，共 {a['samples']} 张；有错 346，无错 720。
- 请求参数：同一提示词、同一原图、temperature=0、max_tokens=2048；两个服务各 4 并发。
- 接口有效率：两模型均为 1066/1066，无失败。

## 结论

checkpoint-500 不具备可部署意义上的提升。它把全部 1066 张图片都预测为“有错”，有错召回达到 100%，但无错特异度为 0%，属于单类决策塌缩。

## 逐样本配对

|结果|图片数|
|---|---:|
|两者都正确|{a['both_correct']}|
|仅 checkpoint-500 正确|{a['only_checkpoint_correct']}|
|仅 base 正确|{a['only_base_correct']}|
|两者都错误|{a['both_wrong']}|

- checkpoint accuracy：{a['accuracy_checkpoint']:.4%}
- base accuracy：{a['accuracy_base']:.4%}
- accuracy 差值：{a['accuracy_delta_checkpoint_minus_base']:+.4%}
- McNemar exact 双侧 p：{a['mcnemar_exact_two_sided_p']:.6g}

尽管 checkpoint 的总体 accuracy 更高，但这是测试集中有错样本占 32.46% 与“全部报错”策略造成的，不能解释为有效泛化提升。
"""
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
