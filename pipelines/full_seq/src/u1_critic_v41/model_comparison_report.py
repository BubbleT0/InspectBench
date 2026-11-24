from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .report import _script_json


WORKFLOW = "adversarial_consensus"


def _unit_summary(unit: dict[str, Any]) -> dict[str, Any]:
    result = unit.get("result") if isinstance(unit, dict) else None
    consolidated = unit.get("consolidated_result") if isinstance(unit, dict) else None
    error = unit.get("error") if isinstance(unit, dict) else None
    if not isinstance(result, dict):
        return {
            "status": "error" if error else "pending",
            "error": error,
            "verdict": "error" if error else "pending",
            "confidence": None,
            "coverage_status": "",
            "coverage_channels": {},
            "edit_regions": [],
            "rationale": "",
        }
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    final = consolidated if isinstance(consolidated, dict) else {}
    return {
        "status": "ok" if final else "error",
        "error": error or unit.get("consolidation_errors", []),
        "verdict": final.get("verdict", result.get("verdict", "uncertain")),
        "confidence": final.get("confidence", result.get("confidence")),
        "coverage_status": metadata.get("coverage_status", ""),
        "coverage_channels": metadata.get("coverage_channels", {}),
        "edit_regions": final.get("edit_regions", []),
        "rationale": final.get("rationale", ""),
    }


def _load_model(root: Path, label: str) -> dict[str, Any]:
    def read_jsonl(name: str) -> list[dict[str, Any]]:
        path = root / name
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.open(encoding="utf-8")
            if line.strip()
        ]

    result_rows = {
        row["sample_id"]: row
        for row in read_jsonl("results.jsonl")
        if row.get("workflow") == WORKFLOW and row.get("status") == "ok"
    }
    error_rows = {
        row["sample_id"]: row
        for row in read_jsonl("errors.jsonl")
        if row.get("workflow") == WORKFLOW
    }
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    records = {}
    for image in manifest["images"]:
        sample_id = image["image_id"]
        row = result_rows.get(sample_id)
        records[sample_id] = _unit_summary(
            {
                "result": row.get("result") if row else None,
                "consolidated_result": (
                    row.get("consolidated_result") if row else None
                ),
                "error": error_rows.get(sample_id),
            }
        )
    verdicts = Counter(row["verdict"] for row in records.values())
    return {
        "label": label,
        "root": str(root.resolve()),
        "records": records,
        "stats": {
            "success": sum(row["status"] == "ok" for row in records.values()),
            "error": sum(row["status"] == "error" for row in records.values()),
            "edit_regions": sum(len(row["edit_regions"]) for row in records.values()),
            "verdicts": dict(verdicts),
        },
    }


def _direct_issues(critic: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parent in ("artifact", "text_correctness"):
        group = critic.get(parent, {})
        if not isinstance(group, dict):
            continue
        for category, leaf in group.items():
            if not isinstance(leaf, dict):
                continue
            severity = str(leaf.get("severity", ""))
            for issue in leaf.get("issues", []):
                if not isinstance(issue, dict):
                    continue
                bbox = issue.get("bbox_2d", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                output.append(
                    {
                        "artifact_type": category,
                        "observation": str(issue.get("description", "")),
                        "abnormality": str(issue.get("description", "")),
                        "materiality": severity,
                        "confidence": critic.get("confidence"),
                        "bbox_norm": bbox,
                        "location": str(issue.get("location", "")),
                    }
                )
    return output


def _load_direct_model(
    root: Path,
    label: str,
    images: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path_to_id = {
        Path(row["source_path"]).resolve(): sample_id
        for sample_id, row in images.items()
    }
    records: dict[str, dict[str, Any]] = {
        sample_id: {
            "status": "pending",
            "error": None,
            "verdict": "pending",
            "confidence": None,
            "coverage_status": "单轮直调不提供覆盖证明",
            "coverage_channels": {},
            "edit_regions": [],
            "rationale": "",
        }
        for sample_id in images
    }
    for result_path in sorted(root.glob("*.critic.jsonl")):
        for line in result_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source", {})
            image_path = source.get("image_path")
            if not image_path:
                continue
            sample_id = path_to_id.get(Path(image_path).resolve())
            if sample_id is None:
                continue
            if row.get("status") != "ok" or not isinstance(
                row.get("critic"), dict
            ):
                records[sample_id] = {
                    **records[sample_id],
                    "status": "error",
                    "error": row.get("error", "direct critic failed"),
                    "verdict": "error",
                }
                continue
            critic = row["critic"]
            issues = _direct_issues(critic)
            regions = [
                {
                    "edit_region_id": f"direct_{index:03d}",
                    "edit_target": issue["location"],
                    "edit_region_rationale": (
                        "单轮模型直接定位的问题区域；未经覆盖账本、"
                        "Skeptic、Judge 或在线整编。"
                    ),
                    "bbox_norm": issue["bbox_norm"],
                    "confidence": issue["confidence"],
                    "issues": [issue],
                }
                for index, issue in enumerate(issues, 1)
            ]
            records[sample_id] = {
                **records[sample_id],
                "status": "ok",
                "verdict": (
                    "artifact_present"
                    if critic.get("has_quality_issue")
                    else "no_artifact"
                ),
                "confidence": critic.get("confidence"),
                "edit_regions": regions,
                "rationale": str(critic.get("summary", "")),
            }
    verdicts = Counter(row["verdict"] for row in records.values())
    return {
        "label": label,
        "root": str(root.resolve()),
        "records": records,
        "stats": {
            "success": sum(row["status"] == "ok" for row in records.values()),
            "error": sum(row["status"] != "ok" for row in records.values()),
            "edit_regions": sum(
                len(row["edit_regions"]) for row in records.values()
            ),
            "verdicts": dict(verdicts),
        },
    }


def build_model_comparison_report(
    left_root: Path,
    left_label: str,
    right_root: Path,
    right_label: str,
    output_root: Path,
    direct_root: Path | None = None,
    direct_label: str = "GPT 单轮直调",
) -> Path:
    left_manifest = json.loads((left_root / "manifest.json").read_text("utf-8"))
    right_manifest = json.loads((right_root / "manifest.json").read_text("utf-8"))
    left_images = {row["image_id"]: row for row in left_manifest["images"]}
    right_images = {row["image_id"]: row for row in right_manifest["images"]}
    if set(left_images) != set(right_images):
        raise ValueError("Model comparison requires identical sample IDs")
    for image_id, left_image in left_images.items():
        if Path(left_image["source_path"]).resolve() != Path(
            right_images[image_id]["source_path"]
        ).resolve():
            raise ValueError(f"Model comparison source mismatch: {image_id}")

    models = [
        _load_model(left_root, left_label),
        _load_model(right_root, right_label),
    ]
    if direct_root is not None:
        models.append(
            _load_direct_model(direct_root, direct_label, left_images)
        )
    records = []
    disagreements = 0
    for image in left_manifest["images"]:
        sample_id = image["image_id"]
        units = [model["records"][sample_id] for model in models]
        disagreement = len({unit["verdict"] for unit in units}) > 1
        disagreements += int(disagreement)
        records.append(
            {
                "sample_id": sample_id,
                "disagreement": disagreement,
                "units": units,
            }
        )

    payload = {
        "labels": [model["label"] for model in models],
        "models": [{"label": model["label"], "stats": model["stats"]} for model in models],
        "records": records,
        "disagreements": disagreements,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(left_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index = output_root / "index.html"
    index.write_text(_html(payload), encoding="utf-8")
    return index


def _html(payload: dict[str, Any]) -> str:
    return """<!doctype html><html lang=zh-CN><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Critic 模型与管线结果对比</title>
<style>
:root{--bg:#f4f6f8;--card:#fff;--line:#d9dee5;--ink:#172033;--muted:#677184;--accent:#2563eb;--bad:#b42318;--ok:#067647}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:5;padding:14px 22px;background:#ffffffed;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}h1{font-size:21px;margin:0 0 10px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar input{min-width:250px;padding:8px 10px;border:1px solid var(--line);border-radius:8px}.toolbar label{color:var(--muted)}.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}.stat{background:#eef2ff;padding:6px 10px;border-radius:7px}.wrap{max-width:2200px;margin:auto;padding:16px}.sample{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:18px;overflow:hidden}.sample-head{display:flex;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--line)}.disagree{color:var(--bad);font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:14px;padding:14px}.model{min-width:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.model h2{font-size:16px;margin:0;padding:9px 12px;background:#f8fafc;border-bottom:1px solid var(--line)}.visual{position:relative;background:#111;display:flex;justify-content:center;align-items:center;min-height:520px}.visual img{display:block;max-width:100%;width:auto;height:auto;max-height:78vh}.overlay{position:absolute;pointer-events:none}.box{position:absolute;border:3px solid var(--accent);box-shadow:0 0 0 1px #fff8 inset}.box span{position:absolute;left:-2px;top:-25px;background:var(--accent);color:#fff;border-radius:4px;padding:1px 6px;font-weight:700}.content{padding:12px}.verdict{display:inline-block;padding:2px 7px;border-radius:999px;background:#eef2ff;font-weight:700}.artifact_present{color:var(--bad)}.no_artifact{color:var(--ok)}.region{margin-top:10px;padding:10px;border-left:4px solid var(--accent);background:#f8fafc}.region h3{font-size:14px;margin:0 0 5px}.issue{margin:5px 0;color:#253047}.meta{color:var(--muted)}details{margin-top:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:10px;border-radius:7px;max-height:360px;overflow:auto}.hidden{display:none}@media(max-width:1000px){.grid{grid-template-columns:1fr}.visual{min-height:360px}}
</style></head><body><header><h1>Critic 模型与管线结果对比</h1><div class=toolbar><input id=query placeholder="按 sample_id 搜索"><label><input id=onlyDiff type=checkbox> 只看结论分歧</label></div><div id=stats class=stats></div></header><main id=main class=wrap></main>
<script>const DATA=""" + _script_json(payload) + """;const esc=x=>String(x??"").replace(/[&<>\"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function fit(img){const visual=img.parentElement,overlay=visual.querySelector('.overlay'),ir=img.getBoundingClientRect(),vr=visual.getBoundingClientRect();overlay.style.left=(ir.left-vr.left)+'px';overlay.style.top=(ir.top-vr.top)+'px';overlay.style.width=ir.width+'px';overlay.style.height=ir.height+'px'}
function boxes(unit){return(unit.edit_regions||[]).map((r,i)=>{const b=r.bbox_norm||[];if(b.length!==4)return'';const[x1,y1,x2,y2]=b;return`<div class=box style="left:${x1/10}%;top:${y1/10}%;width:${(x2-x1)/10}%;height:${(y2-y1)/10}%"><span>${i+1}</span></div>`}).join('')}
function issues(region){return(region.issues||[]).map(x=>`<div class=issue><b>${esc(x.artifact_type)}</b>：${esc(x.observation)}；${esc(x.abnormality)}</div>`).join('')}
function unit(u,label){const regions=(u.edit_regions||[]).map((r,i)=>`<div class=region><h3>${i+1}. ${esc(r.edit_target)}</h3>${issues(r)}<div>${esc(r.edit_region_rationale)}</div><div class=meta>bbox [${esc((r.bbox_norm||[]).join(', '))}] · 置信度 ${esc(r.confidence)}</div></div>`).join('')||'<div class=region>没有需要局部重绘的区域</div>';return`<section class=model><h2>${esc(label)}</h2><div class=visual><img loading=lazy decoding=async src="/images/${esc(currentId)}" onload="fit(this)"><div class=overlay>${boxes(u)}</div></div><div class=content><span class="verdict ${esc(u.verdict)}">${esc(u.verdict)}</span> · 置信度 ${esc(u.confidence)}<div class=meta>覆盖状态：${esc(u.coverage_status||'未知')}</div>${regions}<details><summary>查看最终整编说明</summary><pre>${esc(u.rationale)}</pre></details>${u.status==='error'?`<details open><summary>失败信息</summary><pre>${esc(JSON.stringify(u.error,null,2))}</pre></details>`:''}</div></section>`}
let currentId='';function card(r){currentId=r.sample_id;return`<article class=sample><div class=sample-head><b>${esc(r.sample_id)}</b>${r.disagreement?'<span class=disagree>最终结论不一致</span>':'<span>最终结论一致</span>'}</div><div class=grid>${r.units.map((u,i)=>unit(u,DATA.labels[i])).join('')}</div></article>`}
function render(){const q=query.value.trim().toLowerCase(),diff=onlyDiff.checked;main.innerHTML=DATA.records.filter(r=>(!q||r.sample_id.toLowerCase().includes(q))&&(!diff||r.disagreement)).map(card).join('');requestAnimationFrame(()=>document.querySelectorAll('.visual img').forEach(i=>i.complete&&fit(i)))}
stats.innerHTML=DATA.models.map(m=>`<div class=stat><b>${esc(m.label)}</b>：成功 ${m.stats.success} · 失败 ${m.stats.error} · 编辑区域 ${m.stats.edit_regions}</div>`).join('')+`<div class=stat>结论分歧 ${DATA.disagreements}</div>`;query.oninput=render;onlyDiff.onchange=render;window.onresize=()=>document.querySelectorAll('.visual img').forEach(fit);render();</script></body></html>"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", required=True, type=Path)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-root", required=True, type=Path)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--direct-root", type=Path)
    parser.add_argument("--direct-label", default="GPT 单轮直调")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        build_model_comparison_report(
            args.left_root,
            args.left_label,
            args.right_root,
            args.right_label,
            args.output_root,
            direct_root=args.direct_root,
            direct_label=args.direct_label,
        )
    )


if __name__ == "__main__":
    main()
