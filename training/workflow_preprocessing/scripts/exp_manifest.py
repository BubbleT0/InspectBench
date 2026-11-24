#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agen-sft-pipeline 实验清单工具:每个实验一份 manifest.json(贯穿各阶段, 含全量 env),
EXPERIMENTS.md 由所有 manifest 渲染。用法:
  init   <exp>                    新建/重置 manifest
  set    <exp> <dotkey> <value>   写字段(如 stages.qc.tier3 77 / env.EXP_SUFFIX xxx)
  setjson <exp> <dotkey> <json>   写结构化值
  render                          从所有 manifest 重渲 EXPERIMENTS.md
根目录: $ASP_ROOT (默认 /path/to/storage/multimodal_ckpt/research/autotrain)
"""
import json,os,sys,glob
ROOT=os.environ.get("ASP_ROOT","/path/to/storage/multimodal_ckpt/research/autotrain")
EXPDIR=f"{ROOT}/experiments"; TABLE=f"{ROOT}/EXPERIMENTS.md"
def mpath(e): return f"{EXPDIR}/{e}/manifest.json"
def load(e):
    p=mpath(e)
    return json.load(open(p)) if os.path.exists(p) else {"exp":e,"stages":{},"data":{},"env":{},"hparams":{}}
def save(e,m):
    os.makedirs(f"{EXPDIR}/{e}",exist_ok=True); json.dump(m,open(mpath(e),"w"),ensure_ascii=False,indent=2)
def dset(m,dot,val):
    ks=dot.split("."); d=m
    for k in ks[:-1]: d=d.setdefault(k,{})
    d[ks[-1]]=val
def render():
    exps=[json.load(open(f)) for f in sorted(glob.glob(f"{EXPDIR}/*/manifest.json"))]
    L=["# aGen 训练实验表  (由 agen-sft-pipeline manifest 自动渲染, 勿手改)","",
       "> 单一真源 = experiments/<exp>/manifest.json。改用 `exp_manifest.py set`。",""]
    for m in exps:
        h=m.get("hparams",{}); e=m.get("env",{}); d=m.get("data",{}); s=m.get("stages",{})
        L.append(f"## {m.get('exp')}  ·  `{m.get('job_id','-')}`  ·  {m.get('status','-')}  ·  {m.get('date','-')}")
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append(f"| meta(openai) | `{d.get('meta_openai','-')}` |")
        L.append(f"| meta(swift) | `{d.get('meta_swift','-')}` |")
        L.append(f"| swift 数据 | `{d.get('swift_dir','-')}` |")
        L.append(f"| train DATA_DIR | `{d.get('train_data_dir','-')}` |")
        L.append(f"| repeat_map | `{d.get('repeat_map','-')}` |")
        L.append(f"| cache | `{d.get('cache_dir','-')}` |")
        L.append(f"| 启动脚本 | `{m.get('launch_script','-')}` |")
        L.append(f"| train 脚本 | `{m.get('train_script','-')}` |")
        L.append(f"| 模型 | `{m.get('model','-')}` |")
        hp=" · ".join(f"{k}={v}" for k,v in h.items())
        L.append(f"| 超参 | {hp or '-'} |")
        ev=" · ".join(f"{k}={v}" for k,v in e.items())
        L.append(f"| **env 全量** | {ev or '-'} |")
        L.append(f"| 数据构成 | {m.get('data_note','-')} |")
        qc=s.get("qc",{})
        if qc: L.append(f"| 质检 | {json.dumps(qc,ensure_ascii=False)} |")
        val=s.get("validate",{})
        if val: L.append(f"| 校验 | {json.dumps(val,ensure_ascii=False)} |")
        L.append(f"| eval | {m.get('eval','-')} |")
        L.append(f"| 日志 | `{m.get('log_dir','-')}` |")
        L.append("")
    open(TABLE,"w").write("\n".join(L))
    print("rendered",len(exps),"exps ->",TABLE)
def main():
    a=sys.argv
    if a[1]=="init": save(a[2],{"exp":a[2],"stages":{},"data":{},"env":{},"hparams":{}}); print("init",a[2])
    elif a[1]=="set": m=load(a[2]); dset(m,a[3],a[4]); save(a[2],m); print("set",a[3],"=",a[4])
    elif a[1]=="setjson": m=load(a[2]); dset(m,a[3],json.loads(a[4])); save(a[2],m); print("setjson",a[3])
    elif a[1]=="render": render()
    else: print(__doc__)
if __name__=="__main__": main()
