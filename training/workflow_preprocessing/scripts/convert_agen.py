#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agentv4 回流 session -> 《Agent 数据格式规定》目标格式 转换器
- 每个 session = 一条 trajectory 记录
- 按 agentType / 类别(success|retry|failed|unknown|unindexed) 分文件
- 预留 thinking(reasoning_content) 抽取钩子: 现在回流为空转, 自刷带 think 数据自动生效
用法:
  python3 convert_agen.py --harness <.harness dir> --out <out dir> [--limit N] [--only-session SID] [--force-thinking on|off]
"""
import json, os, glob, argparse, collections, re, sys

SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>")
BAD_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>")

# ---------- 读取工具 ----------
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out

# ---------- 1) 建 session 索引 ----------
def build_index(harness):
    idx = {}
    for f in iter_workflow_files(harness, "session-index.jsonl", recursive=True):
        for o in load_jsonl(f):
            sid = o.get("sessionId")
            if not sid:
                continue
            rec = idx.setdefault(sid, {})
            if o.get("type") == "start":
                rec["runId"] = o.get("runId")
                rec["label"] = o.get("label")
                rec["agentType"] = o.get("agentType")
                rec["phase"] = o.get("phase")
                rec["parentSessionId"] = o.get("parentSessionId")
            elif o.get("type") == "end":
                rec["ok"] = o.get("ok")
                rec["tokens"] = o.get("tokens")
    # retry 判定: 同 (runId,label) 多个 session, 按 sessionId 时间序, 除最后一个都算 retry
    def sid_ts(sid):
        m = re.match(r"s-(\d+)-", sid)
        return int(m.group(1)) if m else 0
    grp = collections.defaultdict(list)
    for sid, r in idx.items():
        grp[(r.get("runId"), r.get("label"))].append(sid)
    retry = set()
    for k, sids in grp.items():
        if k[0] is None or k[1] is None:
            continue
        if len(sids) > 1:
            sids_sorted = sorted(sids, key=sid_ts)
            for sid in sids_sorted[:-1]:
                retry.add(sid)
    for sid in retry:
        idx[sid]["is_retry"] = True
    return idx

# ---------- case(顶层 run)成败 ----------
RUNID_RE = re.compile(r"AGEN_RUN_ID=(wf_[0-9a-f.]+)")

def build_case_status(harness):
    """返回 (known_tops:set, failed_tops:set)。

    case = 顶层 run。agentv4 的成功终态既可能是 EXECUTED（执行完成），
    也可能是 DELIVERED（交付完成）；NEEDS_REVIEW/REJECTED/TASK_FAILED 等
    非成功终态都按失败处理。
    """
    known, failed = set(), set()
    for f in iter_workflow_files(harness, "result.json"):
        wf = f.split(os.sep)[-2]
        known.add(wf)
        try:
            o = json.load(open(f, encoding="utf-8", errors="replace"))
        except Exception:
            failed.add(wf); continue
        if o.get("status") not in SUCCESS_CASE_STATUSES:
            failed.add(wf)
    return known, failed

def top_case_of(records, idx_rec):
    """从 transcript 首个 user prompt 的 AGEN_RUN_ID 取顶层 case; 兜底用 session-index runId"""
    for o in records:
        m = o.get("message", {})
        if m.get("role") != "user":
            continue
        c = m.get("content")
        txt = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        mm = RUNID_RE.search(txt)
        if mm:
            return mm.group(1).split(".")[0]
    rid = idx_rec.get("runId")
    return rid.split(".")[0] if rid else None

def classify(rec):
    if not rec:
        return "unindexed"
    if rec.get("is_retry"):
        return "retry"
    ok = rec.get("ok")
    if ok is False:
        return "failed"
    if ok is True:
        return "success"
    return "unknown"

# ---------- 全局目录: 工具名->schema, agentType->system (给缺 request.json 的 session 兜底) ----------
TOOL_CATALOG = {}          # name -> anthropic tool def
SYSTEM_BY_AGENTTYPE = {}   # agentType -> system text
SUCCESS_CASE_STATUSES = {"EXECUTED", "DELIVERED"}
WORKFLOW_DIR_NAMES = ("workflow-runs", "workflows")


def iter_workflow_files(harness, filename, recursive=False):
    """Yield workflow metadata files from both known harness layouts."""
    seen = set()
    for dirname in WORKFLOW_DIR_NAMES:
        pattern = os.path.join(
            harness,
            dirname,
            "**" if recursive else "wf_*",
            filename,
        )
        for path in glob.glob(pattern, recursive=recursive):
            if path not in seen:
                seen.add(path)
                yield path

def build_catalogs(harness, idx):
    d0 = os.path.join(harness, "logs")
    n = 0
    for sdir in glob.glob(os.path.join(d0, "*", "debug", "model")):
        sid = sdir.split(os.sep)[-3]
        reqs = sorted(glob.glob(os.path.join(sdir, "*.request.json")), key=_mcall_no)
        if not reqs:
            continue
        try:
            o = json.load(open(reqs[-1], encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for t in o.get("tools", []) or []:
            nm = t.get("name") or t.get("function", {}).get("name")
            if nm and nm not in TOOL_CATALOG:
                TOOL_CATALOG[nm] = t
        at = idx.get(sid, {}).get("agentType")
        if at and at not in SYSTEM_BY_AGENTTYPE:
            st = system_text(o.get("system"))
            if st:
                SYSTEM_BY_AGENTTYPE[at] = st
        n += 1
    return n

# ---------- 2) system + tools 来源: logs/<sid>/debug/model/*.request.json ----------
def _mcall_no(path):
    m = re.search(r"mcall_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

def load_agent_spec(harness, sid):
    d = os.path.join(harness, "logs", sid, "debug", "model")
    reqs = sorted(glob.glob(os.path.join(d, "*.request.json")), key=_mcall_no)
    if not reqs:
        return None
    # system/model/thinking 取任一; tools 取全 session 并集(不同 mcall 工具集可能不同)
    base = None
    tools_by_name = {}
    for p in reqs:
        try:
            o = json.load(open(p, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if base is None:
            base = o
        for t in o.get("tools", []) or []:
            nm = t.get("name") or t.get("function", {}).get("name")
            if nm and nm not in tools_by_name:
                tools_by_name[nm] = t
    if base is None:
        return None
    base = dict(base)
    base["tools"] = list(tools_by_name.values())
    return base

def system_text(system_field):
    if system_field is None:
        return ""
    if isinstance(system_field, str):
        return system_field
    parts = []
    for b in system_field if isinstance(system_field, list) else [system_field]:
        if isinstance(b, dict):
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)

def convert_tools(tools_field):
    out = []
    seen = set()
    for t in tools_field or []:
        name = t.get("name") or t.get("function", {}).get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        params = t.get("inputSchema") or t.get("input_schema") or t.get("parameters") \
            or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": params,
            },
        })
    return out

# ---------- reasoning 抽取钩子(预留 thinking 接口) ----------
def extract_reasoning(content_blocks):
    parts = []
    for b in content_blocks:
        if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
            parts.append(b.get("text") or b.get("thinking") or b.get("data") or "")
    txt = "\n".join(p for p in parts if p).strip()
    for tag in BAD_TAGS:
        txt = txt.replace(tag, "")
    return txt

# ---------- 3) 转换 transcript ----------
def blocks_text(content_blocks):
    return "\n".join(b.get("text", "") for b in content_blocks
                     if isinstance(b, dict) and b.get("type") == "text").strip()

def tool_result_text(tr_content):
    if isinstance(tr_content, str):
        return tr_content
    if isinstance(tr_content, list):
        return "\n".join(b.get("text", "") for b in tr_content
                         if isinstance(b, dict) and b.get("type") == "text").strip()
    return ""

def convert_transcript(records, system, tools):
    # id -> tool name (tool_result 不带名字)
    id2name = {}
    for o in records:
        for b in o.get("message", {}).get("content", []) if isinstance(o.get("message", {}).get("content"), list) else []:
            if b.get("type") == "tool_use":
                id2name[b.get("id")] = b.get("name")

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})

    for o in records:
        m = o.get("message", {})
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, list):
            if isinstance(content, str) and content:
                msgs.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = blocks_text(content)
            for tg in BAD_TAGS:            # 关 think 时偶尔把标签当正文吐出, 无损剥标签留正文
                text = text.replace(tg, "")
            text = text.strip()
            reasoning = extract_reasoning(content)
            tool_calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    args = b.get("input")
                    if not isinstance(args, dict):
                        args = {}
                    tool_calls.append({
                        "id": b.get("id"),
                        "type": "function",
                        "function": {"name": b.get("name"), "arguments": args},
                    })
            am = {"role": "assistant", "content": text}
            if tool_calls:
                am["tool_calls"] = tool_calls
            if reasoning:
                am["reasoning_content"] = reasoning
            # content 与 tool_calls 不能同时空
            if not text and not tool_calls:
                am["content"] = "(no content)"
            msgs.append(am)
        else:  # user 消息: 可能是首条 prompt(text) 或 tool_result
            has_tr = any(b.get("type") == "tool_result" for b in content)
            if has_tr:
                for b in content:
                    if b.get("type") == "tool_result":
                        c = tool_result_text(b.get("content"))
                        if not c:
                            c = "(empty)"
                        tid = b.get("toolUseId") or b.get("tool_use_id")
                        msgs.append({
                            "role": "tool",
                            "name": id2name.get(tid, "unknown"),
                            "tool_call_id": tid,
                            "content": c,
                            "success": not bool(b.get("is_error")),
                        })
            else:
                t = blocks_text(content)
                if t:
                    msgs.append({"role": "user", "content": t})
    return msgs

def merge_consecutive_assistants(msgs):
    out = []
    for m in msgs:
        if out and m["role"] == "assistant" and out[-1]["role"] == "assistant":
            prev = out[-1]
            c = "\n".join(x for x in [prev.get("content", ""), m.get("content", "")] if x)
            prev["content"] = c or "(no content)"
            tc = prev.get("tool_calls", []) + m.get("tool_calls", [])
            if tc:
                prev["tool_calls"] = tc
            r = "\n".join(x for x in [prev.get("reasoning_content", ""), m.get("reasoning_content", "")] if x)
            if r:
                prev["reasoning_content"] = r
        else:
            out.append(m)
    return out

def truncate_to_last_assistant(msgs):
    last = -1
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            last = i
    if last < 0:
        return msgs  # 无 assistant, 交给校验拒
    return msgs[:last + 1]

# ---------- 校验(实现 doc 硬规则, 官方 validator 前的自检) ----------
def validate(rec):
    errs = []
    msgs = rec.get("messages", [])
    if not msgs:
        errs.append("empty messages")
        return errs
    roles = [m["role"] for m in msgs]
    for m in msgs:
        if m["role"] not in ("system", "user", "assistant", "tool"):
            errs.append(f"bad role {m['role']}")
    if msgs[-1]["role"] != "assistant":
        errs.append("last message not assistant")
    # 非 assistant content 不能空
    for m in msgs:
        if m["role"] != "assistant":
            if not (isinstance(m.get("content"), str) and m["content"].strip()):
                errs.append(f"empty content in role={m['role']}")
    # 连续两条 assistant
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == "assistant" and msgs[i-1]["role"] == "assistant":
            errs.append("consecutive assistant")
            break
    # assistant content 与 tool_calls 不能同时空
    for m in msgs:
        if m["role"] == "assistant":
            if not (m.get("content") or "").strip() and not m.get("tool_calls"):
                errs.append("assistant content & tool_calls both empty")
    # tool_calls name 必须在 tools
    tool_names = {t["function"]["name"] for t in rec.get("tools", [])}
    ids_seen = set()
    for m in msgs:
        for tc in m.get("tool_calls", []) if m["role"] in ("assistant", "user") else []:
            fn = tc.get("function", {}).get("name")
            if fn not in tool_names:
                errs.append(f"toolcall name not in tools: {fn}")
            args = tc.get("function", {}).get("arguments")
            if not (isinstance(args, dict) or (isinstance(args, str) and args.strip())):
                errs.append("empty toolcall arguments")
            tid = tc.get("id")
            if tid in ids_seen:
                errs.append(f"dup toolcall id {tid}")
            ids_seen.add(tid)
    # 3 连全等 toolcall
    sig_run = []
    for m in msgs:
        if m["role"] == "user":
            sig_run = []
        for tc in m.get("tool_calls", []):
            sig = json.dumps(tc.get("function", {}), sort_keys=True, ensure_ascii=False)
            sig_run.append(sig)
            if len(sig_run) >= 3 and sig_run[-1] == sig_run[-2] == sig_run[-3]:
                errs.append("3 identical consecutive toolcalls")
    # 特殊 token / 标签
    def check_str(s, where):
        if not isinstance(s, str):
            return
        for tk in SPECIAL_TOKENS:
            if tk in s:
                errs.append(f"special token in {where}")
        for tg in BAD_TAGS:
            if tg in s and where in ("reasoning", "content"):
                errs.append(f"bad tag {tg} in {where}")
    for m in msgs:
        if m["role"] == "assistant":
            check_str(m.get("content"), "content")
            check_str(m.get("reasoning_content"), "reasoning")
    # reasoning 要求(仅当 enable_thinking): 至少一条 assistant 带 reasoning
    if rec.get("enable_thinking"):
        if not any(m["role"] == "assistant" and (m.get("reasoning_content") or "").strip() for m in msgs):
            errs.append("enable_thinking but no reasoning_content")
    return errs

# ---------- 主流程 ----------
def process_session(harness, sid, idx, known_tops, failed_tops, drop_unknown):
    tpath = os.path.join(harness, "sessions", sid, "transcript.jsonl")
    if not os.path.exists(tpath):
        return None, ["no transcript"]
    records = load_jsonl(tpath)
    if not records:
        return None, ["empty transcript"]
    # 跳过非对话桩(只有 local_command / 无任何 message 的 session)
    if not any("message" in o and isinstance(o.get("message", {}).get("content"), (list, str)) for o in records):
        return None, ["DROP:not-a-conversation"]
    rec_idx0 = idx.get(sid, {})
    # 缺 request.json -> 丢(无可靠 system/tools)
    spec = load_agent_spec(harness, sid)
    if spec is None:
        return None, ["DROP:no-request-json"]
    # case(顶层 run)成败 -> 失败/未知 case 整体丢
    top = top_case_of(records, rec_idx0)
    if top in failed_tops:
        return None, ["DROP:case-failed"]
    if top not in known_tops:
        if drop_unknown:
            return None, ["DROP:case-unknown"]
    system = system_text(spec.get("system"))
    tools = convert_tools(spec.get("tools"))
    msgs = convert_transcript(records, system, tools)
    msgs = merge_consecutive_assistants(msgs)
    # 末尾非 assistant = session 没跑完(停在 tool_result)-> 整条丢弃(不截断, 避免造出"以未回应工具调用结尾"的误导样本)
    real = [m for m in msgs if m["role"] != "system"]
    if not real or real[-1]["role"] != "assistant":
        return None, ["DROP:ends-non-assistant"]
    # 补齐被调用但 tools 里缺失的工具(从全局目录, 再不行合成最小 schema)
    have = {t["function"]["name"] for t in tools}
    called = set()
    for m in msgs:
        for tc in m.get("tool_calls", []):
            nm = tc.get("function", {}).get("name")
            if nm:
                called.add(nm)
    for nm in called - have:
        if nm in TOOL_CATALOG:
            tools.extend(convert_tools([TOOL_CATALOG[nm]]))
        else:
            tools.append({"type": "function", "function": {
                "name": nm, "description": "", "parameters": {"type": "object", "properties": {}}}})
    rec_idx = idx.get(sid, {})
    enable_thinking = any(m.get("role") == "assistant" and (m.get("reasoning_content") or "").strip() for m in msgs)
    total_steps = sum(1 for m in msgs if m["role"] == "assistant")
    cat = classify(rec_idx)
    out = {
        "status": "completed" if rec_idx.get("ok") else ("failed" if cat in ("failed", "retry") else "completed"),
        "total_steps": total_steps,
        "enable_thinking": enable_thinking,
        "messages": msgs,
        "tools": tools,
        "metadata": {
            "session_id": sid,
            "wf_id": rec_idx.get("runId"),
            "case": top,
            "agent_type": rec_idx.get("agentType") or "_unindexed",
            "label": rec_idx.get("label"),
            "category": cat,
            "model": (spec or {}).get("model"),
            "thinking_setting": json.dumps((spec or {}).get("thinking")) if spec else None,
            "source_transcript": tpath,
        },
    }
    # image/audio/video: 回流无内联多模 -> 不加键(避免 meta 路径存在性检查)
    errs = validate(out)
    return out, errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-session", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drop-unknown-case", action="store_true",
                    help="丢弃无法判定成败(run 目录缺失)的 case 下的 session")
    args = ap.parse_args()

    idx = build_index(args.harness)
    ncat = build_catalogs(args.harness, idx)
    known_tops, failed_tops = build_case_status(args.harness)
    print(f"catalogs: {len(TOOL_CATALOG)} tools (from {ncat} sessions)")
    print(f"cases: {len(known_tops)} known, {len(failed_tops)} failed; drop_unknown_case={args.drop_unknown_case}")
    sess = [p.split(os.sep)[-2] for p in glob.glob(os.path.join(args.harness, "sessions", "*", "transcript.jsonl"))]
    if args.only_session:
        sess = [s for s in sess if s == args.only_session]
    if args.limit:
        sess = sess[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    writers = {}
    stats = collections.Counter()
    err_stats = collections.Counter()
    invalid = []
    drop_stats = collections.Counter()
    for sid in sess:
        rec, errs = process_session(args.harness, sid, idx, known_tops, failed_tops, args.drop_unknown_case)
        if rec is None:
            drop_stats[errs[0] if errs else "skip"] += 1
            continue
        at = rec["metadata"]["agent_type"]
        cat = rec["metadata"]["category"]
        # 无效轨迹(过不了 doc 硬规则)直接丢弃, 不落盘
        if errs:
            drop_stats["DROP:invalid-format"] += 1
            for e in errs:
                err_stats[e] += 1
            invalid.append((sid, at, cat, errs))
            continue
        stats[f"{cat}"] += 1
        if args.dry_run:
            if args.only_session or args.limit <= 5:
                print(json.dumps(rec, ensure_ascii=False, indent=2)[:4000])
                print("VALIDATION: OK")
            continue
        # 分文件: <out>/<agentType>/<category>.jsonl
        ddir = os.path.join(args.out, at)
        os.makedirs(ddir, exist_ok=True)
        key = os.path.join(ddir, f"{cat}.jsonl")
        w = writers.get(key)
        if w is None:
            w = open(key, "a", encoding="utf-8")
            writers[key] = w
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for w in writers.values():
        w.close()
    print("=== 保留类别分布 ===", dict(stats))
    print("=== 丢弃原因 ===", dict(drop_stats))
    print("=== 校验错误 TOP ===")
    for e, c in err_stats.most_common(20):
        print(f"  {c:6d}  {e}")
    print(f"总处理 {len(sess)}, 无效 {len(invalid)}")

if __name__ == "__main__":
    main()
