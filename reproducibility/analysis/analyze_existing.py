#!/usr/bin/env python3
"""Reanalyse saved predictions only. No model calls or training.

Run from any directory: python path/to/analyze_existing.py
Requires numpy and scipy. Baseline is checked against the submitted aggregates
and independently against the submitted specification scorer for every image.
"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OFFICIAL = ROOT / 'writing/official_paper'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bundle_mode = (HERE.parent/'inputs/configurations.json').is_file()
if bundle_mode:
    ROOT=HERE.parent
    OFFICIAL=ROOT
sc = module(OFFICIAL/'scorer/bbox_scorer.py', 'submitted_scorer')
# Read the configuration dictionary via AST, without importing the run script.
import ast
if bundle_mode:
    meta=json.loads((ROOT/'inputs/configurations.json').read_text())
    env={'ANNOTATIONS':[ROOT/'inputs/annotations.jsonl']}
    configs={r['label']:ROOT/r['file'] for r in meta}
    names={r['label']:r['name'] for r in meta}
    expected={r['label']:r for r in meta}
else:
    tree = ast.parse((ROOT/'data_0804/new_results/score_all_0917.py').read_text())
    env = {'ROOT': ROOT/'data_0804', 'OUT': ROOT/'data_0804/new_results'}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ANNOTATIONS','CONFIGS','ZEROSHOT_0917_CONFIGS') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'config','exec'),env)
    configs = {**env['CONFIGS'], **env['ZEROSHOT_0917_CONFIGS']}
    names = {r['original_configuration']:r['model_name']+' / '+r['pipeline_name'] for r in csv.DictReader((OFFICIAL/'author_notes/MODEL_PIPELINE_MAPPING.csv').open(encoding='utf-8-sig'))}
annotations = sc.read_jsonl(env['ANNOTATIONS'])
sources = {}
for p in env['ANNOTATIONS']:
    source = 'GPT-image-2' if p.name.startswith('GPT') else 'Public-4' if p.name.startswith('External') else 'Natural' if p.name.startswith('Natural') else 'SenseNova-U1.5'
    for sid,a in sc.read_jsonl([p]).items(): sources[sid] = a['source_group'] if bundle_mode else source
ids = list(annotations)
assert len(ids)==1039
assert sum(len(a['annotation']['boxes']) for a in annotations.values())==1983

def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def penalty(r,t,k):return math.exp(-k*max(0.,abs(math.log(r))-math.log(t))**2)

def geometry(a,p):
    w,h=a['image_width'],a['image_height']
    gs=[sc.xywh(b['bbox_xywh'],w,h) for b in a['annotation']['boxes']]
    ps=[sc.xywh(b['bbox_1000_xywh'],w,h,True) for b in (p.get('boxes',[]) if p and p.get('status')=='ok' else [])]
    return gs,ps,w*h

def evaluate(gs,ps,ia,quality_first=False,k=4.,beta=.5):
    m,n=len(gs),len(ps)
    pairs=[[sc.pair_stats(p,g,ia) for g in gs] for p in ps]
    # For penalty sensitivity, change pair-election quality as well as group quality.
    if k!=4:
        for j,p in enumerate(ps):
            for t,g in enumerate(gs):
                pairs[j][t]['score']=pairs[j][t]['dice']**.7*penalty(sc.area(p)/sc.area(g),sc.tolerance(sc.area(g)/ia),k)
    owners=[]
    reversals=[]
    for t in range(m):
        candidates=[j for j in range(n) if pairs[j][t]['coverage']>0]
        if not candidates: owners.append(None);continue
        cq=max(candidates,key=lambda j:(pairs[j][t]['coverage'],pairs[j][t]['score'],-j))
        qq=max(candidates,key=lambda j:(pairs[j][t]['score'],pairs[j][t]['coverage'],-j))
        owners.append(qq if quality_first else cq)
        gap=pairs[qq][t]['score']-pairs[cq][t]['score']
        if gap>1e-12:reversals.append((t,gap))
    qualities=[0.]*m;credits=[0.]*n
    for j,p in enumerate(ps):
        owned=[t for t in range(m) if owners[t]==j]
        if not owned:continue
        ua=sc.union_area(gs[t] for t in owned)
        ti=sc.union_area(sc.intersection(p,gs[t]) for t in owned)/sc.area(p)
        tau=sc.tolerance(ua/ia)*len(owned)**beta
        psi=penalty(sc.area(p)/ua,tau,k)
        for t in owned:qualities[t]=sc.harmonic(pairs[j][t]['coverage'],ti)**.7*psi
        credits[j]=sum(qualities[t] for t in owned)/len(owned)
    r=sum(qualities)/m if m else 0
    p=sum(credits)/n if n else 0
    return dict(M=m,N=n,F=sc.harmonic(p,r) if m else math.exp(-n),qualities=qualities,zero_overlap=owners.count(None),owners=owners,reversals=reversals)

def summary(rows,weight=.7):
    pos=[r for r in rows if r['M']];neg=[r for r in rows if not r['M']]
    ng=sum(r['M'] for r in pos)
    qp=sum(math.sqrt(r['M'])*r['F'] for r in pos)/sum(math.sqrt(r['M']) for r in pos) if pos else None
    qn=np.mean([r['F'] for r in neg]) if neg else None
    return dict(images=len(rows),flawed=len(pos),clean=len(neg),regions=ng,
        S=100*(weight*math.sqrt(qp)+(1-weight)*math.sqrt(qn))**2 if qp is not None and qn is not None else None,
        Qplus=qp,Qminus=qn,R01=100*sum(q>=.1 for r in pos for q in r['qualities'])/ng if ng else None,
        R05=100*sum(q>=.5 for r in pos for q in r['qualities'])/ng if ng else None,
        FPR=100*sum(r['N']>0 for r in neg)/len(neg) if neg else None,
        sensitivity=100*sum(r['N']>0 for r in pos)/len(pos) if pos else None,
        zero_overlap=100*sum(r['zero_overlap'] for r in pos)/ng if ng else None)

allrows={};subgroups=[];ownership=[];variants=[];matching=[];validation=[];manifest=[];examples=[]
variant_specs={'quality_first':dict(quality_first=True),'penalty_0':dict(k=0),'penalty_2':dict(k=2),'penalty_8':dict(k=8),'group_0':dict(beta=0),'group_025':dict(beta=.25),'group_075':dict(beta=.75)}
for label,path in configs.items():
    name=names[label];preds=sc.read_jsonl([path]);rows=[];vr={v:[] for v in variant_specs}
    iouhits={.1:0,.5:0};npred=0;maxerr=0.
    for sid in ids:
        a=annotations[sid];p=preds.get(sid);gs,ps,ia=geometry(a,p)
        r=evaluate(gs,ps,ia);r['sample_id']=sid;rows.append(r)
        frozen=sc.score_sample(a,p)
        err=max([abs(r['F']-frozen['F'])]+[abs(x-y) for x,y in zip(r['qualities'],frozen['qualities'])])
        maxerr=max(maxerr,err);assert err<1e-10,(name,sid,err)
        for v,kwargs in variant_specs.items():vr[v].append(evaluate(gs,ps,ia,**kwargs))
        # Class-agnostic maximum-cardinality one-to-one IoU matching. A small
        # tie term maximizes IoU only among equal-cardinality assignments.
        if gs and ps:
            iou=np.array([[sc.area(sc.intersection(p,g))/(sc.area(p)+sc.area(g)-sc.area(sc.intersection(p,g))) for g in gs] for p in ps])
            for t in iouhits:
                eligible=iou>=t
                cost=eligible.astype(float)+iou/(min(len(gs),len(ps))+1)*1e-3
                pi,gi=linear_sum_assignment(-cost)
                iouhits[t]+=int(eligible[pi,gi].sum())
        npred+=len(ps)
        qr=vr['quality_first'][-1]
        for g,gap in r['reversals']:
            examples.append(dict(configuration=name,sample_id=sid,gt_index=g,base_quality_gap=gap,original_final_credit=r['qualities'][g],quality_first_final_credit=qr['qualities'][g]))
    allrows[label]=rows
    baseline=summary(rows)
    original=expected[label]['expected'] if bundle_mode else json.loads((ROOT/'data_0804/new_results'/label.replace(' ','_').replace('/','_')/'metrics.json').read_text())
    assert abs(baseline['S']-original['headline']['elastic_bbox_score'])<1e-8,(name,baseline['S'])
    assert abs(baseline['R01']/100-original['detection']['recall_0_1'])<1e-8
    validation.append(dict(configuration=name,score=baseline['S'],max_per_image_discrepancy=maxerr,submitted_score=original['headline']['elastic_bbox_score']))
    variants.append(dict(configuration=name,variant='original',**baseline))
    for v,vrows in vr.items():variants.append(dict(configuration=name,variant=v,**summary(vrows)))
    for weight in [.5,.6,.8,.9]:variants.append(dict(configuration=name,variant=f'lambda_{weight}',**summary(rows,weight)))
    for source in ['GPT-image-2','SenseNova-U1.5','Public-4','Natural','without_natural','generated_matched_sources']:
        def include(sid):
            return sources[sid]!='Natural' if source=='without_natural' else sources[sid] in ['GPT-image-2','SenseNova-U1.5'] if source=='generated_matched_sources' else sources[sid]==source
        subset=[r for r in rows if include(r['sample_id'])]
        subgroups.append(dict(configuration=name,source=source,**summary(subset)))
    qr=vr['quality_first']
    gain=sum(q>=.1 and p<.1 for x,y in zip(rows,qr) for p,q in zip(x['qualities'],y['qualities']))
    loss=sum(q<.1 and p>=.1 for x,y in zip(rows,qr) for p,q in zip(x['qualities'],y['qualities']))
    ownership.append(dict(configuration=name,regions=1983,any_base_quality_reversal=sum(len(r['reversals']) for r in rows),base_gap_at_least_01=sum(g>=.1 for r in rows for _,g in r['reversals']),images_with_reversal=sum(bool(r['reversals']) for r in rows),R01_original=baseline['R01'],R01_quality_first=summary(qr)['R01'],rescued_regions=gain,lost_regions=loss,S_original=baseline['S'],S_quality_first=summary(qr)['S']))
    matching.append(dict(configuration=name,total_predictions=npred,**{f'recall_iou_{t}':100*h/1983 for t,h in iouhits.items()},**{f'precision_iou_{t}':100*h/npred if npred else 0 for t,h in iouhits.items()}))
    manifest.append(dict(configuration=name,prediction_file=str(path.relative_to(ROOT)),sha256=sha(path),prediction_records=len(preds),evaluated_records=len(ids),extra_ids_excluded=len(set(preds)-set(ids)),missing_ids=len(set(ids)-set(preds))))
    print(name,round(baseline['S'],4),'validated',flush=True)

# Stratified paired image bootstrap, same draws for every configuration.
B=10000;seed=20261112;rng=np.random.default_rng(seed)
mm=np.array([len(annotations[s]['annotation']['boxes']) for s in ids]);pos=np.flatnonzero(mm);neg=np.flatnonzero(mm==0)
pi=rng.choice(pos,(B,len(pos)));ni=rng.choice(neg,(B,len(neg)))
weights=np.sqrt(mm);draws={}
for label,rows in allrows.items():
    ff=np.array([r['F'] for r in rows]);hit=np.array([sum(q>=.1 for q in r['qualities']) for r in rows]);nn=np.array([r['N'] for r in rows])
    qp=(ff[pi]*weights[pi]).sum(1)/weights[pi].sum(1);qn=ff[ni].mean(1)
    draws[label]=dict(S=100*(.7*np.sqrt(qp)+.3*np.sqrt(qn))**2,R01=100*hit[pi].sum(1)/mm[pi].sum(1),FPR=100*(nn[ni]>0).mean(1))
pairs=[('0722_ck145','0722_ck145 + v6'),('0722_ck145 + v4','0722_ck145 + v6'),('0722_ck145 + v6','0722_ck145 + v6.3'),('0722_ck145','0722_ck145 + v6.3'),('0820_ck1001','0820_ck1001 + v6.3'),('0826_ck136','0826_ck136 + v6.3'),('0722_ck145','0826_ck136'),('0826_ck136','0722_ck145 + v6'),('0722_ck145 + v6','0826_ck136 + v6.3')]
paired=[]
for a,b in pairs:
    aa,bb=summary(allrows[a]),summary(allrows[b])
    for metric in ['S','R01','FPR']:
        dif=draws[b][metric]-draws[a][metric];lo,hi=np.quantile(dif,[.025,.975])
        paired.append(dict(baseline=names[a],comparison=names[b],metric=metric,difference=bb[metric]-aa[metric],lower95=float(lo),upper95=float(hi),replicates=B,seed=seed))

# Aggregate ranking sensitivity; all settings are reported, including reversals.
rank=[]
ref=[v for v in variants if v['variant']=='original']
for variant in sorted({v['variant'] for v in variants}):
    vs=[v for v in variants if v['variant']==variant]
    ss=[v['S'] for v in vs];rr=[v['R01'] for v in vs]
    rank.append(dict(variant=variant,S_spearman=float(spearmanr([v['S'] for v in ref],ss).statistic),R01_spearman=float(spearmanr([v['R01'] for v in ref],rr).statistic),best_S=vs[int(np.argmax(ss))]['configuration'],max_R01=max(rr)))

for filename,data in [('source_strata.csv',subgroups),('ownership.csv',ownership),('sensitivity.csv',variants),('matching.csv',matching),('paired_bootstrap.csv',paired),('validation.csv',validation),('ranking_sensitivity.csv',rank),('ownership_examples.csv',examples)]:write_csv(filename,data)
(OUT/'input_manifest.json').write_text(json.dumps(dict(predictions=manifest,annotations=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p)) for p in env['ANNOTATIONS']],scorer_sha256=sha(OFFICIAL/'scorer/bbox_scorer.py'),bootstrap=dict(replicates=B,seed=seed,rng='numpy.default_rng',strata='flawed/clean',paired=True)),indent=2)+'\n')
# Mathematical counterexample, exact dimensions independent of any image content.
gs=[(0.,0.,100.,100.)];ps=[(0.,0.,99.,100.),(0.,0.,200.,200.)]
(OUT/'counterexample.json').write_text(json.dumps({'tight_only':evaluate(gs,ps[:1],1e6),'with_broad':evaluate(gs,ps,1e6),'quality_first':evaluate(gs,ps,1e6,quality_first=True)},indent=2)+'\n')
print('Saved all analyses to',OUT)
