#!/usr/bin/env python3
"""Specification-based implementation of merged-ground-truth-ownership-v2.

Derived from data/README_SCORER.md for this manuscript package. This is NOT the
original author implementation or its verify_equivalence.py. The accompanying
unit tests are synthetic; the benchmark CSV is never overwritten by this code.
Python >=3.9, standard library only. Rectangles are internal (x1,y1,x2,y2).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Rect = Tuple[float, float, float, float]
FAMILIES = {
    'D1_Artifact/A1_RenderingArtifact', 'D1_Artifact/A2_StructuralArtifact',
    'D1_Artifact/A3_TopologyArtifact', 'D1_Artifact/A4_RelationalAndPhysicalArtifact',
    'D2_TextCorrectness/A1_TextVisualIntegrity', 'D2_TextCorrectness/A2_TextContentFidelity',
}
VERSION = 'merged-ground-truth-ownership-v2'


def area(r: Rect) -> float:
    """Return a rectangle's continuous geometric area."""
    return max(0.0, r[2]-r[0])*max(0.0, r[3]-r[1])


def intersection(a: Rect, b: Rect) -> Rect:
    """Intersect two rectangles; an empty intersection has zero area."""
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    return x1,y1,max(x1,min(a[2],b[2])),max(y1,min(a[3],b[3]))


def union_area(rectangles: Iterable[Rect]) -> float:
    """Compute exact union area with x-boundary sweeping and merged y intervals."""
    rects=[r for r in rectangles if area(r)>0]
    xs=sorted({x for r in rects for x in (r[0],r[2])})
    total=0.0
    for left,right in zip(xs,xs[1:]):
        ys=sorted((r[1],r[3]) for r in rects if r[0]<right and r[2]>left)
        length=0.0; start=end=None
        for low,high in ys:
            if start is None:
                start,end=low,high
            elif low>end:
                length+=end-start;start,end=low,high
            else:
                end=max(end,high)
        if start is not None:length+=end-start
        total+=(right-left)*length
    return total


def tolerance(fraction: float) -> float:
    """Size-dependent symmetric area-ratio tolerance from the supplied scorer."""
    return 1.25+0.35*math.exp(-fraction/0.01)


def penalty(ratio: float, tau: float) -> float:
    """Gaussian penalty on log-area excess outside [1/tau,tau]."""
    return math.exp(-4*max(0.0,abs(math.log(ratio))-math.log(tau))**2)


def harmonic(a: float, b: float) -> float:
    """Harmonic mean with the explicit zero-denominator convention."""
    return 2*a*b/(a+b) if a+b>0 else 0.0


def pair_stats(prediction: Rect, truth: Rect, image_area: float) -> Dict[str,float]:
    """Return coverage, tightness, Dice and one-to-one election quality."""
    i=area(intersection(prediction,truth)); pa,ga=area(prediction),area(truth)
    if pa<=0 or ga<=0 or image_area<=0:raise ValueError('Areas must be positive.')
    c,t=i/ga,i/pa;d=harmonic(c,t)
    return dict(coverage=c,tightness=t,dice=d,score=d**.7*penalty(pa/ga,tolerance(ga/image_area)))


def xywh(values: Sequence[float], width: float, height: float, normalized: bool=False) -> Rect:
    """Validate a finite in-bounds box and convert its coordinate convention."""
    if len(values)!=4:raise ValueError('A box must have four xywh coordinates.')
    x,y,w,h=map(float,values)
    if not all(math.isfinite(v) for v in (x,y,w,h)):raise ValueError('Non-finite box.')
    if normalized:x,w,y,h=x*width/1000,w*width/1000,y*height/1000,h*height/1000
    if w<=0 or h<=0 or x<0 or y<0 or x+w>width+1e-8 or y+h>height+1e-8:
        raise ValueError(f'Invalid or out-of-bounds box: {(x,y,w,h)}')
    return x,y,x+w,y+h


def score_sample(annotation: Dict[str,Any], prediction: Optional[Dict[str,Any]]) -> Dict[str,Any]:
    """Score one image, including error/missing-as-empty and diagnostic counts."""
    sid=annotation['sample_id'];width=float(annotation['image_width']);height=float(annotation['image_height'])
    if not isinstance(sid,str) or not sid:raise ValueError('sample_id must be a nonempty string.')
    if not math.isfinite(width*height) or min(width,height)<=0:raise ValueError('Invalid image dimensions.')
    ann=annotation['annotation'];gb=ann['boxes']
    if not isinstance(ann.get('no_error'),bool) or ann['no_error'] != (len(gb)==0):
        raise ValueError('Ground-truth no_error and boxes disagree.')
    gs=[xywh(b['bbox_xywh'],width,height) for b in gb]
    families=[b['error_type']['code'] for b in gb]
    if any(f not in FAMILIES for f in families):raise ValueError('Unknown reference family.')
    if prediction is not None and prediction.get('sample_id')!=sid:raise ValueError('Sample ID mismatch.')
    failed=prediction is None or prediction.get('status')!='ok'
    pb=[] if failed else prediction.get('boxes',[])
    ps=[xywh(b['bbox_1000_xywh'],width,height,True) for b in pb]
    m,n=len(gs),len(ps);ia=width*height
    pairs=[[pair_stats(p,g,ia) for g in gs] for p in ps]
    owners=[]
    for k in range(m):
        candidates=[j for j in range(n) if pairs[j][k]['coverage']>0]
        owners.append(max(candidates,key=lambda j:(pairs[j][k]['coverage'],pairs[j][k]['score'],-j)) if candidates else None)
    groups=[[k for k in range(m) if owners[k]==j] for j in range(n)]
    qualities=[0.0]*m;credits=[0.0]*n
    for j,owned in enumerate(groups):
        if not owned:continue
        ua=union_area(gs[k] for k in owned)
        ti=union_area(intersection(ps[j],gs[k]) for k in owned)/area(ps[j])
        tau=tolerance(ua/ia)*math.sqrt(len(owned))
        psi=penalty(area(ps[j])/ua,tau)
        for k in owned:qualities[k]=harmonic(pairs[j][k]['coverage'],ti)**.7*psi
        credits[j]=sum(qualities[k] for k in owned)/len(owned)
    recall=sum(qualities)/m if m else 0.0
    precision=sum(credits)/n if n else 0.0
    image_score=harmonic(precision,recall) if m else math.exp(-n)
    duplicate=background=0
    for j,g in enumerate(groups):
        if g:continue
        is_duplicate=any(owners[k] is not None and pairs[j][k]['score']>.1 for k in range(m))
        duplicate+=int(is_duplicate);background+=int(not is_duplicate)
    class_correct=class_n=0
    for k,q in enumerate(qualities):
        if q>=.1 and owners[k] is not None:
            pred_family=pb[owners[k]].get('family')
            if pred_family in FAMILIES:
                class_n+=1;class_correct+=int(pred_family==families[k])
    return dict(sample_id=sid,M=m,N=n,Q=sum(qualities),P=precision,R=recall,F=image_score,
        failed=failed,owners=owners,qualities=qualities,credits=credits,gt_families=families,
        zero_overlap=sum(x is None for x in owners),effective_miss=sum(q<.1 for q in qualities),
        merged_predictions=sum(len(g)>=2 for g in groups),
        split_candidates=sum(sum(pairs[j][k]['score']>.1 for j in range(n))>=2 for k in range(m)),
        duplicate_fp=duplicate,background_fp=background,family_correct=class_correct,family_denominator=class_n)


def mean(values: Sequence[float]) -> Optional[float]:
    """Mean with null for an undefined empty population."""
    return sum(values)/len(values) if values else None


def aggregate(samples: List[Dict[str,Any]]) -> Dict[str,Any]:
    """Aggregate image/region metrics; pure-class headline score is undefined."""
    if not samples:raise ValueError('No samples to aggregate.')
    pos=[r for r in samples if r['M']>0];clean=[r for r in samples if r['M']==0]
    wp=sum(math.sqrt(r['M']) for r in pos)
    sp=sum(math.sqrt(r['M'])*r['F'] for r in pos)/wp if wp else None
    sn=mean([r['F'] for r in clean])
    score=100*(.7*math.sqrt(sp)+.3*math.sqrt(sn))**2 if sp is not None and sn is not None else None
    qs=[q for r in pos for q in r['qualities']];total_gt=len(qs)
    result=dict(metric_version=VERSION,images=len(samples),positive_images=len(pos),clean_images=len(clean),gt_regions=total_gt,
        region_bbox_score=score,positive_elastic=sp,clean_soft=sn,
        failed_or_missing_predictions=sum(r['failed'] for r in samples),
        failed_positive_predictions=sum(r['failed'] for r in pos),failed_clean_predictions=sum(r['failed'] for r in clean),
        image_fnr=mean([float(r['N']==0) for r in pos]),clean_fpr=mean([float(r['N']>0) for r in clean]),
        image_error=mean([float((r['M']>0)!=(r['N']>0)) for r in samples]),clean_boxes=mean([r['N'] for r in clean]),
        zero_overlap_miss=sum(r['zero_overlap'] for r in pos)/total_gt if total_gt else None,
        effective_miss=sum(q<.1 for q in qs),found_mean_quality=mean([q for q in qs if q>=.1]),
        count_mae=mean([abs(r['N']-r['M']) for r in samples]),count_bias=mean([r['N']-r['M'] for r in samples]),
        background_fp=sum(r['background_fp'] for r in samples),duplicate_fp=sum(r['duplicate_fp'] for r in samples),
        merged_predictions=sum(r['merged_predictions'] for r in samples),split_candidates=sum(r['split_candidates'] for r in samples))
    for t,suffix in [(.1,'0_1'),(.5,'0_5'),(.75,'0_75')]:
        result['recall_at_'+suffix]=sum(q>=t for q in qs)/total_gt if total_gt else None
    for prefix,key in [('D1_Artifact/','artifact_region_bbox_score'),('D2_TextCorrectness/','text_region_bbox_score')]:
        top=bot=0.0
        for r in pos:
            gq=[q for q,f in zip(r['qualities'],r['gt_families']) if f.startswith(prefix)]
            if gq:top+=math.sqrt(len(gq))*sum(gq)/len(gq);bot+=math.sqrt(len(gq))
        result[key]=100*top/bot if bot else None
    nc=sum(r['family_denominator'] for r in samples)
    result['family_accuracy_on_found']=sum(r['family_correct'] for r in samples)/nc if nc else None
    return result


def quantile(values: List[float], p: float) -> float:
    """Linear-interpolation empirical quantile for nonempty observations."""
    xs=sorted(values);h=(len(xs)-1)*p;i=int(h);j=min(i+1,len(xs)-1)
    return xs[i]+(h-i)*(xs[j]-xs[i])


def bootstrap(samples: List[Dict[str,Any]], repetitions: int=2000, seed: int=20260804) -> Dict[str,Any]:
    """Stratified image bootstrap; RNG is Python random, not NumPy's stream."""
    pos=[r for r in samples if r['M']];clean=[r for r in samples if not r['M']]
    if repetitions<1 or not pos or not clean:return {}
    rng=random.Random(seed);values=[]
    # Recompute the complete headline, including positive sqrt(M) weights.
    for _ in range(repetitions):
        p=[pos[rng.randrange(len(pos))] for _ in pos]
        c=[clean[rng.randrange(len(clean))] for _ in clean]
        sp=sum(math.sqrt(r['M'])*r['F'] for r in p)/sum(math.sqrt(r['M']) for r in p)
        sn=sum(r['F'] for r in c)/len(c)
        values.append(100*(.7*math.sqrt(sp)+.3*math.sqrt(sn))**2)
    return dict(repetitions=repetitions,seed=seed,rng='random.Random',confidence=.95,
                lower=quantile(values,.025),upper=quantile(values,.975))


def read_jsonl(paths: Sequence[Path]) -> Dict[str,Dict[str,Any]]:
    """Read distinct sample IDs across one or more JSONL files with useful errors."""
    records={}
    for path in paths:
        with path.open(encoding='utf-8') as f:
            for i,text in enumerate(f,1):
                if not text.strip():continue
                try:
                    row=json.loads(text);sid=row['sample_id']
                    if sid in records:raise ValueError(f'duplicate sample_id {sid}')
                    records[sid]=row
                except (ValueError,KeyError,TypeError) as ex:
                    raise ValueError(f'{path}:{i}: {ex}') from ex
    return records


def main() -> None:
    """CLI for valid normalized-box predictions and absolute-box annotations."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',nargs='+',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--per-sample-out',type=Path)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20260804)
    args=p.parse_args()
    try:
        annotations=read_jsonl(args.annotations);predictions=read_jsonl([args.predictions])
        unknown=set(predictions)-set(annotations)
        if unknown:raise ValueError(f'{len(unknown)} prediction IDs have no reference annotation.')
        if args.bootstrap<0:raise ValueError('--bootstrap must be nonnegative.')
        samples=[score_sample(a,predictions.get(sid)) for sid,a in annotations.items()]
        result=aggregate(samples);result['headline_bootstrap']=bootstrap(samples,args.bootstrap,args.seed)
        result['implementation']='manuscript-specification implementation; synthetic-tested, not original author code'
        output=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:args.output.write_text(output,encoding='utf-8')
        else:print(output,end='')
        if args.per_sample_out:args.per_sample_out.write_text(json.dumps(samples,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except (OSError,ValueError,KeyError,TypeError) as ex:
        p.exit(2,f'error: {ex}\n')

if __name__=='__main__':main()
