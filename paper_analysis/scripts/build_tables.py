#!/usr/bin/env python3
"""Generate manuscript tables and transparent derivations from supplied aggregates.

No model inference or re-scoring of benchmark predictions is performed here.
"""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, TABLES, NOTES = ROOT/'data', ROOT/'tables', ROOT/'author_notes'
KEYS = ['configuration','score','artifact','text','positive_soft','clean_soft','image_fnr','clean_fpr','image_error','clean_boxes','recall_01','recall_05','mean_quality','zero_overlap_miss','count_mae','absolute_bias','background_fp','failed','ci_low','ci_high']
MODELS = [
('Qwen3.6-27b','Qwen3.6-27B','Reference model alias; spelling normalized only.'),
('Qwen3.8-27b','Qwen3.8-27B','Reference model alias; spelling normalized only.'),
('qwen3.7-plus','Qwen3.7-Plus','Reference model alias; spelling normalized only.'),
('gpt-5.6-sol','GPT-5.6-sol','Endpoint suffix retained; no inferred reasoning budget.'),
('gemini-3.5-flash','Gemini-3.5-Flash','Reference model alias; spelling normalized only.'),
('0722_ck145','Workflow-SFT','Generation-workflow trajectory supervision; parent of Decision-SFT.'),
('0820_ck740','Trajectory-SFT-R1','Four-source critic trajectory run: repeat-1, stratified validation run tag.'),
('0820_ck1001','Trajectory-SFT-R2','Four-source critic trajectory run: repeat-2 run tag.'),
('0826_ck136','Decision-SFT','Workflow-SFT continued on 4,000 balanced critic examples with decision and region targets.'),
('0828_ck444','Decision-SFT-Balanced+','Expanded balanced critic mixture; approximate source description, not an exact size claim.'),
('0828_ck688','Decision-SFT-Pool+','Expanded full critic pool; over 20,000 examples in the source description.'),
]
PIPES = [
('','Direct','Direct','Single model call; output is a variable-length region list.'),
('v4','Full-Seq','Inspect-Seq','Full local review with sequential execution and candidate verification.'),
('v5','Conservative','Inspect-Guard','Conservative inspection comparator; name does not imply an extra gate.'),
('v6','Full-Sched','Inspect-Sched','Full local review with parallel inventories, region/Skeptic micro-batches, and cached views.'),
('v6.3','Selective','Inspect-Select','Selective local review with dual global scouts and candidate verification.'),
]
SOURCES = [('gpt','GPT-image-2'),('external4','Public-4'),('sensenova_u1_5','SenseNova-U1.5'),('natural_clean_110','Natural clean')]
FLOW = [
('gpt','GPT-image-2',242,182,26,34,208),
('external4','Public-4',712,384,115,213,499),
('sensenova_u1_5','SenseNova-U1.5',350,152,70,128,222),
('natural_clean_110','Natural clean',125,102,8,15,110),
]
ORDER = [m[0] for m in MODELS]+['0722_ck145 + v4','0722_ck145 + v5','0722_ck145 + v6','0722_ck145 + v6.3','0820_ck1001 + v6.3','0826_ck136 + v6.3']

def esc(s):
    return s.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')

def write_table(name, rows):
    (TABLES/name).write_text('% Generated from supplied aggregates by scripts/build_tables.py.\n'+'\n'.join(rows)+'\n')

def line(cells):
    return ' & '.join(map(str,cells))+r' \\'

def label(r):
    return r['model']+(' / '+r['procedure'] if r['procedure']!='Direct' else '')

def read_data():
    a=json.loads((DATA/'annotation_0917_analysis.json').read_text())
    with (DATA/'summary_0917.csv').open(encoding='utf-8-sig',newline='') as f:
        raw=list(csv.reader(f))
    assert len(raw[0])==len(KEYS)
    by={}
    for cells in raw[1:]:
        assert len(cells)==len(KEYS)
        r=dict(zip(KEYS,[cells[0]]+list(map(float,cells[1:]))))
        base,_,suffix=r['configuration'].partition(' + ')
        r['model']=dict((x[0],x[1]) for x in MODELS)[base]
        r['procedure']=dict((x[0],x[1]) for x in PIPES)[suffix]
        assert r['configuration'] not in by
        by[r['configuration']]=r
    assert set(by)==set(ORDER) and len(by)==17
    return a,[by[x] for x in ORDER]

def main():
    for p in [DATA,TABLES,NOTES]: p.mkdir(exist_ok=True)
    a,rows=read_data(); o=a['overall']; ss=a['by_source']; by={r['configuration']:r for r in rows}
    for s in [o,*ss.values()]:
        assert s['images']==s['positive']+s['clean']
        assert len(s['per_image_counts'])==s['positive']
        assert sum(s['per_image_counts'])==s['gt_boxes']
        for fld,expected in [('families',s['gt_boxes']),('area_buckets',s['gt_boxes']),('gt_count_buckets',s['positive'])]:
            assert sum(s[fld].values())==expected
    for fld in ['images','positive','clean','gt_boxes']:
        assert o[fld]==sum(s[fld] for s in ss.values())
    assert (o['images'],o['positive'],o['clean'],o['gt_boxes'])==(1039,718,321,1983)
    assert set(ss)=={key for key,_ in SOURCES}
    assert tuple(ss['sensenova_u1_5'][key] for key in ['images','positive','clean','gt_boxes'])==(222,122,100,591)
    residuals=[]
    aggregate_checks=[]
    for r in rows:
        assert 0<=r['recall_05']<=r['recall_01']<=1
        assert 0<=r['zero_overlap_miss']+r['recall_01']<=1
        assert r['ci_low']<=r['score']<=r['ci_high']
        combine=lambda p,n:100*(.7*math.sqrt(max(0,p))+.3*math.sqrt(max(0,n)))**2
        p,n=r['positive_soft'],r['clean_soft']
        assert combine(p-.00005,n-.00005)-.0005<=r['score']<=combine(p+.00005,n+.00005)+.0005
        rebuilt=combine(p,n)
        residual=abs(rebuilt-r['score'])
        residuals.append(residual)
        lower=combine(p-.00005,n-.00005)
        upper=combine(p+.00005,n+.00005)
        aggregate_checks.append(dict(
            original_configuration=r['configuration'],paper_configuration=label(r),
            exported_headline=r['score'],rounded_positive_component=p,
            rounded_clean_component=n,reconstructed_headline=rebuilt,
            signed_residual=rebuilt-r['score'],absolute_residual=residual,
            component_rounding_score_lower=lower,
            component_rounding_score_upper=upper,
            headline_rounding_lower=r['score']-.0005,
            headline_rounding_upper=r['score']+.0005,
            rounding_intervals_overlap=(lower<=r['score']+.0005 and upper>=r['score']-.0005)))
        assert abs((718*r['image_fnr']+321*r['clean_fpr'])/1039-r['image_error'])<=.0001
    # These checks use rounded aggregates, not raw-prediction scorer runs.
    with (DATA/'aggregate_score_check.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate_checks[0]))
        w.writeheader();w.writerows(aggregate_checks)
    worst=max(aggregate_checks,key=lambda r:r['absolute_residual'])
    validation_scope=dict(
        release_validation=dict(source='README_SCORER.md, Section 15',
            status='source-reported; not rerun in this package',images=1039,
            maximum_absolute_point_estimate_difference=0,
            zero_derived_from='exact agreement of the checked outputs stated by the source',
            metrics=['headline','detection','counts','false positives'],
            configuration_ids_enumerated=False,configuration_count=None,
            covers_all_17_configurations_established=False,bootstrap_equality_claimed=False),
        aggregate_validation=dict(status='computed from rounded CSV values',
            configurations=17,maximum_absolute_residual=worst['absolute_residual'],
            maximum_residual_configuration=worst['paper_configuration'],
            all_rounding_intervals_overlap=all(r['rounding_intervals_overlap'] for r in aggregate_checks),
            raw_prediction_equivalence_test=False),
        input_csv_sha256=hashlib.sha256((DATA/'summary_0917.csv').read_bytes()).hexdigest())
    (DATA/'score_validation_scope.json').write_text(json.dumps(validation_scope,indent=2)+'\n')
    full,triage=by['0722_ck145 + v6'],by['0722_ck145 + v6.3']
    traj,traj_s=by['0820_ck1001'],by['0820_ck1001 + v6.3']
    dec,dec_s=by['0826_ck136'],by['0826_ck136 + v6.3']
    abstract_checks=dict(
        source='summary_0917.csv; annotation_0917_analysis.json; annotation_review_flow.csv',
        images=o['images'],defect_regions=o['gt_boxes'],clean_controls=o['clean'],
        candidates=1429,retention_pct=1039/1429*100,configurations=len(rows),
        highest_headline_configuration=label(max(rows,key=lambda r:r['score'])),
        highest_headline_score=max(r['score'] for r in rows),
        maximum_recall_at_quality_01_pct=100*max(r['recall_01'] for r in rows),
        full_selective_same_checkpoint=dict(
            full_configuration=label(full),selective_configuration=label(triage),
            score_difference=full['score']-triage['score'],
            clean_fpr_reduction_percentage_points=100*(triage['clean_fpr']-full['clean_fpr']),
            full_no_overlap_region_pct=100*full['zero_overlap_miss']),
        trajectory_selective_vs_direct=dict(checkpoint=traj['model'],
            clean_fpr_change_percentage_points=100*(traj_s['clean_fpr']-traj['clean_fpr'])),
        decision_selective_vs_direct=dict(checkpoint=dec['model'],
            clean_fpr_change_percentage_points=100*(dec_s['clean_fpr']-dec['clean_fpr'])),
        calculations_use_exported_rounded_values=True,new_experiment=False)
    (DATA/'abstract_numeric_checks.json').write_text(json.dumps(abstract_checks,indent=2)+'\n')
    # Mappings are complete, bijective and independent of results/rank.
    mapping=[]
    for r in rows:
        base,_,suffix=r['configuration'].partition(' + ')
        mt=next(x for x in MODELS if x[0]==base); pt=next(x for x in PIPES if x[0]==suffix)
        mapping.append(dict(original_configuration=r['configuration'],original_model=base,original_pipeline=suffix or 'no suffix',model_name=mt[1],pipeline_name=pt[1],previous_pipeline_name=pt[2],paper_configuration=label(r),model_description=mt[2],pipeline_description=pt[3]))
    with (NOTES/'MODEL_PIPELINE_MAPPING.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    # Identifier traceability remains in the author-only CSV, not in manuscript tables.
    # Source table is the sole image-source composition visualization.
    out=[]
    for k,n in SOURCES:
        s=ss[k]
        cells=[n,*[f'{s[v]:,}' for v in ['images','positive','clean','gt_boxes']]]
        cells += [f"{s['gt_boxes']/s['positive']:.3f}",s['max_boxes_per_image']] if s['positive'] else [r'\multicolumn{2}{c}{Clean-only}']
        out.append(line(cells))
    out += [r'\midrule',line(['Total','1,039',718,321,'1,983','2.762',26])]
    write_table('sources.tex',out)
    fl=[]
    for key,n,c,p,q,d,r in FLOW:
        assert p+q+d==c and p+q==r and r==ss[key]['images']
        fl.append(dict(source_key=key,source=n,candidates=c,first_pass=p,corrected_retained=q,discarded=d,retained=r,first_pass_rate=p/c,retention_rate=r/c))
    assert [sum(x[k] for x in fl) for k in ['candidates','first_pass','corrected_retained','discarded','retained']]==[1429,820,219,390,1039]
    with (DATA/'annotation_review_flow.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fl[0]));w.writeheader();w.writerows(fl)
    write_table('review_flow.tex',[line([x['source'],x['candidates'],x['first_pass'],x['corrected_retained'],x['discarded'],x['retained'],f"{100*x['retention_rate']:.2f}\\%"] ) for x in fl]+[r'\midrule',line(['Total','1,429',820,219,390,'1,039',r'72.71\%'])])
    write_table('review_rates.tex',[line([x['source'],f"{100*x['first_pass_rate']:.2f}\\%",f"{100*(x['candidates']-x['first_pass'])/x['candidates']:.2f}\\%",f"{100*x['corrected_retained']/x['candidates']:.2f}\\%",f"{100*x['discarded']/x['candidates']:.2f}\\%"] ) for x in fl])
    # Main table: raw values are not altered or rounded before formatting.
    out=[]
    for i,r in enumerate(rows):
        if i in [5,11]:out.append(r'\midrule')
        sc=f"{r['score']:.3f}"; sc=(r'\textbf{'+sc+'}') if r['score']==max(x['score'] for x in rows) else sc
        out.append(line([label(r),sc,f"{100*r['recall_01']:.2f}",f"{100*r['recall_05']:.2f}",f"{100*(1-r['image_fnr']):.2f}",f"{100*r['clean_fpr']:.2f}",int(r['failed'])]))
    write_table('main_results.tex',out)
    partitions=[]
    for r in rows:
        z=r['zero_overlap_miss']; low=1-z-r['recall_01']; mid=r['recall_01']-r['recall_05']; high=r['recall_05']
        assert min(z,low,mid,high)>=0 and abs(sum([z,low,mid,high])-1)<1e-9
        partitions.append(dict(configuration=r['configuration'],paper_configuration=label(r),no_overlap=z,weak_overlap=low,quality_01_to_05=mid,quality_ge_05=high))
    with (DATA/'region_outcome_partition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(partitions[0]));w.writeheader();w.writerows(partitions)
    select=['Qwen3.6-27b','gemini-3.5-flash','0722_ck145 + v4','0722_ck145 + v6','0826_ck136','0826_ck136 + v6.3']
    write_table('outcomes.tex',[line([p['paper_configuration'],*[f'{100*p[k]:.2f}' for k in ['no_overlap','weak_overlap','quality_01_to_05','quality_ge_05']]]) for name in select for p in partitions if p['configuration']==name])
    out=[]
    for name in ['0722_ck145','0820_ck1001','0826_ck136']:
        r,s=by[name],by[name+' + v6.3']
        out.append(line([r['model'],f"{s['score']-r['score']:+.3f}",f"{100*(s['recall_01']-r['recall_01']):+.2f}",f"{100*(s['recall_05']-r['recall_05']):+.2f}",f"{100*(s['clean_fpr']-r['clean_fpr']):+.2f}",f"{s['background_fp']-r['background_fp']:+.0f}"]))
    write_table('matched.tex',out)
    for fn,cols in [('components.tex',['ci','artifact','text','positive_soft','clean_soft']),('diagnostics.tex',['image_error','clean_boxes','zero_overlap_miss','mean_quality','count_mae','absolute_bias','background_fp'])]:
        out=[]
        for i,r in enumerate(rows):
            if i in [5,11]:out.append(r'\midrule')
            vals=[]
            for k in cols:
                if k=='ci': val=f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
                elif k in ['positive_soft','clean_soft','image_error','zero_overlap_miss']:val=f'{100*r[k]:.2f}'
                elif k=='background_fp':val=str(int(r[k]))
                elif k in ['artifact','text']:val=f'{r[k]:.3f}'
                else:val=f'{r[k]:.4f}'
                vals.append(val)
            out.append(line([label(r),*vals]))
        write_table(fn,out)
    # A finer density histogram supplements, rather than duplicates, the 5-bin figure.
    bins=range(1,27); hist={k:Counter(ss[k]['per_image_counts']) for k,n in SOURCES[:3]}
    exact=[]
    for n in bins:
        if any(hist[k][n] for k in hist):
            exact.append(line([n,*[hist[k][n] for k,name in SOURCES[:3]],sum(hist[k][n] for k in hist)]))
    write_table('density_exact.tex',exact)
    modelmatrix=[]
    for mid,mname,_ in MODELS:
        vals=[]
        for pid,pn,prev,desc in PIPES:
            rid=mid+(' + '+pid if pid else '')
            vals.append(r'$\checkmark$' if rid in by else '--')
        modelmatrix.append(line([mname,*vals]))
    write_table('configuration_matrix.tex',modelmatrix)
    report=dict(counts=dict(images=1039,flawed=718,clean=321,regions=1983),configuration_count=17,review_flow=dict(candidates=1429,first_pass=820,corrected_retained=219,discarded=390,retained=1039),multi_region_images=sum(n>=2 for n in o['per_image_counts']),best_score_configuration=max(rows,key=lambda r:r['score'])['configuration'],training_target_boxes=4368,validation_target_boxes=851,training_positive_mean=4368/2000,validation_positive_mean=851/400,checks=dict(source_totals=True,flow_conservation=True,one_to_one_mapping=True,outcome_partition=True,score_component_rounding=True),max_aggregate_rounding_residual=max(residuals),scope='Aggregate arithmetic only; original benchmark point estimates and intervals retained.',sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'summary_0917.csv',DATA/'annotation_0917_analysis.json',DATA/'README_SCORER.md',ROOT/'figures/fig_pipeline.png']})
    (DATA/'numerical_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (DATA/'results_renamed.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))



def build_revision_tables():
    """Format revised tables and check derived claims without altering scores."""
    from decimal import Decimal, ROUND_HALF_UP
    a, rows = read_data()
    by = {(r['model'], r['procedure']): r for r in rows}
    def fmt(value, decimals=1, percent=False, signed=False):
        number = Decimal(str(value)) * (100 if percent else 1)
        number = number.quantize(Decimal('1').scaleb(-decimals), rounding=ROUND_HALF_UP)
        return format(number, ('+' if signed else '') + f'.{decimals}f')
    def rate(r,k,decimals=2):
        return fmt(r[k],decimals,percent=True)
    def marked(r):
        return label(r) + (r'$^{'+str(int(r['failed']))+'}$' if r['failed'] else '')
    best = {k: max(r[k] for r in rows) for k in ['score','recall_01','recall_05']}
    best['image_fnr'] = min(r['image_fnr'] for r in rows)
    best['clean_fpr'] = min(r['clean_fpr'] for r in rows)
    out=[]
    for i,r in enumerate(rows):
        if i in [0,5,11]:
            if i:out.append(r'\midrule')
            heading={0:'Direct: base / commercial',5:'Direct: fine-tuned',11:'Multi-stage inspection'}[i]
            out.append(r'\multicolumn{6}{@{}l}{\emph{'+heading+r'}} \\')
        vals=[]
        for k in ['score','recall_01','recall_05','image_fnr','clean_fpr']:
            value=fmt(r[k],1) if k=='score' else fmt(1-r[k] if k=='image_fnr' else r[k],2,percent=True)
            if r[k]==best[k]:value=r'\textbf{'+value+'}'
            if k=='score':value+=' ['+fmt(r['ci_low'],1)+', '+fmt(r['ci_high'],1)+']'
            vals.append(value)
        out.append(line([marked(r),*vals]))
    write_table('main_results.tex',out)
    # All original fields remain in data; full headline precision is also in the appendix.
    out=[]
    for i,r in enumerate(rows):
        if i in [5,11]:out.append(r'\midrule')
        out.append(line([label(r),fmt(r['score'],3),f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",fmt(r['artifact'],3),fmt(r['text'],3),rate(r,'positive_soft'),rate(r,'clean_soft')]))
    write_table('components.tex',out)
    out=[];bg=[]
    for i,r in enumerate(rows):
        if i in [5,11]:out.append(r'\midrule')
        raw=Decimal(str(r['clean_boxes']))*321
        n=int(raw.to_integral_value(rounding=ROUND_HALF_UP))
        assert abs(raw-Decimal(n))<=Decimal('0.01605')
        flawed=int(r['background_fp'])-n
        assert flawed>=0
        bg.append({'configuration':r['configuration'],'clean_prediction_total':n,'background_fp_all':int(r['background_fp']),'background_fp_flawed':flawed})
        out.append(line([label(r),rate(r,'image_error'),fmt(r['clean_boxes'],4),rate(r,'zero_overlap_miss'),fmt(r['mean_quality'],4),fmt(r['count_mae'],4),fmt(r['absolute_bias'],4),int(r['background_fp']),flawed]))
    write_table('diagnostics.tex',out)
    with (DATA/'background_fp_partition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bg[0]));w.writeheader();w.writerows(bg)
    chosen=[('Qwen3.6-27B','Direct'),('Gemini-3.5-Flash','Direct'),('Workflow-SFT','Full-Seq'),('Workflow-SFT','Full-Sched'),('Decision-SFT','Direct'),('Decision-SFT','Selective')]
    out=[]
    for pair in chosen:
        r=by[pair];z=r['zero_overlap_miss'];low=1-z-r['recall_01'];mid=r['recall_01']-r['recall_05']
        out.append(line([label(r),*[fmt(v,1,percent=True) for v in [z,low,mid,r['recall_05']]]]))
    write_table('outcomes.tex',out)
    def pair(r):
        return rate(r,'recall_01',1)+' / '+rate(r,'clean_fpr',1)+(r'$^{'+str(int(r['failed']))+'}$' if r['failed'] else '')
    write_table('procedure_ladder.tex',[line([pair(by[('Workflow-SFT',proc)]) for proc in ['Direct','Conservative','Selective','Full-Sched','Full-Seq']])])
    out=[];changes={}
    for model in ['Workflow-SFT','Decision-SFT']:
        d,v=by[(model,'Direct')],by[(model,'Selective')]
        delta=' / '.join(fmt(v[k]-d[k],2,percent=True,signed=True) for k in ['recall_01','clean_fpr'])
        out.append(line([model,pair(d),pair(v),r'\multicolumn{2}{l}{'+delta+'}']))
    deltas=[]
    for proc in ['Direct','Selective']:
        w,d=by[('Workflow-SFT',proc)],by[('Decision-SFT',proc)]
        deltas.append(' / '.join(fmt(d[k]-w[k],2,percent=True,signed=True) for k in ['recall_01','clean_fpr']))
    out.append(r'\midrule')
    out.append(line([r'$\Delta$(Decision $-$ Workflow)',*deltas,r'\multicolumn{2}{l}{}']))
    write_table('two_by_two.tex',out)
    out=[]
    for key,name,n,first,corr,discard,retain in FLOW:
        out.append(line([name,n,*[fmt(x/n,2,percent=True)+r'\%' for x in [first,n-first,corr,discard]]]))
    out.append(r'\midrule');out.append(line(['Total',1429,*[fmt(x/1429,2,percent=True)+r'\%' for x in [820,609,219,390]]]))
    write_table('review_combined.tex',out)
    frontier=[r for r in rows if not any(x['clean_fpr']<=r['clean_fpr'] and x['recall_01']>=r['recall_01'] and (x['clean_fpr']<r['clean_fpr'] or x['recall_01']>r['recall_01']) for x in rows)]
    assert len(frontier)==6
    assert {r['configuration'] for r in frontier}=={'0828_ck688','0828_ck444','0826_ck136','0826_ck136 + v6.3','0722_ck145 + v6','0722_ck145 + v4'}
    high_sens=[r for r in rows if 1-r['image_fnr']>=.97]
    assert len(high_sens)==5
    assert sum(r['text']>r['artifact'] for r in rows)==15
    for model in ['Trajectory-SFT-R2','Workflow-SFT','Decision-SFT']:
        d,v=by[(model,'Direct')],by[(model,'Selective')]
        changes[model]={k:float(Decimal(str(v[k]))-Decimal(str(d[k]))) for k in ['score','clean_fpr','recall_01','recall_05']}
        assert changes[model]['score']>0
    thresholds={str(t):{'dice':t**(1/.7),'iou':t**(1/.7)/(2-t**(1/.7))} for t in [.1,.5]}
    check={
      'configurations':len(rows),'direct_configurations':sum(r['procedure']=='Direct' for r in rows),
      'pareto_configurations':[label(r) for r in frontier],
      'high_sensitivity_configurations':[label(r) for r in high_sens],
      'high_sensitivity_max_recall_01':max(r['recall_01'] for r in high_sens),
      'minimum_no_overlap':min(r['zero_overlap_miss'] for r in rows),
      'maximum_spatial_reach':1-min(r['zero_overlap_miss'] for r in rows),
      'maximum_recall_01':max(r['recall_01'] for r in rows),'maximum_recall_05':max(r['recall_05'] for r in rows),
      'top_score_configuration':label(max(rows,key=lambda r:r['score'])),
      'always_flag_error':321/1039,'qwen36_recall_rank':1+sum(r['recall_01']>rows[0]['recall_01'] for r in rows),
      'text_higher_than_artifact_count':sum(r['text']>r['artifact'] for r in rows),
      'direct_selective_changes':changes,'single_region_no_penalty_thresholds':thresholds,
      'top_three_score_intervals':[(label(r),r['ci_low'],r['ci_high']) for r in sorted(rows,key=lambda r:r['score'],reverse=True)[:3]],
      'note':'All derived from original configuration aggregates; no new model inference or raw-prediction evaluation.'}
    (DATA/'review_numeric_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    with (DATA/'coverage_restraint_points.csv').open('w',newline='') as f:
        fields=['model','procedure','clean_fpr','recall_01','score','failed','nondominated']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in fields[:-1]},'nondominated':r in frontier})
    print('Revision tables: 17 original configurations; 6 nondominated; 5 high-sensitivity; 15 text > artifact.')



def build_editorial_tables():
    """Apply the review's presentation changes and verify new numerical claims."""
    from decimal import Decimal, ROUND_HALF_UP
    a,rows=read_data()
    by={(x['model'],x['procedure']):x for x in rows}
    def fmt(v,n=1,pct=False):
        """Use the manuscript's decimal half-up rounding convention."""
        x=Decimal(str(v))*(100 if pct else 1)
        return format(x.quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_UP),f'.{n}f')
    best={k:max(x[k] for x in rows) for k in ['score','recall_01','recall_05']}
    best.update({k:min(x[k] for x in rows) for k in ['image_fnr','clean_fpr']})
    out=[]
    for i,x in enumerate(rows):
        if i in (0,5,11):
            if i:out.append(r'\midrule')
            name={0:'Direct: base / commercial',5:'Direct: fine-tuned',11:'Multi-stage inspection'}[i]
            out.append(r'\multicolumn{7}{@{}l}{\emph{'+name+r'}} \\')
        values=[]
        for k in ['score','recall_01','recall_05','image_fnr','clean_fpr']:
            value=fmt(1-x[k] if k=='image_fnr' else x[k],pct=(k!='score'))
            if x[k]==best[k]:value=r'\textbf{'+value+'}'
            if k=='score':value+=' ['+fmt(x['ci_low'])+', '+fmt(x['ci_high'])+']'
            values.append(value)
        out.append(line([label(x),*values,int(x['failed'])]))
    write_table('main_results.tex',out)
    out=[]
    for key in [('Workflow-SFT','Full-Seq'),('Workflow-SFT','Full-Sched'),('Decision-SFT','Direct'),('Decision-SFT','Selective')]:
        x=by[key];z=x['zero_overlap_miss'];parts=[z,1-z-x['recall_01'],x['recall_01']-x['recall_05'],x['recall_05']]
        out.append(line([x['model']+' / '+x['procedure'],*[fmt(p,pct=True) for p in parts]]))
    write_table('outcomes.tex',out)
    out=[]
    for key,name,n,first,corr,discard,retain in FLOW:
        count=lambda v:f'{v} ({fmt(v/n,pct=True)}'+r'\%)'
        out.append(line([name,n,*[count(v) for v in [first,corr,discard]],retain,fmt(retain/n,pct=True)+r'\%']))
    out.append(r'\midrule')
    out.append(line(['Total','1,429',*[f'{v} ({fmt(v/1429,pct=True)}'+r'\%)' for v in [820,219,390]],'1,039',fmt(1039/1429,pct=True)+r'\%']))
    write_table('review_combined.tex',out)
    # These tables are superseded by the main result table and the combined review table.
    for name in ['review_flow.tex','review_rates.tex','procedure_ladder.tex','two_by_two.tex','matched.tex']:
        (TABLES/name).unlink(missing_ok=True)
    rank=lambda row,k:1+sum(x[k]>row[k] for x in rows)
    trajectory=by[('Trajectory-SFT-R1','Direct')];decision=by[('Decision-SFT','Direct')]
    reversal={x['model']:{'recall_rank':rank(x,'recall_01'),'score_rank':rank(x,'score'),'sensitivity_pct':100*(1-x['image_fnr']),'clean_fpr_pct':100*x['clean_fpr']} for x in [trajectory,decision]}
    assert (reversal['Trajectory-SFT-R1']['recall_rank'],reversal['Decision-SFT']['recall_rank'])==(3,12)
    assert (reversal['Trajectory-SFT-R1']['score_rank'],reversal['Decision-SFT']['score_rank'])==(10,4)
    top=sorted(rows,key=lambda x:x['score'],reverse=True)[:3]
    assert max(x['ci_low'] for x in top)<=min(x['ci_high'] for x in top)
    full=by[('Workflow-SFT','Full-Sched')]
    assert max(decision['ci_low'],full['ci_low'])<=min(decision['ci_high'],full['ci_high'])
    changes=[]
    for m in ['Trajectory-SFT-R2','Workflow-SFT','Decision-SFT']:
        d,s=by[(m,'Direct')],by[(m,'Selective')]
        changes.append({'checkpoint':m,'clean_fpr_direct_pct':100*d['clean_fpr'],'clean_fpr_selective_pct':100*s['clean_fpr'],'delta_clean_fpr_pp':float((Decimal(str(s['clean_fpr']))-Decimal(str(d['clean_fpr'])))*100),'delta_score':float(Decimal(str(s['score']))-Decimal(str(d['score']))),'delta_recall_pp':float((Decimal(str(s['recall_01']))-Decimal(str(d['recall_01'])))*100)})
    assert sum(x['delta_clean_fpr_pp']<0 for x in changes)==1
    assert sum(x['delta_clean_fpr_pp']>0 for x in changes)==2
    assert sum(x['delta_score']>1.8 for x in changes)==1
    lineage=[by[(m,'Direct')] for m in ['Qwen3.6-27B','Workflow-SFT','Decision-SFT']]
    assert all(lineage[i]['recall_01']>lineage[i+1]['recall_01'] and lineage[i]['positive_soft']>lineage[i+1]['positive_soft'] and lineage[i]['clean_soft']<lineage[i+1]['clean_soft'] for i in (0,1))
    record={'rank_reversal':reversal,'direct_selective_changes':changes,
        'full_seq_failed_output_count':int(by[('Workflow-SFT','Full-Seq')]['failed']),
        'full_seq_clean_component_upper_bound_pp':22/321*100,
        'training_negative_domain_alignment':{'benchmark_natural_clean':110,'benchmark_clean':321,'natural_share_pct':110/321*100,'inference':'possible broad-domain alignment, not measured causation'},
        'score_calibration':{'closed_bounds':[0,100],'always_empty':9,'perfect':100,'counterexample_below_9':9/math.e},
        'top3_score_intervals_overlap':True,'decision_direct_full_sched_intervals_overlap':True,
        'top3_no_overlap_pct':{label(x):100*x['zero_overlap_miss'] for x in top},
        'lineage_components':[{k:x[k] for k in ['model','score','recall_01','positive_soft','clean_soft']} for x in lineage],
        'source_files':['data/summary_0917.csv','data/annotation_0917_analysis.json'],
        'raw_prediction_rescoring':False,'paired_significance_test':False}
    (DATA/'editorial_numeric_checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Editorial checks: ranks 3/12 -> 10/4; one FPR decrease, two increases; score floor is not 9.')

if __name__=='__main__':
    main()
    build_revision_tables()
    build_editorial_tables()
