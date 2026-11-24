import json, glob, sys, collections

def jtype(v):
    if isinstance(v,bool): return 'boolean'
    if isinstance(v,int): return 'integer'
    if isinstance(v,float): return 'number'
    if isinstance(v,list): return 'array'
    if isinstance(v,dict): return 'object'
    return 'string'

def tooldef_map(rec):
    m={}
    for t in rec.get('tools',[]):
        fn = t.get('function', t)
        if isinstance(fn,dict) and fn.get('name'): m[fn['name']]=fn
    return m

added=collections.Counter(); patched_records=0
files=glob.glob(sys.argv[1])
for fp in files:
    out=[]; changed=False
    for line in open(fp):
        if not line.strip(): continue
        rec=json.loads(line); tdm=tooldef_map(rec); rc=False
        for msg in rec.get('messages',[]):
            for tc in (msg.get('tool_calls') or []):
                name=tc['function']['name']; args=tc['function']['arguments']
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except: continue
                if not isinstance(args,dict): continue
                td=tdm.get(name)
                if not td: continue
                params=td.setdefault('parameters',{}); props=params.setdefault('properties',{})
                for k,v in args.items():
                    if k not in props:
                        props[k]={'type':jtype(v)}
                        added[f'{name}.{k}']+=1; rc=True
        if rc: changed=True; patched_records+=1
        out.append(json.dumps(rec,ensure_ascii=False))
    if changed:
        open(fp,'w').write('\n'.join(out)+'\n')
print('patched records:',patched_records)
for k,v in added.most_common(15): print(f'  +{v:5d}  {k}')
