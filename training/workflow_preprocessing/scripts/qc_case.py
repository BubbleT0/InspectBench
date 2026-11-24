import glob, json, os, base64, io, re, urllib.request, concurrent.futures, sys
from PIL import Image
MROOT='/path/to/storage/research/backflow/v0.3-0706'
GW='https://api.example.invalid/v1/chat/completions'
KEY=open(MROOT+'/.gw_key').read().strip(); MODEL='gpt-5.4-pro'

def build_cases():
    cases={}
    for f in glob.glob(MROOT+'/openai/*/*.jsonl'):
        if f.endswith('unknown.jsonl'): continue
        for l in open(f):
            if not l.strip(): continue
            m=json.loads(l).get('metadata',{}); c=m.get('case')
            if not c or c in cases: continue
            st=m.get('source_transcript','')
            src=st.split('/.harness/')[0] if '/.harness/' in st else ''
            cases[c]={'src':src}
    return cases

def case_info(c,src):
    H=src+'/.harness/workflow-runs/'+c
    brief=''; img=None
    ij=H+'/input.json'
    if os.path.exists(ij):
        try: brief=(json.load(open(ij,errors='replace')).get('brief') or '')[:400]
        except: pass
    rj=H+'/result.json'
    if os.path.exists(rj):
        try:
            im=(json.load(open(rj,errors='replace')).get('result',{}) or {}).get('image')
            if isinstance(im,str) and im:
                p=os.path.join(src,im)
                if os.path.exists(p): img=p
        except: pass
    return brief,img

def b64(path):
    im=Image.open(path).convert('RGB'); w,h=im.size; mm=max(w,h)
    if mm>1600: im=im.resize((int(w*1600/mm),int(h*1600/mm)))
    buf=io.BytesIO(); im.save(buf,'JPEG',quality=85)
    return base64.b64encode(buf.getvalue()).decode()

TAIL=('\n\n评审这张成品图,只输出JSON:{"tier":1或2或3,"score":1-10,"reason":"30字内"}\n'
  '一档=完整完成/文字清晰可读/排版良好/无明显缺陷;二档=基本完成但轻微问题;'
  '三档=严重问题(关键文字重叠/不可读/缺失/核心未完成/排版混乱/图质极差/严重不符)。')

def vlm(img,brief):
    try: b=b64(img)
    except Exception as e: return None,None,'IMG:'+str(e)[:40]
    body={'model':MODEL,'max_tokens':900,'messages':[{'role':'user','content':[
      {'type':'text','text':'以下是AI设计Agent的任务:\n'+brief+TAIL},
      {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+b}}]}]}
    for a in range(3):
        try:
            req=urllib.request.Request(GW,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            r=json.load(urllib.request.urlopen(req,timeout=150))
            t=r['choices'][0]['message']['content']; mm=re.search(r'\{.*\}',t,re.S)
            d=json.loads(mm.group(0)) if mm else {}
            return d.get('tier'),d.get('score'),str(d.get('reason',''))[:50]
        except Exception as e:
            if a==2: return None,None,'ERR:'+str(e)[:40]

def qc(item):
    c,src=item
    brief,img=case_info(c,src)
    if not img: return {'case':c,'tier':0,'reason':'无终图(跳过)'}
    t,s,r=vlm(img,brief)
    if t not in (1,2,3): return {'case':c,'tier':None,'score':s,'reason':r}
    return {'case':c,'tier':t,'score':s,'reason':r}

if __name__=='__main__':
    cases=build_cases(); items=[(c,d['src']) for c,d in cases.items() if d['src']]
    if len(sys.argv)>1 and sys.argv[1].isdigit(): items=items[:int(sys.argv[1])]
    out=open(MROOT+'/qc_case_result.jsonl','w'); dist={}; done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(qc,items):
            out.write(json.dumps(r,ensure_ascii=False)+'\n'); out.flush()
            done+=1; dist[r['tier']]=dist.get(r['tier'],0)+1
            if done%50==0: print(done,dist,flush=True)
    print('DONE',done,dist)
