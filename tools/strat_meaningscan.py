"""S270: meaning scan. For every common sign, which non-linguistic facts does it predict?
Variables: object type, material, shape, animal motif, cult object, period, excavation area.
Design: test inside Harappa and inside Mohenjo-daro separately (site held fixed); keep only
associations significant (BH q<0.05) and in the same direction in BOTH cities (replication =
the control). Moulded duplicates (same site+type+text) counted once. Sequences: seq (merged),
checked again on seq_raw."""
import json,collections,sys
from scipy.stats import fisher_exact
import numpy as np
m=json.load(open('data/derived/merged-corpus-canonical.json'))
SEQ=sys.argv[1] if len(sys.argv)>1 else 'seq'
def per(r):
    t=str(r.get('time'));
    if t.startswith('Period 3A') or t=='Period 3B' or t.startswith('Period 3B-'): return 'early'
    if t.startswith('Period 3B/C') or t=='Period 3C-1': return 'middle'
    if t.startswith('Period 3C-2') or t.startswith('Period 3C-3') or t.startswith('Period 3C-4') or t.startswith('Period 4'): return 'late'
    return None
VARS={'type':lambda r:str(r['type']).split(':')[0]+(':'+str(r['type']).split(':')[1] if ':' in str(r['type']) else ''),
      'material':lambda r:r.get('material') if r.get('material') not in ('-','',None) else None,
      'shape':lambda r:r.get('shape') if r.get('shape') not in ('-','',None) else None,
      'motif':lambda r:(str(r.get('symbol')).split(':')[0] or 'none') if r.get('symbol')!='-' else None,
      'cult':lambda r:r.get('cult') if r.get('cult') not in ('-',None) else None,
      'period':per,
      'area':lambda r:r.get('area-section') if r.get('area-section') not in ('--','-',None) else None}
def texts(site):
    seen=set(); out=[]
    for r in m:
        if r['site']!=site or not r.get(SEQ): continue
        key=(r['type'],tuple(r[SEQ]))
        if str(r['type']).startswith('TAB') and key in seen: continue
        seen.add(key); out.append(r)
    return out
res={}
for site in ('Harappa','Mohenjo-daro'):
    T=texts(site); S=[set(r[SEQ]) for r in T]
    fs=collections.Counter(x for s in S for x in s)
    for var,f in VARS.items():
        vals=[f(r) for r in T]
        lv=collections.Counter(v for v in vals if v is not None)
        for level,nl in lv.items():
            if nl<15: continue
            for sg,ns in fs.items():
                if ns<15: continue
                a=sum(1 for s,v in zip(S,vals) if v==level and sg in s)
                b=sum(1 for s,v in zip(S,vals) if v is not None and v!=level and sg in s)
                c=nl-a; d=sum(1 for v in vals if v is not None and v!=level)-b
                if a+b<10: continue
                OR,p=fisher_exact([[a,b],[c,d]])
                res[(site,var,level,sg)]=(p,(a+.5)*(d+.5)/((b+.5)*(c+.5)),a,a+b,nl)
# BH per site
keep={}
for site in ('Harappa','Mohenjo-daro'):
    items=sorted([(v[0],k) for k,v in res.items() if k[0]==site])
    n=len(items)
    for i,(p,k) in enumerate(items):
        if p*n/(i+1)<0.05: keep[k]=res[k]
rep=[]
for (site,var,level,sg),v in keep.items():
    if site!='Harappa': continue
    k2=('Mohenjo-daro',var,level,sg)
    if k2 in keep and (v[1]>1)==(keep[k2][1]>1):
        rep.append((var,level,sg,v,keep[k2]))
print(f'[{SEQ}] tests: H {sum(1 for k in res if k[0]=="Harappa")}, MD {sum(1 for k in res if k[0]=="Mohenjo-daro")}; BH-significant: H {sum(1 for k in keep if k[0]=="Harappa")}, MD {sum(1 for k in keep if k[0]=="Mohenjo-daro")}; replicated in both: {len(rep)}')
for var,level,sg,h,md in sorted(rep,key=lambda x:(x[0],-abs(np.log(x[3][1])))):
    print(f'  {var:8s} {str(level):14s} W{sg:<4d} Harappa OR {h[1]:6.2f} ({h[2]}/{h[3]} texts with sign are {level}) | MD OR {md[1]:6.2f} ({md[2]}/{md[3]})')
