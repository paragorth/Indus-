"""Loop-1 follow-up: take every FULL survivor of cycles 1-4 (deduplicated), and ask whether it is a rediscovery of known
structure by re-testing I(reading; fact | finer strata) with a within-strata permutation null:
  F1 = site x type2 x exact length (<=12)        -> kills length->size and sub-type effects
  F2 = F1 x emblem-present                        -> kills the script-only (bar-seal) genre vs animal-seal split
  F3 = F1 x material                              -> kills material-mediated effects
  E  = emblem facts re-tested on objects WITH an emblem only (animal vs animal), strata F1
A survivor that keeps p<0.01 under F1, F2 and F3 (and E if the fact is emblem) is 'beyond known structure'."""
import json,sys,glob,collections,random
import numpy as np
sys.argv=['x','--n','0','--seed','777']
src=open('tools/strat_dark.py').read(); src=src[:src.index('# ---------- run ----------')]
exec(src)
from scipy.stats import chi2
rng=random.Random(777)
full={}
for cyc in (1,2,3,4):
    d=json.load(open(f'data/derived/dark/loop1_cycle{cyc}.json'))['arrows']
    for a in d:
        if a.get('ok_perm') and a.get('ok_held') and a.get('ok_levels'):
            key=(a['kind'],json.dumps(a['p'],sort_keys=True),a['fact']); a['cycle']=cyc
            if key not in full: full[key]=a
print('distinct FULL survivors over 4 cycles:',len(full))
def strata_key(o,level,mode):
    k=o['f']['site']+'|'+o['f']['type2']+'|'+str(min(len(o[level]),12))
    if mode in ('F2',): k+='|'+str(o['f']['emblem'] is not None)
    if mode in ('F3',): k+='|'+str(o['f']['material'])
    return k
def test(kind,p,fact,objs,level,mode,nperm=1000,minn=40):
    xs=[];ys=[];st=[]
    for o in objs:
        y=o['f'][fact]
        if y is None: continue
        if mode=='E' and o['f']['emblem'] is None: continue
        x=read(kind,p,o[level],level)
        if x is None: continue
        xs.append(str(x)); ys.append(str(y)); st.append(strata_key(o,level,'F1' if mode=='E' else mode))
    if len(xs)<minn: return None
    cx=collections.Counter(xs); cy=collections.Counter(ys)
    xs=[x if cx[x]>=5 else 'other' for x in xs]; ys=[y if cy[y]>=5 else 'other' for y in ys]
    xi,yi,si,nx,ny,ns=encode(xs,ys,st)
    if nx<2 or ny<2: return None
    o,pp,k,ge=perm_p(xi,yi,si,nx,ny,ns,nperm,rng)
    return dict(mi=o,p=pp,n=len(xi),ns=ns)
rows=[]
for key,a in sorted(full.items(),key=lambda kv:kv[1]['p_screen']):
    kind,p,fact=a['kind'],a['p'],a['fact']
    r={m:test(kind,p,fact,TRAIN,'seq_raw',m) for m in ('F1','F2','F3')}
    if fact=='emblem': r['E']=test(kind,p,fact,TRAIN,'seq_raw','E')
    rh={m:test(kind,p,fact,TEST,'seq_raw',m,minn=60) for m in ('F1',)}
    ok=all(v and v['p']<0.01 for v in r.values()) and bool(rh['F1'] and rh['F1']['p']<0.05)
    why=[]
    if not (r['F1'] and r['F1']['p']<0.01): why.append('vanishes given site x type2 x exact length (length/sub-type rediscovery)')
    if r['F1'] and r['F1']['p']<0.01 and not (r['F2'] and r['F2']['p']<0.01): why.append('vanishes given emblem-present (script-only vs animal-seal genre, S157)')
    if r['F1'] and r['F1']['p']<0.01 and not (r['F3'] and r['F3']['p']<0.01): why.append('vanishes given material')
    if fact=='emblem' and not (r.get('E') and r['E']['p']<0.01): why.append('vanishes among emblem-bearing seals (it was emblem-present vs absent, S157)')
    if r['F1'] and r['F1']['p']<0.01 and not (rh['F1'] and rh['F1']['p']<0.05): why.append('held-out fails under F1')
    f=lambda v: 'n/a' if not v else 'p %.3g (n %d, strata %d)'%(v['p'],v['n'],v['ns'])
    line=f"{'BEYOND ' if ok else 'redisc '}c{a['cycle']} {kind} {p} -> {fact} | screen G-p {a['p_screen']:.1e} | F1 {f(r['F1'])} | F2 {f(r['F2'])} | F3 {f(r['F3'])}"+(f" | E {f(r.get('E'))}" if fact=='emblem' else '')+f" | held-out F1 {f(rh['F1'])} | {'; '.join(why) or 'survives all finer strata'}"
    print(line,flush=True); rows.append((ok,line))
out=['Loop-1 follow-up on %d distinct FULL survivors (cycles 1-4). Strata: F1 site x type2 x exact length; F2 = F1 x emblem-present; F3 = F1 x material; E = emblem facts among emblem-bearing seals only. Permutation null within strata (1000).'%len(full),
     'BEYOND known structure: %d; rediscoveries: %d'%(sum(o for o,_ in rows),sum(not o for o,_ in rows)),'']
out+=[l for o,l in rows if o]+['']+[l for o,l in rows if not o]
open('data/derived/dark/loop1_followup.txt','w').write('\n'.join(out)+'\n')
