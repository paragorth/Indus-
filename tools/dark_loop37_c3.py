"""S-DARK-37.3: arrow C (independent entries) tested with the S-DARK-19 pair-order statistics across the side boundary.
Fixed-order sign pairs (A before B in >= 95% of co-occurrences, BH 0.05) are fitted on SINGLE-FACE texts only, then
every (sign on face i, sign on face j) pair of a multi-sided object that is a fixed pair is checked:
  (i)   agreement of the recorded face order with the fixed order (A in recorded order => ~ within-text rate; C => 0.5),
  (ii)  orientability: is there ANY face order under which every cross-face fixed pair agrees (A in some order => high),
  (iii) density: how many fixed pairs straddle the boundary at all, against model C (face j replaced by a face of another
        object, same site x type x length) and model A (real single texts of the same length cut at the face boundary).
Usage: python3 tools/dark_loop37_c3.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys,collections,random,itertools,math
from math import comb
sys.path.insert(0,'/home/user/Indus-/tools')
from dark_loop37 import *
LV=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'; NP=int(sys.argv[2]) if len(sys.argv)>2 else 500
rnd=random.Random(373)
objs=load_faces(LV)
M,_=collapse(multi(objs,1))
multi_ids={o['oid'] for o in M}
# ---------- fit fixed pairs on single-face objects (no circularity) ----------
single=[o['faces'][0]['seq'] for o in objs.values() if len([f for f in o['faces'] if f['seq']])==1 and len(o['faces'][0]['seq'])>=2]
def pair_counts(texts):
    cnt=collections.Counter()
    for s in texts:
        c=collections.Counter(s); u=[a for a in s if c[a]==1]
        for i in range(len(u)):
            for j in range(i+1,len(u)):
                a,b=u[i],u[j]
                if a<b: cnt[(a,b,0)]+=1
                else: cnt[(b,a,1)]+=1
    out={}
    for (a,b,d),n in cnt.items(): out.setdefault((a,b),[0,0])[d]+=n
    return out
def binom_one_sided(k,n):
    if n<=80: return sum(comb(n,i) for i in range(k+1))/2**n
    z=(k+0.5-n/2)/math.sqrt(n/4); return 0.5*math.erfc(-z/math.sqrt(2))
pc=pair_counts(single)
tested=[(k,v) for k,v in pc.items() if sum(v)>=5]
cand=[]
for (a,b),(nab,nba) in tested:
    n=nab+nba; mn=min(nab,nba); p=binom_one_sided(mn,n)
    cand.append((a,b,nab,nba,p))
ps=sorted(c[4] for c in cand); m=len(tested); k=0
for i,p in enumerate(ps,1):
    if p<=0.05*i/m: k=i
thr=ps[k-1] if k else -1
FIX={}   # (a,b) -> True if a before b
for a,b,nab,nba,p in cand:
    if p<=thr: FIX[(a,b)]=(nab>=nba)
print(f'== S-DARK-37 cycle 3 level {LV} nperm {NP}: {len(M)} collapsed multi-sided objects; fixed pairs fitted on {len(single)} single-face texts: {len(FIX)} of {m} tested (BH thr {thr:.2e})')
def fixed_dir(a,b):
    """returns +1 if fixed a-before-b, -1 if fixed b-before-a, 0 if not a fixed pair"""
    if a==b: return 0
    key=(a,b) if a<b else (b,a)
    if key not in FIX: return 0
    ab=FIX[key]
    return (1 if ab else -1) if a<b else (-1 if ab else 1)
# within-text agreement rate on the fitting set (reference): fixed pairs agree by construction >= 95%; use the multi-sided faces themselves
def within_face(faces):
    ag=n=0
    for f in faces:
        s=f['seq']; c=collections.Counter(s); u=[x for x in s if c[x]==1]
        for i in range(len(u)):
            for j in range(i+1,len(u)):
                d=fixed_dir(u[i],u[j])
                if d: n+=1; ag+=(d==1)
    return ag,n
def cross(faces):
    """list of +1/-1 for every fixed pair straddling a face boundary, in recorded order (face i before face j)"""
    out=[]
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            for a in set(faces[i]['seq']):
                for b in set(faces[j]['seq']):
                    d=fixed_dir(a,b)
                    if d: out.append(d)
    return out
def orientable(faces):
    """exists a permutation of faces such that all cross fixed pairs agree (None if no cross pair)"""
    if not cross(faces): return None
    for perm in itertools.permutations(faces):
        if all(d==1 for d in cross(list(perm))): return True
    return False
pool=collections.defaultdict(list)
for o in objs.values():
    for f in o['faces']:
        if f['seq']: pool[(o['site'],o['type'],len(f['seq']))].append((o['oid'],f['seq']))
def draw(o,f):
    k=(o['site'],o['type'],len(f['seq'])); cand=[s for oid,s in pool[k] if oid!=o['oid']]
    if len(cand)<3:
        cand=[s for L in (len(f['seq'])-1,len(f['seq'])+1) for oid,s in pool[(o['site'],o['type'],L)] if oid!=o['oid']] or cand
    return rnd.choice(cand) if cand else f['seq']
bylen=collections.defaultdict(list)
for s in single: bylen[len(s)].append(s)
def modelA(o):
    """a real single text of the same total length, cut at the recorded face lengths"""
    L=sum(len(f['seq']) for f in o['faces']); cand=bylen.get(L) or bylen.get(L-1) or bylen.get(L+1)
    if not cand: return None
    s=rnd.choice(cand); out=[]; i=0
    for f in o['faces']:
        n=min(len(f['seq']),len(s)-i) if f is not o['faces'][-1] else len(s)-i
        out.append(dict(seq=s[i:i+n])); i+=n
    return [f for f in out if f['seq']]
def stats(S):
    ag=n=0; orn=orx=0; nobj_cross=0
    for o in S:
        c=cross(o['faces']); n+=len(c); ag+=sum(d==1 for d in c)
        if c: nobj_cross+=1
        r=orientable(o['faces'])
        if r is not None: orx+=1; orn+=r
    return dict(agree=ag,n=n,objs_with_cross=nobj_cross,orientable=orn,orient_tested=orx)
SUBS=[('all',M),
      ('no count faces',[dict(o,faces=[f for f in o['faces'] if not is_count_face(f['seq'])]) for o in M if sum(not is_count_face(f['seq']) for f in o['faces'])>=2]),
      ('Harappa tablets',[o for o in M if o['site']=='Harappa' and o['ot']=='tablet']),
      ('Mohenjo-daro all',[o for o in M if o['site']=='Mohenjo-daro']),
      ('Mohenjo-daro >=3 faces (prisms etc.)',[o for o in M if o['site']=='Mohenjo-daro' and len(o['faces'])>=3]),
      ('Mohenjo-daro two-faced',[o for o in M if o['site']=='Mohenjo-daro' and len(o['faces'])==2]),
      ('seals',[o for o in M if o['ot']=='seal']),('sealings',[o for o in M if o['ot']=='sealing']),
      ('non-MD-Harappa',[o for o in M if not o['big']]),
      ('complete faces >=2 signs',[dict(o,faces=[f for f in o['faces'] if f['complete'] and len(f['seq'])>=2]) for o in M if sum(f['complete'] and len(f['seq'])>=2 for f in o['faces'])>=2])]
for name,S in SUBS:
    if len(S)<3: print(f'\n-- {name}: {len(S)} objects, too few'); continue
    wa,wn=within_face([f for o in S for f in o['faces']])
    ob=stats(S)
    # null 1: face order permuted within object (agreement only)
    perm_ag=[]
    for _ in range(NP):
        a=n=0
        for o in S:
            fs=o['faces'][:]; rnd.shuffle(fs); c=cross(fs); n+=len(c); a+=sum(d==1 for d in c)
        perm_ag.append(a/n if n else 0.5)
    # null 2: model C (faces after the first replaced by faces of other objects, same site x type x length)
    C_n=[];C_or=[];C_ag=[];C_obj=[]
    for _ in range(NP):
        fake=[dict(o,faces=[o['faces'][0]]+[dict(seq=draw(o,f)) for f in o['faces'][1:]]) for o in S]
        st=stats(fake); C_n.append(st['n']); C_or.append(st['orientable']/st['orient_tested'] if st['orient_tested'] else 0); C_ag.append(st['agree']/st['n'] if st['n'] else 0.5); C_obj.append(st['objs_with_cross'])
    # reference A: real single texts cut at the boundary
    A_n=[];A_or=[];A_ag=[];A_obj=[]
    for _ in range(min(NP,200)):
        fake=[]
        for o in S:
            fs=modelA(o)
            if fs and len(fs)>=2: fake.append(dict(o,faces=fs))
        st=stats(fake); A_n.append(st['n']); A_or.append(st['orientable']/st['orient_tested'] if st['orient_tested'] else 0); A_ag.append(st['agree']/st['n'] if st['n'] else 0.5); A_obj.append(st['objs_with_cross'])
    mean=lambda x: sum(x)/len(x) if x else float('nan')
    rate=ob['agree']/ob['n'] if ob['n'] else float('nan'); orate=ob['orientable']/ob['orient_tested'] if ob['orient_tested'] else float('nan')
    print(f'\n-- {name}: {len(S)} objects; within-face fixed-pair agreement {wa}/{wn} = {wa/max(wn,1):.3f}')
    print(f'   cross-boundary fixed pairs: {ob["n"]} on {ob["objs_with_cross"]} objects; model C {mean(C_n):.1f} pairs on {mean(C_obj):.1f} objects (P_hi={pval(ob["n"],C_n):.3f} P_lo={pval(ob["n"],C_n,"lo"):.3f}); model A {mean(A_n):.1f} on {mean(A_obj):.1f}')
    print(f'   agreement with RECORDED face order: {ob["agree"]}/{ob["n"]} = {rate:.3f}; face-order permuted null {mean(perm_ag):.3f} (P_hi={pval(rate,perm_ag):.3f}); model C {mean(C_ag):.3f}; model A (recorded = reading order) {mean(A_ag):.3f}')
    print(f'   orientable objects (some face order satisfies every cross pair): {ob["orientable"]}/{ob["orient_tested"]} = {orate:.3f}; model C {mean(C_or):.3f} (P_hi={pval(orate,C_or):.3f} P_lo={pval(orate,C_or,"lo"):.3f}); model A {mean(A_or):.3f}')
# which fixed pairs straddle boundaries, and which way
cnt=collections.Counter()
for o in M:
    fs=o['faces']
    for i in range(len(fs)):
        for j in range(i+1,len(fs)):
            for a in set(fs[i]['seq']):
                for b in set(fs[j]['seq']):
                    d=fixed_dir(a,b)
                    if d: cnt[((a,b) if d==1 else (b,a),d==1)]+=1
print('\n-- commonest cross-boundary fixed pairs (fixed order a<b; True = recorded face order agrees):')
for (pr,ok),c in cnt.most_common(25): print(f'   {pr[0]}->{pr[1]} agree={ok} {c}')
