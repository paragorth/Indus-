"""S-DARK-37.4: transcription robustness. (a) Wells faces (inscriptions.csv, id n.k) vs IM77 sides (im77_corpus_lines.csv
'side') on the objects both read (link: cisi via data/derived/dark/loop24_pairs.json, accepted pairs): side count, side
partition (bridge W->M, order-free face matching) and recorded side ORDER agreement. (b) The cycle 1-3 statistics re-run
inside IM77 alone (its own side recording, its own sign numbers; fixed pairs fitted on IM77 single-side texts; frame sets
translated W->M through the bridge): two closed sides per object, identical sides, abbreviation pairs, recorded-order
agreement of fixed pairs, orientability, vs the same nulls. (c) Pots and bangles with two texts, listed one by one.
Usage: python3 tools/dark_loop37_c4.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys,collections,random,itertools,math,json
from math import comb
sys.path.insert(0,'/home/user/Indus-/tools')
from dark_loop37 import *
LV=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'; NP=int(sys.argv[2]) if len(sys.argv)>2 else 300
rnd=random.Random(374)
BR={int(k):set(v) for k,v in json.load(open(BRIDGE)).items()}
PROP=ROOT+'/data/derived/dark/bridge_proposals.json'
try:
    PR=json.load(open(PROP)); PRM={}
    if isinstance(PR,dict): PR=PR.get('proposals',PR)
    for r in (PR if isinstance(PR,list) else []):
        w=r.get('wells') or r.get('W') or r.get('w'); m=r.get('mahadevan') or r.get('M') or r.get('m')
        if w is not None and m is not None: PRM.setdefault(int(w),set()).add(int(m))
    if isinstance(PR,dict):
        for k,v in PR.items():
            try: PRM.setdefault(int(k),set()).update(int(x) for x in (v if isinstance(v,list) else [v]))
            except Exception: pass
except Exception: PRM={}
BRX={w:BR.get(w,set())|PRM.get(w,set()) for w in set(BR)|set(PRM)}
objs=load_faces(LV); IM=load_im77()
P=[r for r in json.load(open(ROOT+'/data/derived/dark/loop24_pairs.json')) if r['accepted']]
cisi2im={r['cisi']:r['text_no'] for r in P}
cisi2obj={o['cisi']:o for o in objs.values()}
print(f'== S-DARK-37 cycle 4 level {LV}: Wells objects {len(objs)}, IM77 texts {len(IM)}, accepted cisi links {len(cisi2im)}; bridge {len(BR)} established + {len(PRM)} proposed Wells signs')
# ---------------- (a) side recording agreement on linked objects ----------------
def m_match(wseq,mseq,bridge):
    """share of Wells signs whose bridge M-set meets the IM77 side (order-free)"""
    if not wseq: return 0
    ms=set(mseq); return sum(1 for w in wseq if bridge.get(w,set())&ms)/len(wseq)
def partition_agree(wf,imf,bridge):
    """best one-to-one assignment of Wells faces to IM77 sides; returns (mean score, assignment)"""
    best=(-1,None)
    for perm in itertools.permutations(range(len(imf)),len(wf)):
        sc=sum(m_match(wf[i]['seq'],imf[p]['seq'],bridge) for i,p in enumerate(perm))/len(wf)
        if sc>best[0]: best=(sc,perm)
    return best
rows=[]
for cisi,tn in cisi2im.items():
    o=cisi2obj.get(cisi); im=IM.get(tn)
    if not o or not im: continue
    wf=[f for f in o['faces'] if f['seq']]; imf=im['faces']
    if len(wf)<2 and len(imf)<2: continue
    rows.append((o,im,wf,imf))
print(f'\n(a) linked objects with >= 2 inscribed faces in either transcription: {len(rows)}')
cnt=collections.Counter(); order_ag=order_n=0; part_ok=part_n=0; detail=collections.Counter()
for o,im,wf,imf in rows:
    key=('W' if len(wf)>=2 else 'w1','I' if len(imf)>=2 else 'i1'); cnt[key]+=1
    detail[(o['site'],o['ot'],len(wf),len(imf))]+=1
    if len(wf)>=2 and len(imf)>=2 and len(wf)==len(imf) and len(wf)<=4:
        sc,perm=partition_agree(wf,imf,BRX); part_n+=1
        if sc>=0.5:
            part_ok+=1; order_n+=1; order_ag+=(perm==tuple(range(len(wf))))
print(f'   both multi {cnt[("W","I")]}, Wells multi only {cnt[("W","i1")]}, IM77 multi only {cnt[("w1","I")]}')
print(f'   same side count and content partition recoverable through the bridge (every Wells face finds a distinct IM77 side, mean match >= 0.5): {part_ok}/{part_n}')
print(f'   recorded side ORDER agrees between the transcriptions (Wells face k = IM77 side k): {order_ag}/{order_n} = {order_ag/max(order_n,1):.3f}  (0.5 = no shared convention)')
print('   mismatched side counts by site x type (Wells faces, IM77 sides) :',
      sorted(((k,v) for k,v in detail.items() if k[2]!=k[3]),key=lambda x:-x[1])[:12])
# ---------------- (b) cycle 1-3 statistics inside IM77 ----------------
def tr(S):
    out=set()
    for w in S: out|=BRX.get(w,set())
    return out
CL_M=tr(CL); OPEN_M=tr(OPEN); SUF_M=tr(SUF); NUM_M=tr(NUM)
# M342 (jar) is the W740 closer; keep the translated sets but report them
print(f'\n(b) IM77 alone. frame sets translated: closers {sorted(CL_M)}, openers {sorted(OPEN_M)}, suffixes {sorted(SUF_M)}, numerals {sorted(NUM_M)}')
def closer_last_m(s):
    s=list(s)
    while len(s)>1 and s[-1] in SUF_M: s.pop()
    return bool(s) and s[-1] in CL_M
def count_face_m(s): return bool(s) and all(w in NUM_M or w in BRX.get(700,set()) for w in s) and any(w in NUM_M for w in s)
IMM=[o for o in IM.values() if len(o['faces'])>=2]
# collapse identical side-sets within site x type (S-DARK-13)
seen=set(); IMC=[]
for o in IMM:
    key=(o['site'],o['type'],tuple(sorted(tuple(f['seq']) for f in o['faces'])))
    if key in seen: continue
    seen.add(key); IMC.append(o)
print(f'   IM77 multi-sided texts {len(IMM)}, {len(IMC)} after collapsing identical side-sets; by type: {collections.Counter((o["site"],o["type"]) for o in IMC).most_common(8)}')
# fixed pairs on IM77 single-side texts
single=[o['faces'][0]['seq'] for o in IM.values() if len(o['faces'])==1 and len(o['faces'][0]['seq'])>=2]
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
pc=pair_counts(single); tested=[(k,v) for k,v in pc.items() if sum(v)>=5]; cand=[]
for (a,b),(nab,nba) in tested:
    n=nab+nba; cand.append((a,b,nab,nba,binom_one_sided(min(nab,nba),n)))
ps=sorted(c[4] for c in cand); m=len(tested); k=0
for i,p in enumerate(ps,1):
    if p<=0.05*i/m: k=i
thr=ps[k-1] if k else -1
FIX={(a,b):(nab>=nba) for a,b,nab,nba,p in cand if p<=thr}
print(f'   fixed pairs fitted on {len(single)} IM77 single-side texts: {len(FIX)} of {m} tested')
def fixed_dir(a,b):
    if a==b: return 0
    key=(a,b) if a<b else (b,a)
    if key not in FIX: return 0
    ab=FIX[key]; return (1 if ab else -1) if a<b else (-1 if ab else 1)
def cross(faces):
    out=[]
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            for a in set(faces[i]['seq']):
                for b in set(faces[j]['seq']):
                    d=fixed_dir(a,b)
                    if d: out.append(d)
    return out
def orientable(faces):
    if not cross(faces): return None
    if len(faces)>5: faces=faces[:5]
    for perm in itertools.permutations(faces):
        if all(d==1 for d in cross(list(perm))): return True
    return False
def subseq(a,b):
    it=iter(a); return all(x in it for x in b)
def abbrev(a,b): return (len(b)>=2 and len(a)>len(b) and subseq(a,b)) or (len(a)>=2 and len(b)>len(a) and subseq(b,a))
pool=collections.defaultdict(list)
for o in IM.values():
    for f in o['faces']: pool[(o['site'],o['type'],len(f['seq']))].append((o['oid'],f['seq']))
def draw(o,f):
    k=(o['site'],o['type'],len(f['seq'])); c=[s for oid,s in pool[k] if oid!=o['oid']]
    if len(c)<3: c=[s for L in (len(f['seq'])-1,len(f['seq'])+1) for oid,s in pool[(o['site'],o['type'],L)] if oid!=o['oid']] or c
    return rnd.choice(c) if c else f['seq']
def stats(S):
    r=collections.Counter()
    for o in S:
        fs=o['faces']
        r['two_closed']+=sum(closer_last_m(f['seq']) and not count_face_m(f['seq']) for f in fs)>=2
        for a,b in itertools.combinations(fs,2):
            r['pairs']+=1; r['ident']+=(a['seq']==b['seq']); r['abbrev']+=abbrev(a['seq'],b['seq']); r['shared']+=bool(set(a['seq'])&set(b['seq']))
        c=cross(fs); r['cross_n']+=len(c); r['cross_ag']+=sum(d==1 for d in c)
        x=orientable(fs)
        if x is not None: r['or_n']+=1; r['or']+=x
    return r
SUBS=[('all',IMC),('Harappa miniature tablets',[o for o in IMC if o['type']=='miniature tablet']),
      ('Mohenjo-daro all',[o for o in IMC if o['site']=='Mohenjodaro']),
      ('Mohenjo-daro >=3 sides',[o for o in IMC if o['site']=='Mohenjodaro' and len(o['faces'])>=3]),
      ('seals',[o for o in IMC if o['type']=='seal']),('sealings',[o for o in IMC if o['type']=='sealing']),
      ('copper tablets',[o for o in IMC if o['type']=='copper tablet']),
      ('other sites',[o for o in IMC if o['site'] not in ('Mohenjodaro','Harappa')])]
mean=lambda x: sum(x)/len(x) if x else float('nan')
for name,S in SUBS:
    if len(S)<3: print(f'\n-- {name}: {len(S)} too few'); continue
    ob=stats(S); null=collections.defaultdict(list)
    for _ in range(NP):
        fake=[dict(o,faces=[o['faces'][0]]+[dict(seq=draw(o,f)) for f in o['faces'][1:]]) for o in S]
        st=stats(fake)
        for k in ('two_closed','ident','abbrev','shared','cross_n'): null[k].append(st[k])
        null['or'].append(st['or']/st['or_n'] if st['or_n'] else 0); null['ag'].append(st['cross_ag']/st['cross_n'] if st['cross_n'] else .5)
    perm=[]
    for _ in range(NP):
        a=n=0
        for o in S:
            fs=o['faces'][:]; rnd.shuffle(fs); c=cross(fs); n+=len(c); a+=sum(d==1 for d in c)
        perm.append(a/n if n else .5)
    ag=ob['cross_ag']/ob['cross_n'] if ob['cross_n'] else float('nan'); orr=ob['or']/ob['or_n'] if ob['or_n'] else float('nan')
    print(f'\n-- {name}: {len(S)} objects, {ob["pairs"]} side pairs (null: later sides replaced by sides of other texts, same site x type x length, {NP}x)')
    print(f'   >= 2 closed sides: {ob["two_closed"]} vs {mean(null["two_closed"]):.1f} (P_lo={pval(ob["two_closed"],null["two_closed"],"lo"):.3f})')
    print(f'   identical sides {ob["ident"]} vs {mean(null["ident"]):.1f} (P_hi={pval(ob["ident"],null["ident"]):.3f}); abbreviation pairs {ob["abbrev"]} vs {mean(null["abbrev"]):.1f} (P_hi={pval(ob["abbrev"],null["abbrev"]):.3f}); any shared sign {ob["shared"]} vs {mean(null["shared"]):.1f}')
    print(f'   cross-side fixed pairs {ob["cross_n"]} vs {mean(null["cross_n"]):.1f} (P_lo={pval(ob["cross_n"],null["cross_n"],"lo"):.3f}); agreement with recorded side order {ob["cross_ag"]}/{ob["cross_n"]} = {ag:.3f} vs side-order-permuted {mean(perm):.3f} (P_hi={pval(ag,perm):.3f}), model C {mean(null["ag"]):.3f}')
    print(f'   orientable {ob["or"]}/{ob["or_n"]} = {orr:.3f} vs model C {mean(null["or"]):.3f} (P_hi={pval(orr,null["or"]):.3f})')
# Harappa one-closed + count complementarity in IM77
H=[o for o in IMC if o['type']=='miniature tablet' and len(o['faces'])==2]
one=[o for o in H if sum(closer_last_m(f['seq']) and not count_face_m(f['seq']) for f in o['faces'])==1]
oc=sum(1 for o in one if any(count_face_m(f['seq']) for f in o['faces']))
print(f'\n   IM77 Harappa two-sided miniature tablets {len(H)}: exactly one closed side {len(one)}, of which the other side is a count face {oc}')
# ---------------- (c) pots and bangles ----------------
print('\n(c) pots and bangles with two texts')
for o in objs.values():
    fs=[f for f in o['faces'] if f['seq']]
    if o['ot'] in ('pot','bangle') and len(fs)>=2:
        tn=cisi2im.get(o['cisi']); im=IM.get(tn) if tn else None
        print(f'   Wells {o["cisi"]} {o["site"]} {o["type"]} sides={o["sides"]}: '+' | '.join('-'.join(map(str,f['seq']))+('' if f['complete'] else '[')+' dir='+f['dir'] for f in fs)
              +(f'  -> IM77 {tn} {im["type"]} sides: '+' | '.join('-'.join(map(str,f['seq'])) for f in im['faces']) if im else '  -> not linked to IM77'))
im2cisi={v:k for k,v in cisi2im.items()}
for o in IM.values():
    if len(o['faces'])>=2 and ('pot' in o['type'] or 'bangle' in o['type']):
        c=im2cisi.get(o['oid']); w=cisi2obj.get(c) if c else None
        print(f'   IM77 {o["oid"]} {o["site"]} {o["type"]}: '+' | '.join(f'side{f["k"]} '+'-'.join(map(str,f['seq'])) for f in o['faces'])
              +(f'  -> Wells {c} {w["type"]} faces: '+' | '.join('-'.join(map(str,f['seq'])) for f in w['faces']) if w else f'  -> Wells link {c}'))
# bangles in either file
print('   bangle rows: Wells BNGL objects',sum(1 for o in objs.values() if o['ot']=='bangle'),'(with >= 2 inscribed faces:',sum(1 for o in objs.values() if o['ot']=='bangle' and len([f for f in o['faces'] if f['seq']])>=2),'); IM77 object types containing bangle:',collections.Counter(o['type'] for o in IM.values() if 'bangle' in o['type'].lower()))
