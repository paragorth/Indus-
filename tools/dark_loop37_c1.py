"""S-DARK-37.1: inventory of multi-sided objects + arrow A (one text across sides).
Usage: python3 tools/dark_loop37_c1.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys,collections,random
sys.path.insert(0,'/home/user/Indus-/tools')
from dark_loop37 import *
LV=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'; NP=int(sys.argv[2]) if len(sys.argv)>2 else 1000
rnd=random.Random(37)
objs=load_faces(LV)
allseqs=[f['seq'] for o in objs.values() for f in o['faces'] if f['seq']]
parse=make_parser(learn_qual(allseqs))
print(f'== S-DARK-37 cycle 1 level {LV} nperm {NP}: objects {len(objs)}, faces with signs {len(allseqs)}')
M=multi(objs,1)
Mc,dropped=collapse(M)
print(f'multi-sided objects (>=2 faces with >=1 sign): {len(M)}; after collapsing identical moulded/impressed face-sets: {len(Mc)} (dropped {dropped})')
tab=collections.Counter((o['site'],o['type'],len(o['faces'])) for o in Mc)
print('site x type x inscribed faces (collapsed):')
for k,v in sorted(tab.items(),key=lambda kv:-kv[1])[:25]: print('  ',k,v)
print('  sides column vs inscribed faces:',sorted(collections.Counter((o['sides'],len(o['faces'])) for o in Mc).items()))
# subsets
def subset(name,objs_):
    return name,objs_
SUBS=[('all',Mc),('all-uncollapsed',M),
      ('MD prisms (SEAL/TAB, >=3 faces, Mohenjo-daro)',[o for o in Mc if o['site']=='Mohenjo-daro' and len(o['faces'])>=3]),
      ('MD two-faced',[o for o in Mc if o['site']=='Mohenjo-daro' and len(o['faces'])==2]),
      ('Harappa tablets',[o for o in Mc if o['site']=='Harappa' and o['ot']=='tablet']),
      ('Harappa tablets no count face',[o for o in Mc if o['site']=='Harappa' and o['ot']=='tablet' and not any(is_count_face(f['seq']) for f in o['faces'])]),
      ('sealings (TAG*)',[o for o in Mc if o['ot']=='sealing']),
      ('seals',[o for o in Mc if o['ot']=='seal']),
      ('non-MD-Harappa',[o for o in Mc if not o['big']]),
      ('complete faces only',[dict(o,faces=[f for f in o['faces'] if f['complete']]) for o in Mc if sum(f['complete'] for f in o['faces'])>=2])]
# face-level rates for the independence (C) expectation, by type x site among ALL faces (single- and multi-sided)
rate=collections.defaultdict(lambda:[0,0,0])
for o in objs.values():
    for f in o['faces']:
        if f['seq']:
            r=rate[(o['site'],o['type'])]; r[0]+=1; r[1]+=is_opener_first(f['seq']); r[2]+=closer_last(f['seq'])
def binom_expect(objs_,fn):
    """expected number of objects with >=2 faces satisfying fn if faces independent with type x site rate"""
    e=0.0
    for o in objs_:
        r=rate[(o['site'],o['type'])]; p=r[fn]/r[0] if r[0] else 0; n=len(o['faces'])
        # P(>=2 of n)
        from math import comb
        e+=sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(2,n+1))
    return e
for name,S in SUBS:
    if len(S)<5: print(f'\n-- {name}: n={len(S)} (too few)'); continue
    print(f'\n-- {name}: n={len(S)} objects, {sum(len(o["faces"]) for o in S)} faces')
    # A1: opener on first recorded face; closer-final on last recorded face
    op_objs=[o for o in S if sum(is_opener_first(f['seq']) for f in o['faces'])==1]
    cl_objs=[o for o in S if sum(closer_last(f['seq']) for f in o['faces'])==1]
    def stat(objs_,fn,which):
        c=0
        for o in objs_:
            idx=[i for i,f in enumerate(o['faces']) if fn(f['seq'])][0]
            c+= (idx==0) if which=='first' else (idx==len(o['faces'])-1)
        return c
    for lab,objs_,fn,which in [('opener-face is recorded face 1',op_objs,is_opener_first,'first'),
                               ('opener-face is recorded LAST face',op_objs,is_opener_first,'last'),
                               ('closer-face is recorded LAST face',cl_objs,closer_last,'last'),
                               ('closer-face is recorded face 1',cl_objs,closer_last,'first')]:
        if len(objs_)<3: print(f'   {lab}: n={len(objs_)} too few'); continue
        obs=stat(objs_,fn,which)
        null=[]
        for _ in range(NP):
            c=0
            for o in objs_:
                n=len(o['faces']); pos=rnd.randrange(n)
                c+= (pos==0) if which=='first' else (pos==n-1)
            null.append(c)
        exp=sum(null)/NP
        print(f'   {lab}: {obs}/{len(objs_)} = {obs/len(objs_):.2f}; null (face labels permuted) {exp/len(objs_):.2f}, P_hi={pval(obs,null):.3f} P_lo={pval(obs,null,"lo"):.3f}')
    # A2: how many faces carry a closer / an opener?  one text => at most one of each
    ncl=collections.Counter(sum(closer_last(f['seq']) for f in o['faces']) for o in S)
    nop=collections.Counter(sum(is_opener_first(f['seq']) for f in o['faces']) for o in S)
    e_cl=binom_expect(S,2); e_op=binom_expect(S,1)
    print(f'   faces closer-final per object: {dict(sorted(ncl.items()))}; objects with >=2 closed faces {sum(v for k,v in ncl.items() if k>=2)} vs {e_cl:.1f} expected if faces independent (type x site face rates)')
    print(f'   faces opener-initial per object: {dict(sorted(nop.items()))}; objects with >=2 opener faces {sum(v for k,v in nop.items() if k>=2)} vs {e_op:.1f} expected if independent')
    # A3: frame continuity across the side boundary: concatenate faces in recorded order, parse, count violations
    def violations(seq):
        """openers not initial; closer followed by a non-suffix non-closer sign (internal closer)"""
        v=0
        for i,w in enumerate(seq):
            if i>0 and w in OPEN: v+=1
            if w in CL and i<len(seq)-1 and seq[i+1] not in SUF and seq[i+1] not in CL: v+=1
        return v
    cat=[sum((f['seq'] for f in o['faces']),[]) for o in S]
    obs_v=sum(violations(s)>0 for s in cat)
    # null C: faces drawn from different objects of the same type x site (random concatenation), same face count
    pool=collections.defaultdict(list)
    for o in objs.values():
        for f in o['faces']:
            if f['seq']: pool[(o['site'],o['type'])].append(f['seq'])
    nullC=[]
    for _ in range(min(NP,300)):
        c=0
        for o in S:
            p=pool[(o['site'],o['type'])]
            s=sum((rnd.choice(p) for _ in o['faces']),[])
            c+=violations(s)>0
        nullC.append(c)
    # reference A: single-face texts of the same length distribution
    single=collections.defaultdict(list)
    for o in objs.values():
        fs=[f for f in o['faces'] if f['seq']]
        if len(fs)==1 and fs[0]['complete']: single[len(fs[0]['seq'])].append(fs[0]['seq'])
    refA=[]
    for _ in range(min(NP,300)):
        c=0;n=0
        for s in cat:
            L=len(s); cand=single.get(L) or single.get(L-1) or single.get(L+1)
            if cand: c+=violations(rnd.choice(cand))>0; n+=1
        refA.append(c/max(n,1))
    print(f'   concatenated-in-recorded-order texts with a frame violation (internal closer or non-initial opener): {obs_v}/{len(S)} = {obs_v/len(S):.2f}; '
          f'random same-type faces concatenated (C) {sum(nullC)/len(nullC)/len(S):.2f} (P_lo={pval(obs_v,nullC,"lo"):.3f}); single-face texts of the same length (A) {sum(refA)/len(refA):.2f}')
    # A4: slot-rank continuity at each boundary (last slot of face i <= first slot of face i+1), observed vs face-order permuted
    def bound_ok(o,order):
        ok=tot=0
        fs=[o['faces'][i] for i in order]
        for a,b in zip(fs,fs[1:]):
            la=parse(a['seq']); lb=parse(b['seq'])
            if not la or not lb: continue
            tot+=1; ok+= SLOTRANK[la[-1]]<=SLOTRANK[lb[0]]
        return ok,tot
    obs_ok=obs_tot=0
    for o in S:
        a,b=bound_ok(o,range(len(o['faces']))); obs_ok+=a; obs_tot+=b
    null=[]
    for _ in range(NP):
        c=0
        for o in S:
            order=list(range(len(o['faces']))); rnd.shuffle(order); c+=bound_ok(o,order)[0]
        null.append(c)
    print(f'   boundaries where slot rank continues (end slot of face i <= start slot of face i+1): {obs_ok}/{obs_tot} = {obs_ok/max(obs_tot,1):.2f}; face order permuted {sum(null)/NP/max(obs_tot,1):.2f} (P_hi={pval(obs_ok,null):.3f})')
# examples: MD prisms listing
print('\n-- Mohenjo-daro objects with >=3 inscribed faces (recorded order; * = complete):')
for o in [o for o in Mc if o['site']=='Mohenjo-daro' and len(o['faces'])>=3]:
    print('  ',o['cisi'],o['type'],' | '.join(('*' if f['complete'] else '')+'-'.join(map(str,f['seq'])) for f in o['faces']))
