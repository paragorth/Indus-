"""S-DARK-37.2b: do face TEXTS pair up repeatedly (fixed two-face stock pairs) beyond what independent face assignment gives?
Usage: python3 tools/dark_loop37_c2b.py <level> [nperm]"""
import sys,collections,random,itertools
sys.path.insert(0,'/home/user/Indus-/tools')
from dark_loop37 import *
LV=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'; NP=int(sys.argv[2]) if len(sys.argv)>2 else 1000
rnd=random.Random(37)
objs=load_faces(LV); M=multi(objs,1)   # NOT collapsed: the question is whether hand-made copies repeat the pairing
for name,S in [('Harappa TAB:I (incised, hand-made)',[o for o in M if o['site']=='Harappa' and o['type']=='TAB:I']),
               ('Harappa TAB:B (moulded)',[o for o in M if o['site']=='Harappa' and o['type']=='TAB:B']),
               ('Mohenjo-daro all types',[o for o in M if o['site']=='Mohenjo-daro']),
               ('all, count faces removed, incised only',[dict(o,faces=[f for f in o['faces'] if not is_count_face(f['seq'])]) for o in M if o['type']=='TAB:I' and sum(not is_count_face(f['seq']) for f in o['faces'])>=2])]:
    if len(S)<10: continue
    def key(fs): return tuple(sorted(tuple(f) for f in fs))
    real=[[f['seq'] for f in o['faces']] for o in S]
    def repeated(sets):
        c=collections.Counter(key(fs) for fs in sets); return sum(v for v in c.values() if v>=2), sum(1 for v in c.values() if v>=2)
    obs,groups=repeated(real)
    # null: faces reassigned among objects within (site,type), keeping face counts and face positions? -> positions free
    null=[]
    for _ in range(NP):
        pool=[s for fs in real for s in fs]; rnd.shuffle(pool); p=0; perm=[]
        for fs in real: perm.append(pool[p:p+len(fs)]); p+=len(fs)
        null.append(repeated(perm)[0])
    # how many objects have BOTH faces individually repeated elsewhere (so pairing could repeat by chance)?
    fc=collections.Counter(tuple(s) for fs in real for s in fs)
    both_rep=sum(1 for fs in real if all(fc[tuple(s)]>=2 for s in fs))
    print(f'-- {name}: {len(S)} objects; objects whose whole face-set recurs on another object: {obs} ({groups} distinct sets) vs {sum(null)/NP:.1f} if faces were assigned independently (P_hi={pval(obs,null):.3f}); objects with every face text repeated somewhere {both_rep}')
    c=collections.Counter(key(fs) for fs in real)
    for k,v in c.most_common(8):
        if v>=2: print('     x%d  %s'%(v,' | '.join('-'.join(map(str,f)) for f in k)))
