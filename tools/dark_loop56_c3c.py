"""S-DARK-56 cycle 3 part (c), run separately: sealings vs seals sharing by site (middles >= 1 element; TAG rows)."""
import sys; sys.argv=['x','0']
sys.path.insert(0,'tools')
from dark_loop56 import *
import random,collections,json
R={}
for LV in ['seq_raw','seq_all']:
    objs=load_indus(LV)
    P(f'{LV}: object types after filters', collections.Counter(o['ot'] for o in objs).most_common())
    seals=[o for o in objs if o['ot']=='seal' and len(o['mid'])>=1]; tags=[o for o in objs if o['ot']=='sealing' and len(o['mid'])>=1]
    P(f'  sealings with middle: {len(tags)} by site', collections.Counter(o['site'] for o in tags).most_common())
    res={}
    for st in sorted(set(o['site'] for o in tags)):
        T=[o['mid'] for o in tags if o['site']==st]
        if len(T)<8: continue
        Sm=set(o['mid'] for o in seals if o['site']==st); So=set(o['mid'] for o in seals if o['site']!=st)
        same=sum(1 for m in T if m in Sm)/len(T); other=sum(1 for m in T if m in So)/len(T)
        Es=set(a for o in seals if o['site']==st for a in o['mid'])
        el_same=sum(1 for m in T for a in m if a in Es)/sum(len(m) for m in T)
        labs=[o['site'] for o in seals]; r=random.Random(9); nul=[]; nule=[]
        for _ in range(300):
            r.shuffle(labs); Sp=set(o['mid'] for o,l in zip(seals,labs) if l==st); Ep=set(a for o,l in zip(seals,labs) if l==st for a in o['mid'])
            nul.append(sum(1 for m in T if m in Sp)/len(T)); nule.append(sum(1 for m in T for a in m if a in Ep)/sum(len(m) for m in T))
        mu=sum(nul)/300; mue=sum(nule)/300
        P(f'  {LV} {st}: sealings {len(T)}, seals here {len(Sm)}: verbatim on same-site seal {same:.3f} (permuted sites {mu:.3f}; 97.5% {sorted(nul)[292]:.3f}), on other-site seal {other:.3f}; element coverage by same-site seals {el_same:.3f} (permuted {mue:.3f}; 97.5% {sorted(nule)[292]:.3f})')
        res[st]=dict(n=len(T),seals=len(Sm),same=same,null=mu,other=other,el_same=el_same,el_null=mue)
    R[LV]=res
json.dump(R,open('data/derived/dark/loop56_cycle3c.json','w'),indent=1)
open('data/derived/dark/loop56_cycle3c_log.txt','w').write('\n'.join(LOG)+'\n')
