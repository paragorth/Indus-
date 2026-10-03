"""S-DARK-56 extra: closed suffix/prefix set test. For names >= 2 elements: share of final-position tokens covered by the
10 commonest finals (and the same for initials), and the entropy of the final/initial slot relative to the middle slot.
A Linear B / Akkadian-style name system has a small closed set of endings (or theophoric heads): high final coverage."""
import sys; sys.argv=['x','0']; sys.path.insert(0,'tools')
from dark_loop56 import *
import collections,math,random
def cov(names,r=None,n=1431):
    r=r or random.Random(1)
    names=[t for t in names if len(t)>=2]
    if len(names)>n: names=r.sample(names,n)
    I=collections.Counter(t[0] for t in names); F=collections.Counter(t[-1] for t in names); M=collections.Counter(a for t in names for a in t[1:-1])
    N=len(names)
    c10=lambda C:sum(v for _,v in C.most_common(10))/sum(C.values())
    hn=lambda C:H(C)/math.log2(len(C)) if len(C)>1 else 0
    return dict(n=N,init10=c10(I),fin10=c10(F),mid10=c10(M) if M else float('nan'),Hinit=hn(I),Hfin=hn(F),Hmid=hn(M) if M else float('nan'),kI=len(I),kF=len(F))
rows=[]
for LV in ['seq_raw','seq_all','im77']: rows.append((f'Indus {LV} middles',cov(indus_names(LV,2))))
rows.append(('Indus seq_raw NAME-only',cov(indus_names('seq_raw',2,field='name'))))
for nm in ['ur3_names_dedup','ob_names_dedup','linb_personnel_dedup','latin_names_dedup','proto_elamite_mid','hts','icd10']: rows.append((nm,cov(jl(nm))))
base=indus_names('seq_raw',2); r=random.Random(2); pool=[a for n in base for a in n]; L0=[len(n) for n in base]
rows.append(('freq-matched random',cov(sorted(set(tuple(r.choice(pool) for _ in range(r.choice(L0))) for _ in range(6000))))))
P('corpus                         n    top10 init  top10 final  top10 middle   Hnorm init/final/middle  k_init k_final')
for k,v in rows: P(f'{k:28s} {v["n"]:5d}   {v["init10"]:.3f}       {v["fin10"]:.3f}        {v["mid10"]:.3f}        {v["Hinit"]:.3f} / {v["Hfin"]:.3f} / {v["Hmid"]:.3f}     {v["kI"]:4d}  {v["kF"]:4d}')
open('data/derived/dark/loop56_suffix_log.txt','w').write('\n'.join(LOG)+'\n')
