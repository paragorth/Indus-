"""S307: replication of S306 in IM77 without a bridge for W407/W845. Blind: which final sign, and which final
bigram, is most enriched on copper tablets vs other Mohenjo-daro objects (Fisher, BH over candidates with >= 5
occurrences)? S306 predicts a copper-specific final sign preceded by a (M124/M125 = W61)-type sign."""
import csv,collections
from scipy.stats import fisher_exact
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows:
    if r['site']=='Mohenjodaro': sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    key=(ls[0]['object_type']=='copper tablet',tuple(s))
    if s and key not in seen: seen.add(key); T.append(key)
cu=[s for c,s in T if c]; ot=[s for c,s in T if not c]
print('copper',len(cu),'other MD',len(ot))
for name,f in (('final',lambda s:s[-1]),('final bigram',lambda s:tuple(s[-2:]) if len(s)>=2 else None)):
    a=collections.Counter(f(s) for s in cu); b=collections.Counter(f(s) for s in ot); res=[]
    for k in set(a)|set(b):
        if k is None or a[k]+b[k]<5: continue
        p=fisher_exact([[a[k],len(cu)-a[k]],[b[k],len(ot)-b[k]]],alternative='greater')[1]; res.append((p,k,a[k],b[k]))
    res.sort(); m=len(res)
    for i,(p,k,x,y) in enumerate(res[:6]): print(f'  {name} {k}: copper {x}/{len(cu)} vs other {y}/{len(ot)}; p={p:.1e} (BH {min(1,p*m/(i+1)):.1e})')
ex=[s for s in cu if s[-1]==res[0][1][-1]] if False else None
