"""S261: do script-only seals use a different vocabulary than animal seals of the same late levels?
Per sign: share of script-only seals containing it vs animal seals. Control: permute the
script-only label within site+period bins, 2000x; report signs beyond the 99.5th percentile."""
import json,collections,random
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
S=[r for r in m if str(r.get('type','')).startswith('SEAL') and r['symbol']!='-' and r.get('seq')]
lab=[r['symbol']=='' for r in S]; bins=[(r['site'],r.get('time')) for r in S]
sets=[set(r['seq']) for r in S]
signs=[s for s,c in collections.Counter(x for st in sets for x in st).items() if c>=15]
def diff(lb):
    a=collections.Counter(); b=collections.Counter(); na=sum(lb); nb=len(lb)-na
    for st,l in zip(sets,lb):
        for x in st: (a if l else b)[x]+=1
    return {s:a[s]/na-b[s]/nb for s in signs}
obs=diff(lab)
byb=collections.defaultdict(list)
for i,bn in enumerate(bins): byb[bn].append(i)
random.seed(2); null=collections.defaultdict(list)
for _ in range(2000):
    lb=lab[:]
    for idx in byb.values():
        v=[lab[i] for i in idx]; random.shuffle(v)
        for i,x in zip(idx,v): lb[i]=x
    d=diff(lb)
    for s in signs: null[s].append(d[s])
out=[]
for s in signs:
    ns=sorted(null[s]); hi=ns[int(.995*len(ns))]; lo=ns[int(.005*len(ns))]
    if obs[s]>hi or obs[s]<lo: out.append((round(obs[s],3),s))
print('signs tested',len(signs),'script-only seals',sum(lab),'animal',len(lab)-sum(lab))
for d,s in sorted(out): print(f'W{s}: script-only minus animal share {d:+.3f}')
