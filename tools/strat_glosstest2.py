"""S293: gloss tests, batch 2.
G4 'twins W91 = partners / a community': predicted to stand next to a number more often than other person
   signs (a count of partners or shares). Control: other person signs W90, W93, W99, W100, W121, W140, W142,
   W125, W151, W156, W176 pooled; permutation within site (5,000x).
G5 'closing words = offices/titles': predicted to cluster by building (block-house) within Mohenjo-daro more
   than middle (name) signs do. Statistic: share of same-house pairs among texts sharing a closer vs among texts
   sharing a middle sign of similar frequency. Control: house labels permuted among the texts."""
import json,random,collections
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUMS={1,2,3,4,5,16,17,18,31,32,33,34,55,56}
PERS={90,93,99,100,121,140,142,125,151,156,176}
rnd=random.Random(5)
import sys; LEVEL=sys.argv[1] if len(sys.argv)>1 else 'block-house'
for key in ['seq_raw','seq_strong','seq_all']:
    seen=set(); R=[]
    for r in C:
        s=r.get(key)
        if not s or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); R.append((r,s))
    lab=[];val=[];strat=[]
    for r,s in R:
        for i,a in enumerate(s):
            if a==91 or a in PERS:
                nb=(i>0 and s[i-1] in NUMS and s[i-1]!=2) or (i+1<len(s) and s[i+1] in NUMS and s[i+1]!=2)
                lab.append(a==91); val.append(nb); strat.append(r['site'])
    def f(L,V): return sum(v for l,v in zip(L,V) if l)/max(1,sum(L))
    obs=f(lab,val); groups=collections.defaultdict(list)
    for i,s in enumerate(strat): groups[s].append(i)
    ge=0
    for _ in range(5000):
        L=lab[:]
        for idx in groups.values():
            sub=[lab[i] for i in idx]; rnd.shuffle(sub)
            for i,v in zip(idx,sub): L[i]=v
        ge+=f(L,val)>=obs
    base=sum(v for l,v in zip(lab,val) if not l)/sum(1 for l in lab if not l)
    print(f'== {key}\nG4 twins next to a number: {obs:.2f} (n={sum(lab)}) vs other person signs {base:.2f}; P={(ge+1)/5001:.4f}')
    # G5
    M=[(r,s) for r,s in R if r['site']=='Mohenjo-daro' and r.get(LEVEL) not in (None,'','-','--') and len(s)>=3]
    house=[r[LEVEL] for r,s in M]
    CL={740,520,151,156,527,226,617,154,158,236}
    def pairs(h,which):
        tot=same=0
        by=collections.defaultdict(list)
        for i,(r,s) in enumerate(M):
            for a in set(which(s)): by[a].append(i)
        for a,idx in by.items():
            if a==740 or len(idx)<3 or len(idx)>60: continue
            for x in range(len(idx)):
                for y in range(x+1,len(idx)):
                    tot+=1; same+=h[idx[x]]==h[idx[y]]
        return same/max(1,tot)
    clo=lambda s:[s[-1]] if s[-1] in CL else []
    mid=lambda s:[a for a in s[1:-1] if a not in NUMS and a not in CL and a not in (817,861,820,60,400,90)]
    oc=pairs(house,clo); om=pairs(house,mid)
    nc=[];nm=[]
    for _ in range(500):
        h=house[:]; rnd.shuffle(h); nc.append(pairs(h,clo)); nm.append(pairs(h,mid))
    pc=(sum(x>=oc for x in nc)+1)/501; pm=(sum(x>=om for x in nm)+1)/501
    print(f'G5 Mohenjo-daro texts with a house {len(M)}: same-house pair rate, non-jar closers {oc:.3f} (null {sum(nc)/500:.3f}, P={pc:.3f}); middle signs {om:.3f} (null {sum(nm)/500:.3f}, P={pm:.3f})')
