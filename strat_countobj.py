"""S199: For the counted signs (S198: W740, 220, 700, 390, 900, 405), does the numeral before them depend on the object type
(seal vs tablet/other) or on the site? Control: permute object type (or site) labels among the numeral+sign tokens 5000x,
statistic = sum over signs of the chi-square-like mismatch (total variation distance)."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
NUM=set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56}
CT=[740,220,700,390,900,405]
tok=[]
seen=set()
for x in d:
    s=tuple(x['seq'])
    key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        if s[i] in CT and s[i-1] in NUM:
            tok.append((s[i],s[i-1],'SEAL' if x['type'].startswith('SEAL') else 'OTHER',x['site']))
def tvd(tok,lab):
    tot=0
    for w in CT:
        g=C.defaultdict(C.Counter)
        for (ww,n,*_),l in zip(tok,lab):
            if ww==w: g[l][n]+=1
        ks=[k for k in g if sum(g[k].values())>=5]
        for a in ks:
            for bb in ks:
                if a<bb:
                    A=sum(g[a].values());B=sum(g[bb].values())
                    tot+=0.5*sum(abs(g[a][n]/A-g[bb][n]/B) for n in set(g[a])|set(g[bb]))
    return tot
random.seed(2)
for name,idx in [('object',2),('site',3)]:
    lab=[t[idx] for t in tok]
    if name=='site': lab=[l if l in ('Mohenjo-daro','Harappa') else 'X' for l in lab]
    r=tvd(tok,lab); null=[]
    for _ in range(2000):
        random.shuffle(lab); null.append(tvd(tok,lab))
    print(name,'TVD %.2f null %.2f P=%.4f'%(r,sum(null)/len(null),sum(v>=r for v in null)/len(null)))
for w in CT:
    for o in ('SEAL','OTHER'):
        print(w,o,C.Counter(n for ww,n,oo,s in tok if ww==w and oo==o).most_common(6))
