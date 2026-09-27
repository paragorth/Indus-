"""S223: Test of Bhaskar 2025 (SSRN 5460614, 'A map and key to spell ancient Harappan'; PDF supplied by the user).
Claim: a 'T sign' spells a pictorial animal X (Charts 1-2, Table 2). Glyphs matched to Wells numbers by rendering (docs/mapkey-sign-match.png):
gavial W235 (M65), elephant W820 (M391), bison-1 W850, joined tiger W455, buffalo W515, zebu W930, markhor W415.
Prediction if true: the sign is enriched on seals showing its animal. Control: permute animal labels among seals within site 5000x; one-sided."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
PAIRS=[('Gavi',235),('Elep',820),('Gaur',850),('Tigr',455),('Buff',515),('Zebu',930),('Goat',415)]
seals=[x for x in d if x['type'].startswith('SEAL') and x['symbol'] not in ('','-','Unknown')]
def an(s): return 'Tigr' if s in ('Tigr','Htgr') else 'Goat' if s.startswith('Goat') else s.split(':')[0]
rows=[(x['site'],an(x['symbol']),set(x['seq'])) for x in seals]
def stats(labs):
    out={}
    for a,w in PAIRS:
        on=[w in s for (st,_,s),l in zip(rows,labs) if l==a]; off=[w in s for (st,_,s),l in zip(rows,labs) if l!=a]
        out[(a,w)]=(sum(on),len(on),sum(off),len(off))
    return out
real=stats([r[1] for r in rows])
by=C.defaultdict(list)
for i,r in enumerate(rows): by[r[0]].append(i)
random.seed(73); N=5000; ge=C.Counter()
for _ in range(N):
    labs=[r[1] for r in rows]
    for s,ix in by.items():
        p=[labs[i] for i in ix]; random.shuffle(p)
        for i,l in zip(ix,p): labs[i]=l
    st=stats(labs)
    for k,v in st.items():
        if v[0]>=real[k][0]: ge[k]+=1
for k,(a,n,b,m) in real.items():
    print('%-5s W%-4d on its animal %d/%d (%.3f)  other seals %d/%d (%.3f)  P(one-sided)=%.4f'%(k[0],k[1],a,n,a/max(n,1),b,m,b/m,ge[k]/N))
