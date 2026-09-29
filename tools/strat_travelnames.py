"""S331: travelling names. Name units = adjacent pair (name-initial sign, name-final sign) from the S311 sets, and more
generally any adjacent pair inside the NAME slot of the S310 parse. Which name units occur at 3+ different sites?
Control: site labels permuted among texts (1,000x): expected number of pairs spread over >= 3 sites, given counts."""
import json,collections,random
P=json.load(open('data/derived/parsed_texts.json'))
occ=collections.defaultdict(list)
for i,o in enumerate(P):
    s=o['seq']; L=o['slots']
    for k in range(len(s)-1):
        if L[k]=='NAME' and L[k+1]=='NAME': occ[(s[k],s[k+1])].append(i)
occ={g:v for g,v in occ.items() if len(v)>=3}
sites=[o['site'] for o in P]
def wide(S): return sum(1 for v in occ.values() if len({S[i] for i in v})>=3)
obs=wide(sites); rnd=random.Random(36); null=[]
for _ in range(1000):
    p=sites[:]; rnd.shuffle(p); null.append(wide(p))
null.sort(); print(f'name pairs in >=3 texts: {len(occ)}; spread over >=3 sites: {obs}; permuted median {null[500]}, 5% {null[50]}; P(fewer)={sum(x<=obs for x in null)/1000:.3f}')
W=sorted(((len({sites[i] for i in v}),g,len(v)) for g,v in occ.items()),reverse=True)[:12]
for n,g,c in W: print('  ',g,'sites',n,'texts',c,sorted({sites[i] for i in occ[g]}))
