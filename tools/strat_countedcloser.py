"""S351: which office word (closer) goes with the counted goods? For texts containing 'numeral + tree-family sign'
(W390/405/407) or 'numeral + W900', compare the final sign with all other texts of length >=3.
Control: permute the 'has counted good' label across texts (2000x) for each closer."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1,3,4,5,16,17,18,31,32,33,34}
GOODS={'tree':{390,405,407},'bracket900':{900}}
for key in ('seq_raw','seq_all'):
    T=list({(r['site'],tuple(r[key])) for r in C if r.get(key) and len(r[key])>=3})
    last=[t[-1] for _,t in T]
    for g,S in GOODS.items():
        lab=[any(t[i] in S and t[i-1] in NUM for i in range(1,len(t))) for _,t in T]
        n=sum(lab); obs=collections.Counter(l for l,x in zip(last,lab) if x)
        base=collections.Counter(last); N=len(T)
        random.seed(2); ge=collections.Counter(); le=collections.Counter()
        top=[c for c,_ in base.most_common(12)]
        for _ in range(2000):
            idx=set(random.sample(range(N),n)); c=collections.Counter(last[i] for i in idx)
            for x in top:
                if c[x]>=obs[x]: ge[x]+=1
                if c[x]<=obs[x]: le[x]+=1
        print(f'== {key} {g}: {n} texts with a counted {g}')
        for x in top:
            e=base[x]*n/N
            flag='  <-- over' if ge[x]/2000<0.01 else ('  <-- under' if le[x]/2000<0.01 else '')
            print(f'  ends W{x}: {obs[x]} (exp {e:.1f}) p_over={ge[x]/2000:.3f} p_under={le[x]/2000:.3f}{flag}')
        print('  top enders in these texts:',obs.most_common(6))
