"""S309 (user point: we hold a tiny fraction of all seals ever made). What does that mean for our numbers?
(a) Good-Turing: chance that the next sign token / next sign-pair / next whole text is one we have never seen
(= share of singletons), and a Chao1 estimate of the total sign inventory. (b) Site bias: how much of each key
result rests on Mohenjo-daro + Harappa (share of texts). seq_raw, site+text dedup."""
import json,collections
C=json.load(open('data/derived/merged-corpus-canonical.json')); seen=set(); T=[]; S=[]
for r in C:
    s=r['seq_raw']
    if s and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append(s); S.append(r['site'])
tok=collections.Counter(a for s in T for a in s); N=sum(tok.values())
f1=sum(1 for v in tok.values() if v==1); f2=sum(1 for v in tok.values() if v==2)
print(f'texts {len(T)}, sign tokens {N}, sign types {len(tok)}, singletons {f1}, doubletons {f2}')
print(f'P(next sign token is an unseen sign) ~ {f1/N:.4f}; Chao1 total signs ~ {len(tok)+f1*f1/(2*f2):.0f}')
bg=collections.Counter((s[i],s[i+1]) for s in T for i in range(len(s)-1)); Nb=sum(bg.values()); b1=sum(1 for v in bg.values() if v==1)
print(f'P(next sign pair is unseen) ~ {b1/Nb:.3f}')
tx=collections.Counter(tuple(s) for s in T if len(s)>=3); t1=sum(1 for v in tx.values() if v==1)
print(f'P(next text of 3+ signs is a text never seen) ~ {t1/sum(tx.values()):.3f}')
sc=collections.Counter(S); print('share of texts from Mohenjo-daro + Harappa:',round((sc['Mohenjo-daro']+sc['Harappa'])/len(T),3))
