"""S304: are title phrases stable over time while name material turns over? Two time splits: Mohenjo-daro
Intermediate vs Late (period field) and Harappa 3B vs 3C (phase field). Features per text: (a) the closing title
phrase (qualifier + closer, S303); (b) the first 'middle' sign (after an opener/marker, not a numeral or closer).
Statistic: Jensen-Shannon divergence between the two periods, standardised against 2,000 permutations of the period
labels (z = (obs - null mean)/null sd). Prediction: titles z near 0, middles z > 0. seq_raw, site+text dedup."""
import json,collections,random,math
C=json.load(open('data/derived/merged-corpus-canonical.json'))
CL={740,520,151,156,527,226,617,154,158,236,700}; SUF={400,90}; FR={817,861,820,2,60}
NUM={1,2,3,4,5,16,17,18,31,32,33,34}
def feats(s):
    core=list(s)
    while len(core)>1 and core[-1] in SUF: core.pop()
    title=(core[-2],core[-1]) if len(core)>=2 and core[-1] in CL else None
    mid=next((a for a in s if a not in FR and a not in NUM and a not in CL and a not in SUF),None)
    return title,mid
def jsd(a,b):
    ca,cb=collections.Counter(a),collections.Counter(b); na,nb=len(a),len(b); out=0
    for k in set(ca)|set(cb):
        p,q=ca[k]/na,cb[k]/nb; m=(p+q)/2
        if p: out+=0.5*p*math.log(p/m)
        if q: out+=0.5*q*math.log(q/m)
    return out
rnd=random.Random(15)
splits={'Mohenjo-daro Intermediate vs Late':lambda r:(r['period'] if r['site']=='Mohenjo-daro' and r['period'] in ('Intermediate','Late') else None),
        'Harappa 3B vs 3C':lambda r:(r['phase'] if r['site']=='Harappa' and r['period']=='3' and r['phase'] in ('B','C') else None)}
for name,g in splits.items():
    seen=set(); rows=[]
    for r in C:
        s=r['seq_raw']; lab=g(r)
        if not lab or not s or len(s)<2 or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); rows.append((lab,)+feats(s))
    for idx,fname in ((1,'title phrase'),(2,'first middle sign')):
        R=[(l,f) for l,*F in rows for f in [F[idx-1]] if f is not None]
        labs=[l for l,_ in R]; fs=[f for _,f in R]; L=sorted(set(labs))
        obs=jsd([f for l,f in R if l==L[0]],[f for l,f in R if l==L[1]])
        null=[]
        for _ in range(2000):
            rnd.shuffle(labs); null.append(jsd([f for l,f in zip(labs,fs) if l==L[0]],[f for l,f in zip(labs,fs) if l==L[1]]))
        mu=sum(null)/len(null); sd=(sum((x-mu)**2 for x in null)/len(null))**.5
        p=(sum(x>=obs for x in null)+1)/2001
        print(f'{name}: {fname}: n={len(R)} ({collections.Counter(labs)}), JSD {obs:.3f}, null {mu:.3f}, z={(obs-mu)/sd:.2f}, P={p:.4f}')
