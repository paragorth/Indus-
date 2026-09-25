"""Strategy 2: single sign X <-> two-sign sequence YZ in otherwise identical texts
(the cuneiform logogram-vs-spelling test). Frames need >=2 context signs in total.
Control: the same search on within-segment shuffled corpora."""
import random, collections as C
import indus_core as IC

def find(segs):
    segs=[tuple(s) for s in set(tuple(s) for s in segs) if len(s)>=3]
    # index by (prefix, suffix) with one-sign hole and two-sign hole
    one=C.defaultdict(set); two=C.defaultdict(set)
    for s in segs:
        for i in range(len(s)):
            P,Q=s[:i],s[i+1:]
            if len(P)+len(Q)>=2: one[(P,Q)].add(s[i])
        for i in range(len(s)-1):
            P,Q=s[:i],s[i+2:]
            if len(P)+len(Q)>=2: two[(P,Q)].add(s[i:i+2])
    pairs=C.Counter(); ex={}
    for k in one.keys() & two.keys():
        for x in one[k]:
            for yz in two[k]:
                if x not in yz:
                    pairs[(x,yz)]+=1; ex.setdefault((x,yz),k)
    return pairs,ex

texts=IC.unique_texts(IC.load_corpus()); segs=[s for t in texts for s in t]
pairs,ex=find(segs)
rec=[(p,c) for p,c in pairs.items() if c>=2]
print('X<->YZ substitutions: total',sum(pairs.values()),' distinct',len(pairs),' recurring (>=2 frames)',len(rec))
rng=random.Random(0); cn=[]
for i in range(10):
    sh=[]
    for s in segs: s=list(s); rng.shuffle(s); sh.append(s)
    pp,_=find(sh); cn.append((sum(pp.values()),sum(1 for c in pp.values() if c>=2)))
print('shuffled: total %.0f  recurring %.1f'%(sum(a for a,b in cn)/len(cn),sum(b for a,b in cn)/len(cn)))
for (x,yz),c in sorted(rec,key=lambda t:-t[1])[:25]:
    P,Q=ex[(x,yz)]; print(f'  {x} <-> {"-".join(yz)}  in {c} frames, e.g. [{"-".join(P)}] _ [{"-".join(Q)}]')

# exclusivity: X whose recurring YZ partner is unique and YZ whose X partner is unique
xp=C.defaultdict(set); yp=C.defaultdict(set)
for (x,yz),c in rec: xp[x].add(yz); yp[yz].add(x)
excl=[(x,yz,c) for (x,yz),c in rec if len(xp[x])==1 and len(yp[yz])==1]
print('\none-to-one recurring X<->YZ pairs:',len(excl))
for x,yz,c in sorted(excl,key=lambda t:-t[2]): P,Q=ex[(x,yz)]; print(f'  {x} <-> {"-".join(yz)} ({c} frames) e.g. [{"-".join(P)}] _ [{"-".join(Q)}]')
