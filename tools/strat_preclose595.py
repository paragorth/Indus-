"""S288: follow-up of S287. In IM77, M252 recurs right before a text-final M391.
Wells: M252 = W595, M391 = W820, M267 = W817/861, jar M342 = W740.
Tests on the canonical corpus (seq_raw, seq_strong, seq_all), site+text dedup, len >= 3:
 (a) share of opener-final texts whose penultimate sign is W595, vs share of all texts with
     that penultimate sign for other final signs (peer control: every final sign with n >= 10);
 (b) where W595 sits in texts: before the jar closer? before a final opener? position profile;
 (c) permutation: shuffle final signs across texts keeping penultimates, 10,000 times."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
OPEN={817,861,820}; JAR=740; X=595
random.seed(1)
for key in ['seq_raw','seq_strong','seq_all']:
    seen=set(); T=[]
    for r in C:
        s=r.get(key)
        if not s or len(s)<3: continue
        k=(r['site'],tuple(s))
        if k in seen: continue
        seen.add(k); T.append(s)
    fin=collections.Counter(s[-1] for s in T)
    pen=collections.Counter((s[-1],s[-2]) for s in T)
    tot595=sum(s.count(X) for s in T); texts595=sum(X in s for s in T)
    print(f'== {key}: texts {len(T)}, W595 tokens {tot595} in {texts595} texts')
    of=[s for s in T if s[-1] in OPEN and not any(x in OPEN for x in s[:-1])]
    k=sum(s[-2]==X for s in of)
    print(f'opener-final {len(of)}, W595 penultimate {k}')
    for s in of:
        if X in s: print('   ',s)
    # peer control: rate of W595 penultimate by final sign
    rates=[]
    for f,n in fin.items():
        if n>=10: rates.append((pen[(f,X)]/n,f,n,pen[(f,X)]))
    rates.sort(reverse=True); print('top finals by W595-penultimate rate',[(round(a,3),f,n,c) for a,f,n,c in rates[:8]])
    obs=sum(1 for s in T if s[-1] in OPEN and s[-2]==X)
    finals=[s[-1] for s in T]; pens=[s[-2] for s in T]; ge=0
    for _ in range(10000):
        random.shuffle(finals)
        if sum(1 for f,p in zip(finals,pens) if f in OPEN and p==X)>=obs: ge+=1
    print(f'W595 directly before any final opener: {obs}; permutation P = {(ge+1)/10001:.4f}')
    # position profile of W595
    pos=collections.Counter()
    nxt=collections.Counter(); prv=collections.Counter()
    for s in T:
        for i,a in enumerate(s):
            if a==X:
                pos['initial' if i==0 else 'final' if i==len(s)-1 else 'medial']+=1
                nxt[s[i+1] if i+1<len(s) else 'END']+=1; prv[s[i-1] if i else 'BEG']+=1
    print('W595 position',dict(pos)); print('  next',nxt.most_common(8)); print('  prev',prv.most_common(8))
    jf=[s for s in T if s[-1]==JAR]
    print(f'jar-final {len(jf)}, W595 penultimate {sum(s[-2]==X for s in jf)}; W595 anywhere in jar-final {sum(X in s for s in jf)} ({sum(X in s for s in jf)/len(jf):.3f}) vs all texts {texts595/len(T):.3f}')
    oi=[s for s in T if s[0] in OPEN]
    print(f'opener-initial {len(oi)}, W595 anywhere {sum(X in s for s in oi)} ; opener-final W595 anywhere {sum(X in s for s in of)}')

# (d) segment-closer profile: share of tokens that are text-final or followed by marker W1/W2
print('\n== (d) segment-closer rate (final or followed by W1/W2), signs with >= 25 tokens, seq_raw')
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if not s or len(s)<3 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s))); T.append(s)
tok=collections.Counter(); cl=collections.Counter()
for s in T:
    for i,a in enumerate(s):
        tok[a]+=1
        if i==len(s)-1 or s[i+1] in (1,2): cl[a]+=1
R=sorted(((cl[a]/tok[a],a,tok[a]) for a in tok if tok[a]>=25),reverse=True)
for i,(r,a,n) in enumerate(R[:12]): print(i+1,a,n,round(r,2))
print('W595 rank',[i+1 for i,(r,a,n) in enumerate(R) if a==X],'of',len(R))
# (e) avoidance of the jar: texts with W595 that contain W740, vs expected from length
import math
has=[s for s in T if X in s]; obs=sum(JAR in s for s in has)
exp=0
for s in has:
    L=len(s)-1
    same=[t for t in T if len(t)==len(s)]
    exp+=sum(JAR in t for t in same)/len(same)
print(f'(e) W595 texts with jar: {obs} observed vs {exp:.1f} expected by length')
# peers: signs with 30-60 tokens, obs/exp ratio
rat=[]
for a in tok:
    if 30<=tok[a]<=60 and a not in (JAR,X):
        hs=[s for s in T if a in s]; o=sum(JAR in s for s in hs)
        e=sum(sum(JAR in t for t in T if len(t)==len(s))/max(1,sum(1 for t in T if len(t)==len(s))) for s in hs)
        rat.append((o/e if e else 0,a))
rat.sort(); print('peer obs/exp jar ratios (lowest 8)',[(round(r,2),a) for r,a in rat[:8]],'n peers',len(rat))
print('W595 ratio',round(obs/exp,2),'rank among peers',sum(r<obs/exp for r,a in rat)+1)
