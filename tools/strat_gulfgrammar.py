"""S277: do the round/western seal texts form a coherent formula of their own?
Leave-one-out: score each western text (all round Gulf-type + foreign-find seal texts, dedup) with a
bigram model trained on the OTHER western texts, vs a model trained on an equal-sized random set of
home square seals. Control: the same comparison for random home texts (own-group model should win
there too, giving the baseline advantage of 'same group' at this sample size)."""
import json,collections,math,random
m=json.load(open('data/derived/merged-corpus-canonical.json'))
HOME={'Harappa','Mohenjo-daro','Dholavira','Lothal','Kalibangan','Chanhu-daro','Banawali','Rakhigarhi'}
FOR={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain",'Susa','Tell Umma','Tello','Ur','Gonur Depe','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun'}
seen=set(); W=[]; H=[]
for r in m:
    s=r.get('seq_raw'); t=str(r['type'])
    if not s or len(s)<2: continue
    k=(r['site'],tuple(s))
    if k in seen: continue
    seen.add(k)
    if t=='SEAL:C' and r['site'] not in HOME or (r['site'] in FOR and t.startswith('SEAL')): W.append(s)
    elif r['site'] in HOME and t=='SEAL:S': H.append(s)
V=len({x for s in W+H for x in s})+2
def train(T):
    c=collections.Counter(); u=collections.Counter()
    for s in T:
        t=['^']+s+['$']
        for a,b in zip(t,t[1:]): c[(a,b)]+=1; u[a]+=1
    return c,u
def lp(model,s,alpha=0.1):
    c,u=model; t=['^']+s+['$']
    return sum(math.log((c[(a,b)]+alpha)/(u[a]+alpha*V)) for a,b in zip(t,t[1:]))/(len(t)-1)
random.seed(8)
def loo_adv(G,pool,R=200):
    adv=[]
    for i,s in enumerate(G):
        own=train(G[:i]+G[i+1:])
        other=[lp(train(random.sample(pool,len(G)-1)),s) for _ in range(20)]
        adv.append(lp(own,s)-sum(other)/len(other))
    return sum(adv)/len(adv), sum(a>0 for a in adv)
print('western texts',len(W),'home square',len(H))
a,npos=loo_adv(W,H); print(f'western: own-group minus home-model log-prob per transition {a:+.3f}; texts better predicted by own group {npos}/{len(W)}')
res=[]
for rep in range(30):
    G=random.sample(H,len(W)); pool=[s for s in H if s not in G]
    res.append(loo_adv(G,pool)[0])
res.sort(); print(f'control (random home groups of the same size): mean {sum(res)/len(res):+.3f}, range {res[0]:+.3f}..{res[-1]:+.3f}')
bg=collections.Counter(); hb=collections.Counter()
for s in W:
    for a,b in set(zip(s,s[1:])): bg[(a,b)]+=1
for s in H:
    for a,b in set(zip(s,s[1:])): hb[(a,b)]+=1
print('bigrams in >=2 western texts (count western / share of home square seals):')
for (a,b),c in bg.most_common():
    if c<2: break
    print(f'  {a}-{b}: {c} western texts | home {hb[(a,b)]} ({hb[(a,b)]/len(H):.3%})')
fs=collections.Counter(s[0] for s in W); ls=collections.Counter(s[-1] for s in W)
print('first signs',fs.most_common(6)); print('last signs',ls.most_common(6))
