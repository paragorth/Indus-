"""S322: fair version of S321. Mesopotamian seal legends (CDLI ATF, @seal sections and seal objects), ONE copy per
distinct full legend (= one per seal, removing repeated impressions). Owner's name = tokens before the first title/
kin word (dub-sar, dumu, arad2/ir3, dam, szabra, ensi2, sukkal, kiszib3, lu2 of office excluded as heads). Uniqueness
vs a bigram model trained on those names, size-matched to the 1,748 Indus middles (50 draws)."""
import sys,random,collections,statistics as st
sys.argv=['x']; sys.path.insert(0,'.'); import strat_repcal as S
G=S.groups()
legs=set()
for k,v in G.items():
    if k[0]=='seal':
        for s in v:
            if 2<=len(s)<=20: legs.add(tuple(x.lower() for x in s))
STOP={'dub','dumu','arad2','ir3','ir11','dam','szabra','ensi2','sukkal','kiszib3','kiszib','sanga','nu','gudu4','sipa','nar','ugula'}
names=[]
for s in legs:
    n=[]
    for x in s:
        if x in STOP: break
        n.append(x)
    if 2<=len(n)<=6: names.append(tuple(n))
print('distinct legends',len(legs),'names extracted',len(names))
def uniq(ms): c=collections.Counter(ms); return sum(1 for m in ms if c[m]==1)/len(ms)
rs=[];ns=[];rnd=random.Random(30)
for _ in range(50):
    sub=rnd.sample(names,min(1748,len(names))); rs.append(uniq(sub))
    b=collections.defaultdict(collections.Counter)
    for m in sub:
        p='S'
        for c in m: b[p][c]+=1; p=c
    def g(n):
        out=[];p='S'
        for _ in range(n):
            src=b[p] if b[p] else b['S']; ks,ws=zip(*src.items()); c=rnd.choices(ks,ws)[0]; out.append(c); p=c
        return tuple(out)
    ns.append(uniq([g(len(m)) for m in sub]))
print(f'one legend per seal, size-matched: real names unique {st.mean(rs):.3f} vs bigram null {st.mean(ns):.3f}; ratio {st.mean(rs)/st.mean(ns):.3f}  (Indus middles 0.846/0.874 = 0.968)')
print(collections.Counter(names).most_common(6))
