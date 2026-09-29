"""S321 (audit idea 3): calibrated name-reuse test. Real personal names (Ur III and later seal legends, line 1 = owner's
name, CDLI; syllables split on '-') vs a bigram model trained on those names: are real names MORE or LESS unique than
the bigram null? Same statistic and null as tools/audit_middles.py (Indus middles: 0.846 vs null 0.874). If real
names sit clearly below their null (names recur) and Indus middles behave the same, the 'name' reading gains support."""
import random,collections,re
SP='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/seal_line1.txt'
names=[]
for l in open(SP,errors='ignore'):
    l=re.sub(r'[#\[\]!?<>]','',l.strip().lower())
    if not l or ' ' in l or 'x' in l.split('-'): continue
    t=tuple(x for x in l.split('-') if x)
    if 2<=len(t)<=6: names.append(t)
def uniq(ms): c=collections.Counter(ms); return sum(1 for m in ms if c[m]==1)/len(ms)
big=collections.defaultdict(collections.Counter)
for m in names:
    p='S'
    for c in m: big[p][c]+=1; p=c
def gen(n):
    out=[];p='S'
    for _ in range(n):
        src=big[p] if big[p] else big['S']; ks,ws=zip(*src.items()); c=random.choices(ks,ws)[0]; out.append(c); p=c
    return tuple(out)
random.seed(0); L=[len(m) for m in names]; nu=sorted(uniq([gen(n) for n in L]) for _ in range(100))
o=uniq(names)
print(f'real seal-owner names n={len(names)}; observed unique {o:.3f}; bigram null median {nu[50]:.3f} [{nu[2]:.3f}, {nu[97]:.3f}]; ratio {o/nu[50]:.3f}')
print('Indus middles (audit): 0.846 vs 0.874; ratio 0.968')
print('most repeated real names:',collections.Counter(names).most_common(5))
# size-matched and duplicate-reduced: one copy per (name, first CDLI occurrence) is unavailable, so dedupe exact names
# within random blocks is not possible; instead subsample 1,748 names (the Indus middle count) and repeat.
rs=[];ns=[]
for k in range(50):
    sub=random.sample(names,1748); rs.append(uniq(sub))
    b2=collections.defaultdict(collections.Counter)
    for m in sub:
        p='S'
        for c in m: b2[p][c]+=1; p=c
    def g2(n):
        out=[];p='S'
        for _ in range(n):
            src=b2[p] if b2[p] else b2['S']; ks,ws=zip(*src.items()); c=random.choices(ks,ws)[0]; out.append(c); p=c
        return tuple(out)
    ns.append(uniq([g2(len(m)) for m in sub]))
import statistics as st
print(f'size-matched (n=1,748, 50 draws): real names unique {st.mean(rs):.3f} vs own bigram null {st.mean(ns):.3f}; ratio {st.mean(rs)/st.mean(ns):.3f}')
