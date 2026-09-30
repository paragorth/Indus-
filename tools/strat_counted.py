"""S350: which signs are COUNTED? For every sign X (>=15 tokens in distinct texts), share of its tokens directly
preceded by a numeral sign. Control: same share after shuffling signs within each text (1000x); z-score.
Tests the W390 = timber-unit anchor (S346): a counted good should rank among the most-counted signs,
and its counts should vary (several different N), unlike a fixed title."""
import json,collections,random,statistics as st,sys
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1,3,4,5,16,17,18,31,32,33,34}
for key in ('seq_raw','seq_strong','seq_all'):
    T=list({(r['site'],tuple(r[key])) for r in C if r.get(key) and len(r[key])>=2})
    texts=[t for _,t in T]
    def counts(tx):
        tot=collections.Counter(); pre=collections.Counter(); nums=collections.defaultdict(collections.Counter)
        for s in tx:
            for i,x in enumerate(s):
                tot[x]+=1
                if i>0 and s[i-1] in NUM: pre[x]+=1; nums[x][s[i-1]]+=1
        return tot,pre,nums
    tot,pre,nums=counts(texts)
    cand=[x for x in tot if tot[x]>=15 and x not in NUM]
    random.seed(1); sims=collections.defaultdict(list)
    for _ in range(300):
        sh=[random.sample(s,len(s)) for s in texts]
        _,p2,_=counts(sh)
        for x in cand: sims[x].append(p2[x])
    rows=[]
    for x in cand:
        m=st.mean(sims[x]); sd=st.pstdev(sims[x]) or 1
        rows.append(((pre[x]-m)/sd,x,pre[x],tot[x],round(m,1),dict(nums[x].most_common(4))))
    rows.sort(reverse=True)
    print(f'== {key}: {len(texts)} texts, {len(cand)} signs tested')
    for z,x,p,t,m,n in rows[:15]: print(f'  W{x}: {p}/{t} after numeral (shuffle {m}), z={z:.1f}, numerals {n}')
    r390=[r for r in rows if r[1]==390]
    if r390: print('  W390 rank',rows.index(r390[0])+1,'of',len(rows))
