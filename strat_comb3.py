"""S197: The comb-box signs W422-426 (no Mahadevan bridge entry). Is the sign before them fixed (W3 = three strokes)?
Does the comb's tooth variant track the numeral before it (a count), or not?
Controls: (a) how often W3 precedes any other sign of similar frequency; (b) shuffle comb variants 10000x vs preceding numeral."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
COMB=set(range(422,427)); NUMS={1,2,3,4,5}
prev=C.Counter(); var_prev=[]; n=0
for x in d:
    s=x['seq']
    for i,w in enumerate(s):
        if w in COMB:
            n+=1; p=s[i-1] if i else None; prev[p]+=1
            if p in NUMS: var_prev.append((w,p))
print('comb tokens',n,'preceded by',prev.most_common(8))
k3=prev[3]
# control a: for every sign type with 10-60 tokens, share of tokens preceded by W3
freq=C.Counter(w for x in d for w in x['seq'])
pre3=C.Counter(); tot=C.Counter()
for x in d:
    s=x['seq']
    for i,w in enumerate(s):
        tot[w]+=1
        if i and s[i-1]==3: pre3[w]+=1
cands=[(pre3[w]/tot[w],w) for w in tot if 10<=tot[w]<=60 and w not in COMB]
cands.sort(reverse=True)
print('comb share after W3 %.2f'%(k3/n))
print('other signs (10-60 tokens) with highest share after W3:',[(w,round(r,2),tot[w]) for r,w in cands[:8]])
print('median share %.3f'%sorted(r for r,w in cands)[len(cands)//2])
print('signs with share >= comb:',sum(r>=k3/n for r,w in cands),'of',len(cands))
# control b: variant vs preceding numeral (only informative if numeral varies)
print('variant/numeral pairs',C.Counter(var_prev))
# where the W3 occurs, what sits before 3+comb
pp=C.Counter()
for x in d:
    s=x['seq']
    for i,w in enumerate(s):
        if w in COMB and i>=2 and s[i-1]==3: pp[s[i-2]]+=1
print('sign before 3+comb',pp.most_common())
# 3+comb position: final?
fin=sum(1 for x in d for i,w in enumerate(x['seq']) if w in COMB and i==len(x['seq'])-1)
print('comb text-final',fin,'of',n)
