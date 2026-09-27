"""S232: Order inside the title zone. Texts containing both a counted pair (short numeral + W220/390/405/900; S198) and a frozen pre-jar title
(590-390/405, 435-690, 840-32, 17-585; S229) — which comes first? Also: counted pair vs the jar. Control: binomial vs 50/50."""
import json,collections as C,math
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30)))-{2}
CT={220,390,405,900}
FT=[(590,390),(590,405),(435,690),(840,32),(17,585)]
seen=set(); order=C.Counter(); ex=[]
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    ft=[i for i in range(len(s)-1) if (s[i],s[i+1]) in FT]
    ct=[i for i in range(1,len(s)) if s[i] in CT and s[i-1] in SHORT and not any(j<=i<=j+1 for j in ft)]
    if ft and ct:
        o='count-first' if max(ct)<min(ft) else 'title-first' if min(ct)>max(ft) else 'mixed'
        order[o]+=1; ex.append((o,s))
print(order)
for o,s in ex[:12]: print(o,s)
n=order['count-first']+order['title-first']; k=max(order['count-first'],order['title-first'])
p=sum(math.comb(n,i) for i in range(k,n+1))/2**n*2
print('two-sided binomial P=%.4f'%min(1,p))
