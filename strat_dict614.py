"""S246: Test of an anonymous 614-page Indus-to-Sanskrit syllabic dictionary (PDF from the user). Readings of CISI objects are aligned
with corpus sign sequences of the same length (one syllable per sign, reading order reversed = best); consistency = share of sign tokens
taking the sign's commonest syllable. Control: give each reading to a random other text of the same length 500x."""
import re,json,collections as C,random
full=open('/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/dict614.txt').read(); lines=[l.strip() for l in full.split('\n')]
reads=[]; cur=None
for l in lines:
    if re.fullmatch(r"[a-z0-9\[\]]+(-[a-z0-9\[\]]+)+",l): cur=tuple(l.split('-')); continue
    if cur:
        for cid in re.findall(r'\b([MHLKC]-\d{1,4})[A-Za-z]?\b',l): reads.append((cur,cid))
d=json.load(open('/home/user/Indus-/data/derived/merged-corpus-reading-order.json'))
by=C.defaultdict(set)
for x in d:
    if x['cisi'] and x['cisi']!='-': by[x['cisi']].add(tuple(x['seq']))
pairs=[]; seen=set()
for syl,cid in reads:
    for s in by.get(cid,()):
        if len(s)==len(syl) and (syl,s) not in seen: seen.add((syl,s)); pairs.append((s[::-1],syl))
def cons(pairs):
    m=C.defaultdict(C.Counter)
    for s,syl in pairs:
        for w,y in zip(s,syl): m[w][y]+=1
    return sum(c.most_common(1)[0][1] for c in m.values())/sum(sum(c.values()) for c in m.values())
o=cons(pairs)
bylen=C.defaultdict(list)
for i,(s,syl) in enumerate(pairs): bylen[len(s)].append(i)
random.seed(137); nl=[]
for _ in range(500):
    sy=[p[1] for p in pairs]
    for L,ix in bylen.items():
        v=[sy[i] for i in ix]; random.shuffle(v)
        for i,x in zip(ix,v): sy[i]=x
    nl.append(cons([(p[0],y) for p,y in zip(pairs,sy)]))
print('distinct text-reading pairs',len(pairs))
print('consistency real %.3f  readings given to wrong seals %.3f (max %.3f)  P=%.3f'%(o,sum(nl)/len(nl),max(nl),sum(v>=o for v in nl)/len(nl)))
