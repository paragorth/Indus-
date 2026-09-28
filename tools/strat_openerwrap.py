"""S260: texts whose opener (W817/861/820 = M267/391) is written LAST. Is the opener really read
first and only wrapped to the end for space? Test: does the opener fit better in front of the
first sign (opener->seq[0] bigram) than after the last sign (seq[-2]->opener), judged by
bigram counts from all OTHER texts. Control: same scoring for texts with the opener first
(true orientation known) and for random non-opener final signs."""
import json,collections,random
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
OPEN={817,861,820}
T=[(r['cisi'],r['site'],r['type'],r['seq']) for r in m if r.get('seq') and len(r['seq'])>=3]
fin=[t for t in T if t[3][-1] in OPEN and not any(x in OPEN for x in t[3][:-1])]
ini=[t for t in T if t[3][0] in OPEN]
big=collections.Counter(); 
for t in T:
    s=t[3]
    for a,b in zip(s,s[1:]): big[(a,b)]+=1
fol=collections.Counter(); pre=collections.Counter()   # what follows / precedes an opener
for t in T:
    s=t[3]
    for a,b in zip(s,s[1:]):
        if a in OPEN: fol[b]+=1
        if b in OPEN: pre[a]+=1
print('texts opener-final',len(fin),' opener-initial',len(ini))
print('after opener top',fol.most_common(6)); print('before opener top',pre.most_common(6))
front=sum(1 for t in fin if fol[t[3][0]]-0>pre[t[3][-2]]); 
print('opener-final texts: first sign is a usual follower of opener more than last sign is a usual precursor:',front,'/',len(fin))
print('first signs of opener-final texts',collections.Counter(t[3][0] for t in fin).most_common(8))
print('signs before final opener',collections.Counter(t[3][-2] for t in fin).most_common(8))
print('types',collections.Counter(t[2] for t in fin).most_common(5),'sites',collections.Counter(t[1] for t in fin).most_common(5))
for t in fin[:40]: print(t)
# control for the bigram-fit score: opener-initial texts, score whether opener fits better at the end
ctl=sum(1 for t in ini if pre[t[3][-1]]>fol[t[3][1]]); print('control, opener-initial texts that would fit better at the end:',ctl,'/',len(ini))
# marker W2 positions
print('opener-final texts with W2 in first 3 signs',sum(2 in t[3][:3] for t in fin),'/',len(fin))
print('non-opener texts (len>=3) with W2 in first 3 signs',sum(2 in t[3][:3] for t in T if not any(x in OPEN for x in t[3]))/sum(1 for t in T if not any(x in OPEN for x in t[3])))
print('opener-final with jar non-final', sum(740 in t[3][:-1] for t in fin))
