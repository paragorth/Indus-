"""S310: parse EVERY text with the grammar found so far and measure how much it explains.
Slots (right to left): SUFFIX (W400/W90 after a closer) <- CLOSER (paradigm, S289) <- TITLE phrase (the closer's
own qualifiers, S303: left partners covering the top 60% of that closer's mass, learned on half the texts;
plus the fish stack and 705/706-33 before the arrow, S297-S300; W176 (+W100) before the jar, S301) ; OPENER
(W817/861/820 + optional W2/W60 marker) at the start; COUNT = numeral + next sign; the rest = NAME (unexplained).
Measures on the held-out half: share of sign tokens assigned to a grammar slot; share of texts that parse as
[OPENER] NAME? [TITLE] CLOSER [SUFFIX]. Control: the same parser on held-out texts shuffled within each text
(100x). Output: data/derived/parsed_texts.json (all texts, slots labelled)."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
OPEN={817,861,820}; MARK={2,60}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and len(s)>=2 and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append((r,list(s)))
rnd=random.Random(20); idx=list(range(len(T))); rnd.shuffle(idx); train=set(idx[:len(idx)//2])
left=collections.defaultdict(collections.Counter)
for i in train:
    s=T[i][1][:]
    while len(s)>1 and s[-1] in SUF: s.pop()
    if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
QUAL={}
for c,cnt in left.items():
    tot=sum(cnt.values()); acc=0; q=set()
    for a,n in cnt.most_common():
        if acc/tot>=0.6: break
        q.add(a); acc+=n
    QUAL[c]=q
def parse(s):
    lab=['NAME']*len(s); i=0; j=len(s)
    if s[0] in OPEN:
        lab[0]='OPENER'; i=1
        if len(s)>1 and s[1] in MARK: lab[1]='MARKER'; i=2
    while j-1>i and s[j-1] in SUF and j>=2 and (s[j-2] in CL or s[j-2] in SUF): lab[j-1]='SUFFIX'; j-=1
    if j-1>=i and s[j-1] in CL:
        c=s[j-1]; lab[j-1]='CLOSER'; j-=1
        if c==520:
            if j-2>=i and s[j-1]==33 and s[j-2] in (705,706): lab[j-1]=lab[j-2]='TITLE'; j-=2
            while j-1>=i and s[j-1] in FISH: lab[j-1]='TITLE'; j-=1
        elif c==740:
            if j-1>=i and s[j-1]==100: lab[j-1]='TITLE'; j-=1
            if j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        elif j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        if j-1>=i and s[j-1] in NUM and lab[j]=='TITLE': lab[j-1]='TITLE'; j-=1
    for k in range(i,j-1):
        if s[k] in NUM and lab[k]=='NAME' and lab[k+1]=='NAME': lab[k]=lab[k+1]='COUNT'
    for k in range(i,j):
        if s[k] in NUM and lab[k]=='NAME': lab[k]='COUNT'
    return lab
def measure(texts):
    tok=exp=0; full=0
    for s in texts:
        L=parse(s); tok+=len(s); exp+=sum(l!='NAME' for l in L)
        full+= ('CLOSER' in L)
    return exp/tok, full/len(texts)
test=[T[i][1] for i in idx[len(idx)//2:]]
e,f=measure(test)
nulls=[measure([rnd.sample(s,len(s)) for s in test]) for _ in range(100)]
ne=sorted(x for x,_ in nulls); nf=sorted(y for _,y in nulls)
print(f'held-out texts {len(test)}: tokens explained by grammar {e:.3f} (shuffled null median {ne[50]:.3f}, max {ne[-1]:.3f}); texts with a closer-headed title {f:.3f} (null {nf[50]:.3f}, max {nf[-1]:.3f})')
out=[]; slotc=collections.Counter(); namelen=collections.Counter()
for r,s in T:
    L=parse(s); out.append({'cisi':r['cisi'],'site':r['site'],'type':r['type'],'seq':s,'slots':L})
    slotc.update(L); namelen[sum(l=='NAME' for l in L)]+=1
json.dump(out,open('data/derived/parsed_texts.json','w'))
tot=sum(slotc.values()); print('all texts: slot shares',{k:round(v/tot,3) for k,v in slotc.most_common()})
print('NAME length distribution (signs per text):',sorted(namelen.items())[:10])
pat=collections.Counter(tuple(dict.fromkeys(l for l in o['slots'])) for o in out)
print('commonest slot patterns:',pat.most_common(8))
