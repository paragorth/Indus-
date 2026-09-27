"""S186: On two-sided tablets (count face N + leaf W700; text face), does the count value depend on the text face's
content (e.g. its title/closer signs)? Control: shuffle counts among tablets (5000x); statistic = chi-square over
frequent text-face signs x count value."""
import csv,collections as C,random,re
r=list(csv.DictReader(open('data/raw/inscriptions.csv')))
obj=C.defaultdict(list)
for x in r:
    if x['sides'] in ('2',) and x['type'].startswith('TAB'):
        obj[x['id'].split('.')[0]].append(x)
NUMV={32:2,33:3,34:4,35:5,36:6,31:1,37:7}
pairs=[]
for k,faces in obj.items():
    if len(faces)!=2: continue
    seqs=[[int(t) for t in re.findall(r'\d{3}',f['text'])] for f in faces]
    cnt=[i for i,s in enumerate(seqs) if len(s)==2 and 700 in s and any(v in NUMV for v in s)]
    if len(cnt)!=1: continue
    c=seqs[cnt[0]]; t=seqs[1-cnt[0]]
    if 0 in t or len(t)<1: continue
    val=[NUMV[v] for v in c if v in NUMV][0]
    pairs.append((val,t,faces[0]['site']))
print('tablets',len(pairs),C.Counter(p[0] for p in pairs))
sig=C.Counter(v for _,t,_ in pairs for v in set(t)); top=[s for s,n in sig.most_common() if n>=12]
print('text-face signs tested',top)
vals=sorted(set(p[0] for p in pairs))
def chi(P):
    tot=0
    for s in top:
        has=[p[0] for p in P if s in p[1]]; no=[p[0] for p in P if s not in p[1]]
        for v in vals:
            a=sum(x==v for x in has); b=sum(x==v for x in no); n=len(P); rv=(a+b)/n
            for obs,grp in ((a,len(has)),(b,len(no))):
                e=grp*rv
                if e>0: tot+=(obs-e)**2/e
    return tot
obs=chi(pairs); random.seed(0); ge=0
counts=[p[0] for p in pairs]
for _ in range(3000):
    random.shuffle(counts); ge+=chi([(c,p[1],p[2]) for c,p in zip(counts,pairs)])>=obs
print('chi %.1f  perm P=%.4f'%(obs,ge/3000))
for s in top:
    has=C.Counter(p[0] for p in pairs if s in p[1]); print(s,dict(sorted(has.items())))
print('all',dict(sorted(C.Counter(p[0] for p in pairs).items())))
