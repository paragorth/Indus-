"""S221: Does S219's pattern (higher count, smaller object) hold on seals? Square seals with exactly one counted pair
(numeral + W220/390/405/900/740; W2 excluded). Value from glyph strokes. Control: permute values within site 5000x (rank correlation with area)."""
import csv,re,collections as C,random
r=list(csv.DictReader(open('data/raw/inscriptions.csv')))
VAL={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9}
VAL.pop(2)
CT={220,390,405,900,740}
rows=[]; KEY=[]
for x in r:
    if x['type']!='SEAL:S' or not x['id'].endswith('.1'): continue
    seq=[int(t) for t in re.findall(r'\d+',x['text'])]
    # raw text order may be reversed; take numeral adjacent to a counted sign on either side
    hits=[(VAL[seq[i]],(seq[i+1] if i+1<len(seq) and seq[i+1] in CT else seq[i-1])) for i in range(len(seq)) if seq[i] in VAL and ((i+1<len(seq) and seq[i+1] in CT) or (i>0 and seq[i-1] in CT))]
    vals=[h_[0] for h_ in hits]
    try: h=float(x['horizontal(mm)']); v=float(x['vertical(mm)'])
    except: continue
    if len(vals)==1 and h>0 and v>0: rows.append((x['site'],vals[0],h*v)); KEY.append((x['site'],hits[0][1],min(len(seq),7)))
print('seals',len(rows),C.Counter(v for s,v,a in rows))
for v in sorted(set(v for s,v,a in rows)):
    a=sorted(x[2] for x in rows if x[1]==v)
    if len(a)>=5: print(' value',v,'n',len(a),'median area %.0f'%a[len(a)//2])
def rk(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]); q=[0]*len(xs)
    for k,i in enumerate(o): q[i]=k
    return q
def corr(rows):
    rv=rk([v+random.random()*1e-6 for s,v,a in rows]); ra=rk([a for s,v,a in rows]); n=len(rows); m=(n-1)/2
    return sum((x-m)*(y-m) for x,y in zip(rv,ra))/sum((x-m)**2 for x in rv)
random.seed(67); o=corr(rows); by=C.defaultdict(list)
for i,x in enumerate(rows): by[x[0]].append(i)
nl=[]
for _ in range(3000):
    rr=list(rows)
    for s,ix in by.items():
        vs=[rows[i][1] for i in ix]; random.shuffle(vs)
        for i,v in zip(ix,vs): rr[i]=(s,v,rows[i][2])
    nl.append(corr(rr))
print('rank corr %.3f P=%.4f'%(o,sum(abs(v)>=abs(o) for v in nl)/3000))

# S221b: permute values only within (site, counted sign, text length)
by2=C.defaultdict(list)
for i,k in enumerate(KEY): by2[k].append(i)
nl2=[]
for _ in range(3000):
    rr=list(rows)
    for k,ix in by2.items():
        vs=[rows[i][1] for i in ix]; random.shuffle(vs)
        for i,v in zip(ix,vs): rr[i]=(rows[i][0],v,rows[i][2])
    nl2.append(corr(rr))
print('within site x counted sign x length: null mean %.3f P=%.4f'%(sum(nl2)/3000,sum(v<=o for v in nl2)/3000))
