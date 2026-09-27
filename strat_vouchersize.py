"""S219: Harappa voucher tablets ('tall N + W700', N = 2/3/4; S216). Does the count go with the tablet's physical size
(horizontal x vertical mm, raw file), as it would for value tokens? Control: permute counts among tablets within type (TAB:B / TAB:I) 5000x;
statistic = Spearman-like correlation of count with area."""
import csv,re,collections as C,random
r=list(csv.DictReader(open('data/raw/inscriptions.csv')))
TALL={32:2,33:3,34:4,35:5,36:6}
objs=C.defaultdict(list)
for x in r: objs[x['id'].split('.')[0]].append(x)
rows=[]; TLEN=[]
for oid,fs in objs.items():
    f0=fs[0]
    if f0['site']!='Harappa' or not f0['type'].startswith('TAB'): continue
    val=None; tl=0
    for f in fs:
        seq=[int(t) for t in re.findall(r'\d+',f['text'])]
        if 700 in seq:
            v=[TALL[w] for w in seq if w in TALL]
            if len(v)==1: val=v[0]
        elif seq: tl=max(tl,len(seq))
    try: h=float(f0['horizontal(mm)']); v_=float(f0['vertical(mm)'])
    except: continue
    if val and h>0 and v_>0: rows.append((f0['type'],val,h*v_,f0['time']+'|'+f0['area-section'][:6])); TLEN.append(tl)
print('tablets with count and size',len(rows),C.Counter((t,v) for t,v,a,tm in rows))
for t in ('TAB:B','TAB:I'):
    for v in (2,3,4):
        a=sorted(x[2] for x in rows if x[0]==t and x[1]==v)
        if a: print(t,v,'n',len(a),'median area %.0f mm2'%a[len(a)//2])
def rank(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]); rk=[0]*len(xs)
    for k,i in enumerate(o): rk[i]=k
    return rk
def corr(rows):
    tot=0
    for t in ('TAB:B','TAB:I'):
        z=[(v,a) for tt,v,a,tm in rows if tt==t]
        if len(z)<5: continue
        rv=rank([v+random.random()*1e-6 for v,a in z]); ra=rank([a for v,a in z]); n=len(z)
        m=(n-1)/2; tot+=sum((x-m)*(y-m) for x,y in zip(rv,ra))/(sum((x-m)**2 for x in rv))
    return tot/2
random.seed(53); o=corr(rows); nl=[]
for _ in range(3000):
    rr=[]
    for t in ('TAB:B','TAB:I'):
        z=[x for x in rows if x[0]==t]; vs=[x[1] for x in z]; random.shuffle(vs)
        rr+=[(t,v,x[2],x[3]) for x,v in zip(z,vs)]
    nl.append(corr(rr))
print('mean rank corr %.3f null %.3f P(|null|>=|obs|)=%.4f'%(o,sum(nl)/3000,sum(abs(v)>=abs(o) for v in nl)/3000))

# S219b: stratify by type x (period|mound): permute counts only within strata; within-stratum correlation
def corr2(rows):
    tot=0;w=0
    st=C.defaultdict(list)
    for t,v,a,k in rows: st[(t,k)].append((v,a))
    for key,z in st.items():
        if len(z)<8 or len({v for v,a in z})<2: continue
        rv=rank([v+random.random()*1e-6 for v,a in z]); ra=rank([a for v,a in z]); n=len(z); m=(n-1)/2
        tot+=n*sum((x-m)*(y-m) for x,y in zip(rv,ra))/sum((x-m)**2 for x in rv); w+=n
    return tot/w
o2=corr2(rows); nl2=[]
st=C.defaultdict(list)
for i,x in enumerate(rows): st[(x[0],x[3])].append(i)
for _ in range(3000):
    rr=list(rows)
    for key,ix in st.items():
        vs=[rows[i][1] for i in ix]; random.shuffle(vs)
        for i,v in zip(ix,vs): rr[i]=(rows[i][0],v,rows[i][2],rows[i][3])
    nl2.append(corr2(rr))
print('within type x period|mound: corr %.3f null %.3f P=%.4f'%(o2,sum(nl2)/3000,sum(abs(v)>=abs(o2) for v in nl2)/3000))
print('strata used',[(k,len(v)) for k,v in st.items() if len(v)>=8])

# S219c: stratify by type x text length of the other face (0 = no text face)
rows3=[(t,v,a,str(min(l,6))) for (t,v,a,k),l in zip(rows,TLEN)]
o3=corr2(rows3); st3=C.defaultdict(list)
for i,x in enumerate(rows3): st3[(x[0],x[3])].append(i)
nl3=[]
for _ in range(3000):
    rr=list(rows3)
    for key,ix in st3.items():
        vs=[rows3[i][1] for i in ix]; random.shuffle(vs)
        for i,v in zip(ix,vs): rr[i]=(rows3[i][0],v,rows3[i][2],rows3[i][3])
    nl3.append(corr2(rr))
print('within type x text length: corr %.3f null %.3f P=%.4f'%(o3,sum(nl3)/3000,sum(abs(v)>=abs(o3) for v in nl3)/3000))
