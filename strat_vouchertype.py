"""S220: Harappa voucher tablets: does the count (2/3/4 tall + W700) depend on the category of the other face?
Categories: opener text (contains W817/861/820), bare title (176-740(-400) or 740-176-400 in raw order), other (names). Mould duplicates removed.
Control: permute counts among tablets within type 5000x; statistic = TVD between categories."""
import csv,re,collections as C,random
r=list(csv.DictReader(open('data/raw/inscriptions.csv')))
TALL={32:2,33:3,34:4}
objs=C.defaultdict(list)
for x in r: objs[x['id'].split('.')[0]].append(x)
rows=[]; seen=set()
for oid,fs in objs.items():
    f0=fs[0]
    if f0['site']!='Harappa' or not f0['type'].startswith('TAB'): continue
    val=None; txt=()
    for f in fs:
        seq=[int(t) for t in re.findall(r'\d+',f['text'])]
        if 700 in seq and sum(w in TALL for w in seq)==1: val=[TALL[w] for w in seq if w in TALL][0]
        elif seq and 0 not in seq: txt=tuple(seq)
    if not val or not txt: continue
    k=(f0['type'],txt,val)
    if f0['type']=='TAB:B':
        if k in seen: continue
        seen.add(k)
    cat='opener' if set(txt)&{817,861,820} else 'title' if set(txt)<= {176,740,400} else 'name'
    rows.append((f0['type'],cat,val,txt))
tab=C.defaultdict(C.Counter)
for t,c,v,_ in rows: tab[c][v]+=1
for c in tab: print(c,dict(sorted(tab[c].items())),sum(tab[c].values()))
def tvd(rows):
    g=C.defaultdict(C.Counter)
    for t,c,v,_ in rows: g[c][v]+=1
    cs=list(g); tot=0
    for i in range(len(cs)):
        for j in range(i+1,len(cs)):
            a,b=g[cs[i]],g[cs[j]]; A=sum(a.values()); B=sum(b.values())
            tot+=0.5*sum(abs(a[k]/A-b[k]/B) for k in (2,3,4))
    return tot
o=tvd(rows); random.seed(61); nl=[]
for _ in range(5000):
    rr=[]
    for t in ('TAB:B','TAB:I'):
        z=[x for x in rows if x[0]==t]; vs=[x[2] for x in z]; random.shuffle(vs)
        rr+=[(x[0],x[1],v,x[3]) for x,v in zip(z,vs)]
    nl.append(tvd(rr))
print('TVD sum %.3f null %.3f P=%.4f'%(o,sum(nl)/5000,sum(v>=o for v in nl)/5000))
print('opener texts:',C.Counter((x[3],x[2]) for x in rows if x[1]=='opener').most_common(8))
