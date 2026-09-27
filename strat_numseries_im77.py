"""S204b: IM77 replication. TALL = M87-M96 (M86 excluded, frame marker); SHORT = M97-M121 (M99 excluded, opener marker). Same statistic and control as S204."""
import csv,random,collections as C
TALL=set(range(87,97)); SHORT=set(range(97,122))-{99}
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
seen=set(); tok=[]
for r in rows:
    s=tuple(int(x) for x in r['signs_clean'].split() if x.isdigit() and x!='0')
    if s in seen: continue
    seen.add(s)
    for i in range(1,len(s)):
        if s[i-1] in TALL|SHORT and s[i] not in TALL|SHORT|{86,99}: tok.append((s[i],'T' if s[i-1] in TALL else 'S'))
def stat(tok):
    g=C.defaultdict(C.Counter)
    for w,sr in tok: g[w][sr]+=1
    out={w:(c['S'],c['T']) for w,c in g.items() if sum(c.values())>=8}
    return sum(max(v)/sum(v) for v in out.values()),out
real,out=stat(tok)
for w,v in sorted(out.items(),key=lambda z:-sum(z[1])): print('M%d short %d tall %d'%(w,*v))
random.seed(9); labs=[sr for w,sr in tok]; null=[]
for _ in range(5000):
    random.shuffle(labs); null.append(stat([(w,l) for (w,_),l in zip(tok,labs)])[0])
print('stat %.1f null %.1f P=%.4f n signs %d'%(real,sum(null)/5000,sum(v>=real for v in null)/5000,len(out)))
