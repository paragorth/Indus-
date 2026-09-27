"""S214: Is 'numeral before counted sign' a reading-order rule? Compare R/L texts with L/R texts (seq is in reading order).
Counted signs (S198): W740, 220, 700, 390, 900, 405 and the fixed partners 520, 585, 575, comb. Share of adjacent numeral-sign pairs where
the numeral comes first. Control: the same share for all other non-numeral signs adjacent to a numeral (baseline), and Fisher-like permutation
of direction labels 5000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(32,38))|{39,55,56})-{2}
T={740,220,700,390,900,405,520,585,575}|set(range(422,427))
rows=[]; seen=set()
for x in d:
    dr=x['dir.'].strip()
    if dr not in ('R/L','L/R'): continue
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    for i in range(1,len(s)):
        a,b=s[i-1],s[i]
        if a in NUM and b in T: rows.append((dr,1))
        elif b in NUM and a in T and not (i>=2 and s[i-2] in NUM): rows.append((dr,0))
for dr in ('R/L','L/R'):
    z=[v for r,v in rows if r==dr]; print(dr,'numeral-first %d/%d = %.2f'%(sum(z),len(z),sum(z)/len(z)))
diff=lambda R:(sum(v for r,v in R if r=='R/L')/sum(1 for r,v in R if r=='R/L'))-(sum(v for r,v in R if r=='L/R')/sum(1 for r,v in R if r=='L/R'))
o=diff(rows); labs=[r for r,v in rows]; random.seed(41); nl=[]
for _ in range(5000):
    random.shuffle(labs); nl.append(diff([(l,v) for l,(r,v) in zip(labs,rows)]))
print('R/L minus L/R %.3f  P(|null|>=|obs|)=%.4f'%(o,sum(abs(v)>=abs(o) for v in nl)/5000))
