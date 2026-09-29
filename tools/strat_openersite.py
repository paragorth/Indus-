"""S332: the five opener units (817-2, 861-2, 820-2/60, 920-60, 692-60): does each belong to a city or an object
type? Opener x site and opener x object type (seal / tablet / tag), mutual information vs labels permuted (5,000x);
per-site shares."""
import json,collections,random,math
C=json.load(open('data/derived/merged-corpus-canonical.json'))
seen=set(); rows=[]
for r in C:
    s=r['seq_raw']
    if not s or len(s)<2 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    o=None
    if s[0] in (817,861,820) and s[1] in (2,60): o=s[0]
    elif s[0] in (920,692) and s[1]==60: o=s[0]
    if o: rows.append((o,r['site'] if r['site'] in ('Mohenjo-daro','Harappa','Dholavira','Lothal','Kalibangan','Chanhu-daro') else 'other',r['type'].split(':')[0]))
def mi(xs,ys):
    n=len(xs); cx=collections.Counter(xs); cy=collections.Counter(ys); cxy=collections.Counter(zip(xs,ys))
    return sum(c/n*math.log(c*n/(cx[x]*cy[y])) for (x,y),c in cxy.items())
rnd=random.Random(37)
for name,k in (('site',1),('object',2)):
    X=[r[0] for r in rows]; Y=[r[k] for r in rows]; o=mi(X,Y); ge=0
    for _ in range(5000): rnd.shuffle(Y); ge+=mi(X,Y)>=o
    print(f'opener x {name}: n={len(rows)} MI={o:.4f} P={(ge+1)/5001:.4f}')
tab=collections.defaultdict(collections.Counter)
for o,s,t in rows: tab[s][o]+=1
for s,c in tab.items(): n=sum(c.values()); print(f'  {s:12s} n={n:3d}',{k:round(v/n,2) for k,v in sorted(c.items())})
tab=collections.defaultdict(collections.Counter)
for o,s,t in rows: tab[t][o]+=1
for s,c in tab.items(): n=sum(c.values()); print(f'  {s:12s} n={n:3d}',{k:round(v/n,2) for k,v in sorted(c.items())})
print('== opener x first content sign after the connective, and x closer')
seen=set(); R=[]
for r in C:
    s=r['seq_raw']
    if not s or len(s)<4 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    if (s[0] in (817,861,820) and s[1] in (2,60)) or (s[0] in (920,692) and s[1]==60): R.append((s[0],s[2],s[-1]))
for name,k in (('first content sign',1),('last sign',2)):
    X=[r[0] for r in R]; Y=[r[k] for r in R]; cy=collections.Counter(Y); Y=[y if cy[y]>=5 else 'rare' for y in Y]
    o=mi(X,Y); ge=0
    for _ in range(5000): rnd.shuffle(Y); ge+=mi(X,Y)>=o
    print(f'opener x {name}: n={len(R)} MI={o:.4f} P={(ge+1)/5001:.4f}')
    tab=collections.defaultdict(collections.Counter)
    for x,y in zip([r[0] for r in R],[r[k] for r in R]): tab[x][y]+=1
    for x,c in tab.items(): print('   ',x,c.most_common(5))
