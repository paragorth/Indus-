"""S290: does the choice of closing unit depend on site, medium or emblem? Texts ending
(before an optional W400/W90 suffix) in the jar or one of the S289 closers. Mutual information
between closer and each factor, with a permutation null (factor labels shuffled, 5,000x).
Medium = SEAL vs TAB; site = Mohenjo-daro vs Harappa vs other; emblem = seal 'symbol' field
(unicorn vs other), for seals only."""
import json,collections,random,math
C=json.load(open('data/derived/merged-corpus-canonical.json'))
CL={740,154,151,158,527,520,156,226,617,236,700}; SUF={400,90}; random.seed(3)
def mi(xs,ys):
    n=len(xs); cx=collections.Counter(xs); cy=collections.Counter(ys); cxy=collections.Counter(zip(xs,ys))
    return sum(c/n*math.log(c*n/(cx[x]*cy[y])) for (x,y),c in cxy.items())
for key in ['seq_raw','seq_strong']:
    rows=[]; seen=set()
    for r in C:
        s=r.get(key)
        if not s or len(s)<2 or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); core=list(s)
        while len(core)>1 and core[-1] in SUF: core.pop()
        if core[-1] in CL: rows.append((core[-1],r))
    print(f'== {key}: {len(rows)} texts with a paradigm closer; closers',collections.Counter(c for c,_ in rows).most_common())
    facs={
     'medium':lambda r:r['type'].split(':')[0] if r['type'].split(':')[0] in('SEAL','TAB') else None,
     'site':lambda r:r['site'] if r['site'] in('Mohenjo-daro','Harappa') else 'other',
     'emblem':lambda r:(('unicorn' if r.get('symbol','').startswith('Bull1') else 'other') if r['type'].startswith('SEAL') and r.get('symbol','') not in ('','-') else None)}
    for name,f in facs.items():
        d=[(c,f(r)) for c,r in rows if f(r)]
        for excl in (0,1,2):
            dd=[(c,v) for c,v in d if not(excl and c==740) and not(excl==2 and c in (700,158,154))]
            xs=[c for c,_ in dd]; ys=[v for _,v in dd]; o=mi(xs,ys); ge=0
            for _ in range(5000):
                random.shuffle(ys); ge+=mi(xs,ys)>=o
            print(f'  {name:7s} {["all","non-jar","non-jar,no-700/158/154"][excl]:24s} n={len(dd)} MI={o:.4f} P={(ge+1)/5001:.4f}')
        tab=collections.defaultdict(collections.Counter)
        for c,v in d: tab[c][v]+=1
        print('   ',{c:dict(t) for c,t in tab.items()})
