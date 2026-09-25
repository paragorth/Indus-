"""Strategy 15: do texts on seals depend on the animal engraved on the seal?
Each seal = (site, animal group, text). Deduplicate identical (animal, text). For each sign x animal,
the statistic is how many seals of that animal contain the sign. Null: shuffle animal labels WITHIN
each site (controls site-level vocabulary). BH-FDR 5% over all sign x animal tests with >=4 expected hits.
If a sign names the animal (or a title bound to it), it should survive; the unicorn is the default."""
import csv, random, collections, re
ANIM=[('unicorn',r'^unicorn|head of unicorn'),('shorthorn',r'short-horned bull'),('zebu',r'humped bull'),
      ('elephant',r'^elephant'),('rhino',r'^rhinoceros'),('tiger',r'tiger'),('buffalo',r'^buffalo'),
      ('goat-antelope',r'goat-antelope'),('ox-antelope',r'ox-antelope'),('hare',r'hare'),('composite',r'fabulous')]
def group(d):
    d=d.lower()
    for g,p in ANIM:
        if re.search(p,d): return g
objs=collections.defaultdict(lambda:{'signs':[],'g':None,'site':None})
for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')):
    if r['object_type']!='seal' or r['fs_category']!='Animals': continue
    g=group(r['fs_description'])
    if not g: continue
    o=objs[r['text_no']]; o['g']=g; o['site']=r['site_code']
    o['signs']+= [x for x in r['signs_clean'].split() if x!='0']
seen=set(); seals=[]
for k,o in objs.items():
    key=(o['g'],' '.join(o['signs']))
    if not o['signs'] or key in seen: continue
    seen.add(key); seals.append((o['site'],o['g'],set(o['signs']),o['signs']))
print('seals',len(seals),collections.Counter(s[1] for s in seals))
signs=collections.Counter(x for s in seals for x in s[2])
cand=[x for x,n in signs.items() if n>=8]
bysite=collections.defaultdict(list)
for i,s in enumerate(seals): bysite[s[0]].append(i)
labels=[s[1] for s in seals]
def counts(lab):
    c=collections.Counter()
    for s,g in zip(seals,lab):
        for x in s[2]: c[(x,g)]+=1
    return c
obs=counts(labels); gn=collections.Counter(labels); N=len(seals)
tests=[(x,g) for x in cand for g in gn if signs[x]*gn[g]/N>=4 or (obs[(x,g)]>=4 and g!='unicorn')]
R=2000; ge=collections.Counter(); rng=random.Random(0)
for _ in range(R):
    lab=labels[:]
    for site,idx in bysite.items():
        sub=[lab[i] for i in idx]; rng.shuffle(sub)
        for i,v in zip(idx,sub): lab[i]=v
    c=counts(lab)
    for t in tests:
        if c[t]>=obs[t]: ge[t]+=1
p={t:(ge[t]+1)/(R+1) for t in tests}
order=sorted(tests,key=lambda t:p[t]); m=len(tests); surv=[]
for k,t in enumerate(order,1):
    if p[t]<=0.05*k/m: surv=order[:k]
print('tests',m,'survivors',len(surv))
for t in order[:25]:
    x,g=t; exp=signs[x]*gn[g]/N
    # position of sign in texts of that animal
    pos=collections.Counter()
    for s in seals:
        if s[1]==g and x in s[2]:
            i=s[3].index(x); L=len(s[3]); pos['I' if i==0 else 'F' if i==L-1 else 'M']+=1
    print(f"{'*' if t in surv else ' '} M{x:>4} {g:13s} obs {obs[t]:3d} exp {exp:5.1f} p {p[t]:.4f} pos(reading order) {dict(pos)}")
