"""S311: inside the NAME residue (S310 parse), is there structure? For name segments of 2-4 signs:
(a) positional specialisation: for signs with >= 15 name tokens, share in the first vs last position of the name;
    statistic = mean |first-share - last-share|; control: shuffle signs within each name (1,000x).
(b) how many distinct name 'first elements' and 'last elements' cover 50% of names (small closed set at one end =
    a name-forming element like '-son of' or a deity word); compare with shuffled.
(c) repeated name 'stems' (same first sign + same last sign) across different sites."""
import json,collections,random
P=json.load(open('data/derived/parsed_texts.json'))
EXCL={400,90,820,817,861,595,407,845,235,240,233,231,220,2,60}
names=[]
for o in P:
    seg=[a for a,l in zip(o['seq'],o['slots']) if l=='NAME']
    # contiguous only
    if 2<=len(seg)<=4:
        idx=[k for k,l in enumerate(o['slots']) if l=='NAME']
        if idx==list(range(idx[0],idx[-1]+1)): names.append((o['site'],seg))
print('name segments 2-4 signs:',len(names))
def spec(N):
    f=collections.Counter(); l=collections.Counter(); t=collections.Counter()
    for _,s in N:
        f[s[0]]+=1; l[s[-1]]+=1
        for a in s: t[a]+=1
    keep=[a for a in t if t[a]>=15 and a not in EXCL]
    return sum(abs(f[a]-l[a])/t[a] for a in keep)/len(keep), f, l, keep
obs,f,l,keep=spec(names); rnd=random.Random(21); null=[]
for _ in range(1000):
    null.append(spec([(st,rnd.sample(s,len(s))) for st,s in names])[0])
null.sort(); print(f'(a) positional specialisation {obs:.3f}; shuffled median {null[500]:.3f}, max {null[-1]:.3f}; P={(sum(x>=obs for x in null)+1)/1001:.4f}')
def cover(c,n):
    acc=0
    for k,(a,v) in enumerate(c.most_common(),1):
        acc+=v
        if acc>=n/2: return k
print(f'(b) signs covering 50% of names: first position {cover(f,len(names))}, last position {cover(l,len(names))}')
t=collections.Counter(a for _,s in names for a in s)
print('   most name-initial-biased:',sorted(((f[a]/t[a],a,t[a]) for a in keep),reverse=True)[:8])
print('   most name-final-biased:  ',sorted(((l[a]/t[a],a,t[a]) for a in keep),reverse=True)[:8])
