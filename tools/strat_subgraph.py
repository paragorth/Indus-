"""S320 (audit idea 2): the substitution graph. Pairs of distinct texts (>= 4 signs, same length) that differ in
exactly ONE position. Which sign pairs substitute for each other? (a) Are substituting pairs more similar in SHAPE
(glyph similarity) than random frequency-matched pairs? High shape similarity = spelling/variant or word family;
low = different words in one slot. (b) Do substitutions stay inside a functional class (numeral<->numeral, fish<->
fish)? (c) Number of such text pairs vs texts shuffled within each text (20x). seq_raw, site+text dedup."""
import json,collections,random,numpy as np
C=json.load(open('data/derived/merged-corpus-canonical.json'))
S=json.load(open('data/derived/glyph_sim_signs.json')); G=np.load('data/derived/glyph_sim.npy'); ix={s:i for i,s in enumerate(S)}
NUM={1,2,3,4,5,16,17,18,31,32,33,34,55,56}; FISH={220,240,235,233,231,226}
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and len(s)>=4 and tuple(s) not in seen: seen.add(tuple(s)); T.append(tuple(s))
def subs(T):
    out=collections.Counter(); byk=collections.defaultdict(list)
    for s in T:
        for i in range(len(s)): byk[(len(s),i,s[:i]+s[i+1:])].append(s[i])
    for k,v in byk.items():
        v=sorted(set(v))
        for a in range(len(v)):
            for b in range(a+1,len(v)): out[(v[a],v[b])]+=1
    return out
O=subs(T); npairs=sum(O.values())
rnd=random.Random(29); null=[sum(subs([tuple(rnd.sample(s,len(s))) for s in T]).values()) for _ in range(20)]
print(f'texts {len(T)}; one-substitution pairs {npairs} ({len(O)} distinct sign pairs); shuffled null median {sorted(null)[10]}, max {max(null)}')
f=collections.Counter(a for s in T for a in s); pool=sorted(f,key=lambda a:f[a]); rk={a:i for i,a in enumerate(pool)}
def sim(a,b): return G[ix[a],ix[b]] if a in ix and b in ix else None
obs=[sim(a,b) for (a,b),n in O.items() for _ in range(n) if sim(a,b) is not None]
rand=[]
for (a,b),n in O.items():
    for _ in range(n):
        a2=rnd.choice(pool[max(0,rk[a]-10):rk[a]+11]); b2=rnd.choice(pool[max(0,rk[b]-10):rk[b]+11])
        if a2!=b2 and sim(a2,b2) is not None: rand.append(sim(a2,b2))
print(f'(a) shape similarity of substituting pairs {np.mean(obs):.3f} vs frequency-matched random {np.mean(rand):.3f}')
cls=lambda a:'num' if a in NUM else 'fish' if a in FISH else 'other'
same=sum(n for (a,b),n in O.items() if cls(a)==cls(b) and cls(a)!='other'); cross=sum(n for (a,b),n in O.items() if {cls(a),cls(b)}=={'num','fish'})
print(f'(b) numeral<->numeral or fish<->fish {same}; numeral<->fish {cross}')
print('top substitution pairs:',O.most_common(15))
