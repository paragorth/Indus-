"""Reusable anchor test (Strategy 132). Given proposed anchors {sign: value} and a word list, count corpus
windows (2-4 signs) made only of anchored signs whose reading is a listed word (>=3 letters), against
(a) the values shuffled among the same signs and (b) values drawn from a syllable pool.
Usage: python3 anchor_test.py anchors.json wordlist.txt   (anchor keys = Wells codes, e.g. "012")
calibrate(): power on planted corpora with TRUE anchors (same number of anchors)."""
import sys, json, random, collections
import indus_core as C, synth
def windows(segs):
    for s in segs:
        for L in (2,3,4):
            for i in range(len(s)-L+1): yield s[i:i+L]
def count(key,segs,lex):
    return sum(1 for w in windows(segs) if all(x in key for x in w) and len(r:=''.join(key[x] for x in w))>=3 and r in lex)
def test(key,segs,lex,pool,R=2000,seed=0):
    rng=random.Random(seed); signs=list(key); vals=[key[s] for s in signs]; obs=count(key,segs,lex)
    perm=[];rand=[]
    for _ in range(R):
        v=vals[:]; rng.shuffle(v); perm.append(count(dict(zip(signs,v)),segs,lex))
        rand.append(count(dict(zip(signs,rng.choices(pool,k=len(signs)))),segs,lex))
    return dict(obs=obs,perm_mean=sum(perm)/R,P_perm=sum(p>=obs for p in perm)/R,rand_mean=sum(rand)/R,P_rand=sum(p>=obs for p in rand)/R)
def calibrate(lex,k=20,noise=0.2,R=500):
    texts=C.unique_texts(C.load_corpus())
    pt,sec=synth.planted_corpus(texts,lex,seed=7,noise=noise)
    segs=[s for t in pt for s in t]
    freq=collections.Counter(x for s in segs for x in s)
    top=[s for s,_ in freq.most_common(k)]
    key={s:sec[s] for s in top}
    sylls=[synth.syllabify(w) for w in lex]
    units=[u for u,_ in collections.Counter(u for sy in sylls if sy for u in sy).most_common(60)]
    return test(key,segs,set(lex),units,R)
if __name__=='__main__':
    if sys.argv[1:2]==['calibrate']:
        for L in sys.argv[2:]:
            lex=C.load_lexicon(L) if L in ('tamil','sanskrit') else {w.strip() for w in open(f'data/derived/lex_{L}.txt') if w.strip()}
            print(L,len(lex),calibrate(lex),flush=True)
    else:
        key=json.load(open(sys.argv[1])); key=key.get('key',key)
        lex={w.strip() for w in open(sys.argv[2]) if w.strip()}
        segs=[s for t in C.unique_texts(C.load_corpus()) for s in t]
        pool=sorted({v for v in key.values()})
        print(test(key,segs,lex,pool))
