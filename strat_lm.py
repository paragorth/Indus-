"""Strategy 120: substitution-cipher style decoding with a syllable-bigram language model
(the method that solves classical ciphers). Planted Tamil, injective key over the syllables actually
used by the LM, score = sum of log P(next syllable | previous) inside segments. Is the truth the optimum?"""
import random, math, json, collections
import indus_core as C, synth
texts=C.unique_texts(C.load_corpus()); lex=C.load_lexicon('tamil'); N=80
pt,sec=synth.planted_corpus(texts,lex,seed=7,noise=0.0)
tr,he=C.split_texts(pt); top=[s for s,_ in C.sign_freq(tr).most_common()][:N]; idx={s:i for i,s in enumerate(top)}
# LM from the lexicon (word-internal syllable bigrams + word joins), add-k smoothing
big=collections.Counter(); uni=collections.Counter()
for w in lex:
    sy=synth.syllabify(w)
    if not sy: continue
    seq=['#']+sy+['#']
    for a,b in zip(seq,seq[1:]): big[(a,b)]+=1; uni[a]+=1
V=len(uni)+1
def lp(a,b): return math.log((big[(a,b)]+0.1)/(uni[a]+0.1*V))
units=[u for u,_ in uni.most_common() if u!='#'][:400]
segs=[[idx[s] for s in seg] for t in tr for seg in t if len(seg)>=2 and all(s in idx for s in seg)]
by=collections.defaultdict(set)
for k,sg in enumerate(segs):
    for s in sg: by[s].add(k)
def sscore(sg,key):
    v=[key[i] for i in sg]
    # treat sequence as running text: '#'-free transitions, words may cross (planted streams)
    return sum(max(lp(a,b), lp(a,'#')+lp('#',b)) for a,b in zip(v,v[1:]))
true=[sec[s] for s in top]
def run(seed,iters=20000):
    rng=random.Random(seed)
    pool=[u for u in units]; rng.shuffle(pool)
    for t in true:
        if t not in pool: pool.append(t)
    key=pool[:N]; spare=pool[N:]
    sc=[sscore(sg,key) for sg in segs]; score=sum(sc); best,bk=score,key[:]
    for it in range(iters):
        T=20.0*(0.2/20.0)**(it/iters)
        if rng.random()<0.5:
            a,b=rng.sample(range(N),2); aff=sorted(by[a]|by[b]); key[a],key[b]=key[b],key[a]; j=None
        else:
            a=rng.randrange(N); j=rng.randrange(len(spare)); aff=sorted(by[a]); key[a],spare[j]=spare[j],key[a]; b=None
        new=[sscore(segs[k],key) for k in aff]; delta=sum(new)-sum(sc[k] for k in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for k,v in zip(aff,new): sc[k]=v
            score+=delta
        else:
            if j is None: key[a],key[b]=key[b],key[a]
            else: key[a],spare[j]=spare[j],key[a]
        if score>best: best,bk=score,key[:]
    return dict(seed=seed,found=round(best),true=round(sum(sscore(sg,true) for sg in segs)),correct=sum(a==b for a,b in zip(bk,true)),tokens=sum(map(len,segs)))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[0,1,2]): print(json.dumps(r),flush=True)
