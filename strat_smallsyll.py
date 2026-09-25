"""Strategy 124: would a SMALL syllabary be identifiable? Planted Tamil restricted to words written
entirely with the 80 commonest syllables (so every token is decoded), clean whole-word texts with
Indus text counts and lengths, injective key, objective = hits on that vocabulary."""
import random, math, json, collections
import indus_core as C, synth, experiment as X, search as S
lex=C.load_lexicon('tamil'); rng0=random.Random(7)
allv=[s for s in (synth.syllabify(w) for w in sorted(lex)) if s and len(s)<=3]
freq=collections.Counter(u for s in allv for u in s); top80={u for u,_ in freq.most_common(80)}
vocab=[s for s in allv if all(u in top80 for u in s)]; rng0.shuffle(vocab); vocab=vocab[:3000]
wts=[1/(r+1) for r in range(len(vocab))]
units=collections.Counter(u for s in vocab for u in s); code={u:f"S{i:03d}" for i,(u,_) in enumerate(units.most_common())}
sec={v:k for k,v in code.items()}
base=C.unique_texts(C.load_corpus()); texts=[]
for segs in base:
    t=[]
    for seg in segs:
        out=[]
        while len(out)<len(seg): out+=rng0.choices(vocab,wts)[0]
        t.append([code[u] for u in out])
    texts.append(t)
N=min(80,len(code))
tr,he=C.split_texts(texts); top=[s for s,_ in C.sign_freq(tr).most_common()][:N]
W,_=X.index_texts(tr,top); prob=S.Problem(W,N)
small=set(''.join(v) for v in vocab); hit=S.string_hit_fn(small,5)
allunits=sorted(top80); true=[sec[s] for s in top]
def run(seed,iters=40000):
    rng=random.Random(seed); pool=allunits[:]; rng.shuffle(pool); key=pool[:N]; spare=pool[N:]
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5 or not spare:
            a,b=rng.sample(range(N),2); aff=sorted(set(prob.by_sign[a])|set(prob.by_sign[b])); key[a],key[b]=key[b],key[a]; j=None
        else:
            a=rng.randrange(N); j=rng.randrange(len(spare)); aff=prob.by_sign[a]; key[a],spare[j]=spare[j],key[a]; b=None
        new=[hit(tuple(key[x] for x in W[w])) for w in aff]; delta=sum(new)-sum(hits[w] for w in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for w,h in zip(aff,new): hits[w]=h
            score+=delta
        else:
            if j is None: key[a],key[b]=key[b],key[a]
            else: key[a],spare[j]=spare[j],key[a]
        if score>best: best,bk=score,key[:]
    return dict(seed=seed,signs=len(code),windows=len(W),found=best,true=sum(hit(tuple(true[x] for x in w)) for w in W),correct=sum(a==b for a,b in zip(bk,true)))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[0,1,2]): print(json.dumps(r),flush=True)
