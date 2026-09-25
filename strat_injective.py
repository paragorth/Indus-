"""Strategy 114: fix the objective's identifiability failure (S113). The window-hit objective prefers
degenerate keys (true key 1,313 hits vs found 3,514 on planted Tamil). Test an INJECTIVE key search
(each sound value used by at most one sign; swap moves only, values drawn from the lexicon's unit
inventory) on planted Tamil: does the true key then score best, and is it recovered?"""
import random, math, sys, json
import indus_core as C, synth, experiment as X, search as S
def run(seed, lang='tamil', noise=0.0, iters=60000, N=80):
    texts=C.unique_texts(C.load_corpus()); lex=C.load_lexicon(lang)
    pt,sec=synth.planted_corpus(texts,lex,seed=7,noise=noise)
    tr,he=C.split_texts(pt); top=[s for s,_ in C.sign_freq(tr).most_common()][:N]
    W,_=X.index_texts(tr,top); prob=S.Problem(W,N)
    fs=C.forms(lex,'full'); hit=S.string_hit_fn(fs,C.MINLEN['full'])
    units=list(C.unit_inventory(lex,'full'))
    rng=random.Random(seed)
    pool=units[:]; rng.shuffle(pool)
    key=pool[:N]; spare=pool[N:]
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits)
    true=[sec.get(s) for s in top]
    truescore=sum(hit(tuple(true[x] for x in w)) for w in W)
    best,bk=score,key[:]
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5:   # swap two signs' values
            a,b=rng.sample(range(N),2); aff=sorted(set(prob.by_sign[a])|set(prob.by_sign[b]))
            key[a],key[b]=key[b],key[a]
        else:                  # replace a value by an unused one (keeps injectivity)
            a=rng.randrange(N); j=rng.randrange(len(spare)); aff=prob.by_sign[a]
            key[a],spare[j]=spare[j],key[a]; b=None
        new=[hit(tuple(key[x] for x in W[w])) for w in aff]
        delta=sum(new)-sum(hits[w] for w in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for w,h in zip(aff,new): hits[w]=h
            score+=delta
        else:
            if b is not None: key[a],key[b]=key[b],key[a]
            else: key[a],spare[j]=spare[j],key[a]
        if score>best: best,bk=score,key[:]
    correct=sum(1 for k,t in zip(bk,true) if k==t)
    return dict(seed=seed,found=best,true=truescore,correct=correct,N=N)
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(4) as p:
        for r in p.map(run,range(4)): print(json.dumps(r),flush=True)
