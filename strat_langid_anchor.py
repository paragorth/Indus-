"""Strategy 137: can anchors identify the LANGUAGE? Planted Tamil logo-syllabic script (S135, word-signs unknown),
20 true syllable anchors. For each candidate lexicon (Tamil, Sanskrit; generic big lists), complete the key with
the true anchors and with the anchor values shuffled among the same signs. Right language: true anchors should
beat shuffled anchors clearly; wrong language: no difference."""
import os, json, random, math
os.environ['MIXED']='1'
import strat_logosyll as L, search as S, indus_core as C
def run(cfg):
    lang,shuffled,seed=cfg
    hit=S.string_hit_fn(C.forms(C.load_lexicon(lang),'full'),5)
    W,prob,true,N,k=L.W,L.prob,L.true,L.N,20
    anch=true[:k][:]
    if shuffled:
        r=random.Random(100+seed); anch=anch[:]; r.shuffle(anch)
    rng=random.Random(seed); pool=list(L.allunits)
    for v in anch:
        if v in pool: pool.remove(v)
    rng.shuffle(pool); key=anch+pool[:N-k]; spare=pool[N-k:]; free=list(range(k,N))
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best=score
    iters=80000
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5 or not spare:
            a,b=rng.sample(free,2); aff=sorted(set(prob.by_sign[a])|set(prob.by_sign[b])); key[a],key[b]=key[b],key[a]; j=None
        else:
            a=rng.choice(free); j=rng.randrange(len(spare)); aff=prob.by_sign[a]; key[a],spare[j]=spare[j],key[a]; b=None
        new=[hit(tuple(key[x] for x in W[w])) for w in aff]; delta=sum(new)-sum(hits[w] for w in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for w,h in zip(aff,new): hits[w]=h
            score+=delta
        else:
            if j is None: key[a],key[b]=key[b],key[a]
            else: key[a],spare[j]=spare[j],key[a]
        best=max(best,score)
    return dict(lexicon=lang,anchors='shuffled' if shuffled else 'true',seed=seed,best=best)
if __name__=='__main__':
    from multiprocessing import Pool
    cfgs=[(l,s,sd) for l in ('tamil','sanskrit') for s in (False,True) for sd in (0,1)]
    with Pool(4) as p:
        for r in p.imap(run,cfgs): print(json.dumps(r),flush=True)
