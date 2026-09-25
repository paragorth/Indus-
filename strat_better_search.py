"""Strategy 143: better search for the realistic anchor case (logo-syllabic planted script, word-signs unknown,
20 anchors, generic big Tamil lexicon): 4 independent restarts x 150,000 iterations, keep the best-scoring key."""
import os, json
os.environ['MIXED']='1'; os.environ['BIGLEX']='1'
import strat_logosyll as L
def run(seed):
    import random, math
    k=20; W,prob,true,N,hit=L.W,L.prob,L.true,L.N,L.hit
    rng=random.Random(seed); pool=list(L.allunits)
    for v in true[:k]:
        if v in pool: pool.remove(v)
    rng.shuffle(pool); key=true[:k]+pool[:N-k]; spare=pool[N-k:]; free=list(range(k,N))
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
    iters=150000
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
        if score>best: best,bk=score,key[:]
    return dict(seed=seed,best=best,correct_free=sum(bk[i]==true[i] for i in free),free=N-k)
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(4) as p: res=p.map(run,range(4))
    for r in res: print(json.dumps(r))
    b=max(res,key=lambda r:r['best']); print('best-scoring restart:',json.dumps(b))
