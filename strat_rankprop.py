"""Strategy 145: cipher-solver trick for the anchored search: initialise each sign with the syllable of similar
frequency rank and propose replacements only among syllables of nearby rank (±25). Logo-syllabic planted script,
word-signs unknown, 20 anchors, generic big lexicon (as S136/S143)."""
import os, json, random, math
os.environ['MIXED']='1'; os.environ['BIGLEX']='1'
import strat_logosyll as L
W,prob,true,N,hit=L.W,L.prob,L.true,L.N,L.hit
ranked=[u for u,_ in L.freq.most_common() if u in L.top]+['#']*70
def run(seed,k=20,iters=150000):
    rng=random.Random(seed)
    used=set(true[:k]); avail=[u for u in ranked if u not in used or u=='#']
    key=true[:k]+[None]*(N-k)
    pool=avail[:]
    for i in range(k,N):
        # nearest-rank unused value
        key[i]=pool.pop(0) if pool else '#'
    free=list(range(k,N))
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        a=rng.choice(free)
        if rng.random()<0.7:
            # swap with a sign of nearby frequency rank
            b=min(N-1,max(k,a+rng.randint(-25,25)))
            if b==a: continue
            aff=sorted(set(prob.by_sign[a])|set(prob.by_sign[b])); key[a],key[b]=key[b],key[a]; j=None
        else:
            if not pool: continue
            j=rng.randrange(min(len(pool),40)); aff=prob.by_sign[a]; key[a],pool[j]=pool[j],key[a]; b=None
        new=[hit(tuple(key[x] for x in W[w])) for w in aff]; delta=sum(new)-sum(hits[w] for w in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for w,h in zip(aff,new): hits[w]=h
            score+=delta
        else:
            if j is None: key[a],key[b]=key[b],key[a]
            else: key[a],pool[j]=pool[j],key[a]
        if score>best: best,bk=score,key[:]
    return dict(seed=seed,best=best,correct_free=sum(bk[i]==true[i] for i in free),free=N-k)
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(4) as p:
        for r in p.map(run,range(4)): print(json.dumps(r))
