"""Strategy 129: 150-sign planted Tamil syllabary, all signs decoded, right vocabulary, harder search,
with the k commonest signs anchored to their true values (k = 40, 75): how much of the rest is recovered?"""
import os, json, random, math
os.environ.setdefault('KSYLL','150'); os.environ.setdefault('NDEC','150')
import strat_syllsize_n as M
def run(k,seed=0,iters=150000):
    W,prob,hit,true=M.W,M.prob,M.hit,M.true; N=len(true)
    rng=random.Random(seed); fixed=set(true[:k]); pool=[u for u in M.allunits if u not in fixed]; rng.shuffle(pool)
    key=true[:k]+pool[:N-k]; spare=pool[N-k:]; free=list(range(k,N))
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
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
    ts=sum(hit(tuple(true[x] for x in w)) for w in W)
    return dict(anchors=k,free=N-k,found=best,true=ts,correct_free=sum(bk[i]==true[i] for i in free))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(run,[40,75]): print(json.dumps(r),flush=True)
