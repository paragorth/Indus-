"""Strategy 134: realistic anchors at Indus scale (~400-sign planted syllabary, all decoded).
(a) 60 correct anchors but a generic big lexicon (all 62,868 Tamil forms);
(b) 60 anchors of which 12 are wrong (values rotated among them), right vocabulary."""
import os, json, random, math
os.environ['KSYLL']='400'; os.environ['NDEC']='400'
import strat_syllsize_n as M, search as S, indus_core as C
def run(cfg):
    mode=cfg; W,prob,true=M.W,M.prob,M.true; N=len(true); k=60
    hit=M.hit
    if mode=='biglex': hit=S.string_hit_fn(C.forms(C.load_lexicon('tamil'),'full'),5)
    anch=true[:k]
    if mode=='noisy':
        rng0=random.Random(5); bad=rng0.sample(range(k),12); vals=[anch[i] for i in bad]; vals=vals[1:]+vals[:1]
        anch=anch[:]; 
        for i,v in zip(bad,vals): anch[i]=v
    rng=random.Random(0); fixed=set(anch); pool=[u for u in M.allunits if u not in fixed]; rng.shuffle(pool)
    key=anch+pool[:N-k]; spare=pool[N-k:]; free=list(range(k,N))
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
    return dict(mode=mode,free=N-k,correct_free=sum(bk[i]==true[i] for i in free),found=best,true=sum(hit(tuple(true[x] for x in w)) for w in W))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(run,['biglex','noisy']): print(json.dumps(r),flush=True)
