"""Strategy 115: how much text would make the key identifiable? Planted Tamil (noise 0) with
(a) the real text shapes, (b) 5x as many texts, (c) texts 3x longer. For each: injective key search
(as S114) vs the true key: does the true key become the optimum, and how many signs are recovered?"""
import random, math, json
import indus_core as C, synth, experiment as X, search as S
def build(variant):
    texts=C.unique_texts(C.load_corpus())
    if variant=='x5': texts=texts*5
    if variant=='long3': texts=[[s*3 for s in t] for t in texts]
    return texts
def run(args):
    variant,seed=args
    lex=C.load_lexicon('tamil'); N=80
    pt,sec=synth.planted_corpus(build(variant),lex,seed=7+seed,noise=0.0)
    tr,he=C.split_texts(pt); top=[s for s,_ in C.sign_freq(tr).most_common()][:N]
    W,_=X.index_texts(tr,top); prob=S.Problem(W,N)
    fs=C.forms(lex,'full'); hit=S.string_hit_fn(fs,C.MINLEN['full'])
    units=list(C.unit_inventory(lex,'full')); rng=random.Random(seed)
    pool=units[:]; rng.shuffle(pool); key=pool[:N]; spare=pool[N:]
    true=[sec.get(s) for s in top]
    ts=sum(hit(tuple(true[x] for x in w)) for w in W)
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
    iters=40000
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5:
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
    return dict(variant=variant,windows=len(W),true=ts,found=best,correct=sum(k==t for k,t in zip(bk,true)))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[('base',0),('x5',0),('long3',0)]): print(json.dumps(r),flush=True)
