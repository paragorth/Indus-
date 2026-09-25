"""Strategy 117: injective key + whole-text coverage objective (planted Tamil, noise 0).
Is the true key the optimum when both safeguards are combined?"""
import random, math, json, collections
import indus_core as C, synth
texts=C.unique_texts(C.load_corpus()); lex=C.load_lexicon('tamil')
pt,sec=synth.planted_corpus(texts,lex,seed=7,noise=0.0)
tr,he=C.split_texts(pt); N=80; top=[s for s,_ in C.sign_freq(tr).most_common()][:N]; idx={s:i for i,s in enumerate(top)}
fs=set(C.forms(lex,'full'))
segs=[[idx[s] for s in seg] for t in tr for seg in t if seg and all(s in idx for s in seg)]
by=collections.defaultdict(list)
for k,sg in enumerate(segs):
    for s in set(sg): by[s].append(k)
def segcov(sg,key):
    vals=[key[i] for i in sg]; n=len(vals); best=[0]*(n+1)
    for i in range(1,n+1):
        best[i]=best[i-1]
        for j in range(max(0,i-8),i):
            w=''.join(vals[j:i])
            if len(w)>=3 and w in fs: best[i]=max(best[i],best[j]+(i-j))
    return best[n]
units=list(C.unit_inventory(lex,'full'))
true=[sec[s] for s in top]
def run(seed,iters=15000):
    rng=random.Random(seed); pool=units[:]; rng.shuffle(pool); key=pool[:N]; spare=pool[N:]
    cov=[segcov(sg,key) for sg in segs]; score=sum(cov); best,bk=score,key[:]
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5:
            a,b=rng.sample(range(N),2); aff=sorted(set(by[a])|set(by[b])); key[a],key[b]=key[b],key[a]; j=None
        else:
            a=rng.randrange(N); j=rng.randrange(len(spare)); aff=by[a]; key[a],spare[j]=spare[j],key[a]; b=None
        new=[segcov(segs[k],key) for k in aff]; delta=sum(new)-sum(cov[k] for k in aff)
        if delta>=0 or rng.random()<math.exp(delta/T):
            for k,v in zip(aff,new): cov[k]=v
            score+=delta
        else:
            if j is None: key[a],key[b]=key[b],key[a]
            else: key[a],spare[j]=spare[j],key[a]
        if score>best: best,bk=score,key[:]
    tot=sum(len(s) for s in segs)
    return dict(seed=seed,found_cov=round(best/tot,3),true_cov=round(sum(segcov(sg,true) for sg in segs)/tot,3),correct=sum(a==b for a,b in zip(bk,true)))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(run,[0,1]): print(json.dumps(r),flush=True)
