"""Strategy 119: best case for statistics: much more and longer text AND anchors.
Planted Tamil built on texts 3x longer and 5x more numerous (15x the tokens), injective key,
k = 0, 20, 40 anchored signs."""
import random, math, json
import indus_core as C, synth, experiment as X, search as S
base=C.unique_texts(C.load_corpus())
texts=[[s*3 for s in t] for t in base]*5
lex=C.load_lexicon('tamil'); N=80
pt,sec=synth.planted_corpus(texts,lex,seed=7,noise=0.0)
tr,he=C.split_texts(pt); top=[s for s,_ in C.sign_freq(tr).most_common()][:N]
W,_=X.index_texts(tr,top); prob=S.Problem(W,N)
fs=C.forms(lex,'full'); hit=S.string_hit_fn(fs,C.MINLEN['full'])
units=list(C.unit_inventory(lex,'full')); true=[sec[s] for s in top]
def run(k,seed=0,iters=30000):
    rng=random.Random(seed)
    fixedvals=set(true[:k]); pool=[u for u in units if u not in fixedvals]; rng.shuffle(pool)
    key=true[:k]+pool[:N-k]; spare=pool[N-k:]; free=list(range(k,N))
    hits=[hit(tuple(key[x] for x in w)) for w in W]; score=sum(hits); best,bk=score,key[:]
    for it in range(iters):
        T=2.0*(0.02/2.0)**(it/iters)
        if rng.random()<0.5:
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
    return dict(anchors=k,windows=len(W),found=best,true=ts,correct_free=sum(bk[i]==true[i] for i in free),free=len(free))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[0,20,40]): print(json.dumps(r),flush=True)
