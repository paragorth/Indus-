"""Strategy 144: anchor-seeded greedy growth instead of annealing. Start from the 20 anchors; repeatedly take the
most frequent unassigned sign and give it the unused value that maximises lexicon hits among windows made only
of assigned signs (windows with unassigned signs are ignored, so wrong early guesses are not rewarded by chance
matches with random values). Same setup as S143 (logo-syllabic, word-signs unknown, generic big lexicon);
also with the right domain vocabulary."""
import os, json, sys
os.environ['MIXED']='1'
mode=sys.argv[1] if len(sys.argv)>1 else 'big'
if mode=='big': os.environ['BIGLEX']='1'
import strat_logosyll as L
W,prob,true,N,hit=L.W,L.prob,L.true,L.N,L.hit
def greedy(k=20):
    key=[None]*N
    for i in range(k): key[i]=true[i]
    pool=list(L.allunits)
    for v in true[:k]:
        if v in pool: pool.remove(v)
    order=list(range(k,N))  # already sorted by frequency (syl list is most_common order)
    for s in order:
        best=None; bestv=None
        cand=set(pool)
        for v in cand:
            key[s]=v; sc=0
            for wi in prob.by_sign[s]:
                w=W[wi]
                if all(key[x] is not None for x in w) and hit(tuple(key[x] for x in w)): sc+=1
            if best is None or sc>best: best,bestv=sc,v
        key[s]=bestv; pool.remove(bestv)
    free=range(k,N)
    return sum(key[i]==true[i] for i in free), N-k
for k in (20,40):
    c,n=greedy(k); print(json.dumps(dict(mode=mode,anchors=k,correct_free=c,free=n)),flush=True)
