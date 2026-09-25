"""Strategy 121: sanity check for S120: does the same LM decoder work on LONG continuous planted text?
75 texts x 200 signs of planted Tamil (15,000 tokens), otherwise identical to S120."""
import sys, json
import strat_lm as L, indus_core as C, synth, collections
texts=[[list(range(200))] for _ in range(75)]
pt,sec=synth.planted_corpus(texts,L.lex,seed=7,noise=0.0)
tr,he=C.split_texts(pt); top=[s for s,_ in C.sign_freq(tr).most_common()][:L.N]; idx={s:i for i,s in enumerate(top)}
# split long texts at unknown signs into runs of known signs
segs=[]
for t in tr:
    for seg in t:
        cur=[]
        for s in seg:
            if s in idx: cur.append(idx[s])
            else:
                if len(cur)>=2: segs.append(cur)
                cur=[]
        if len(cur)>=2: segs.append(cur)
L.segs=segs; L.true=[sec[s] for s in top]
L.by=collections.defaultdict(set)
for k,sg in enumerate(segs):
    for s in sg: L.by[s].add(k)
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(L.run,[0,1,2]): print(json.dumps(r),flush=True)
