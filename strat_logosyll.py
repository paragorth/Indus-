"""Strategy 135: a logo-syllabic planted script (closer to Indus): the 60 most frequent Tamil words are
written with ONE word-sign each; all other words are spelt with syllable signs. Frame-like: texts are
word streams with Indus lengths. Question: with 40 syllable anchors + right vocabulary, does a
syllable-only key search still recover the rest, and does the presence of word-signs break it?"""
import random, math, json, collections, os
import indus_core as C, synth, experiment as X, search as S
lex=C.load_lexicon('tamil'); rng0=random.Random(7)
allv=[s for s in (synth.syllabify(w) for w in sorted(lex)) if s and len(s)<=3]
freq=collections.Counter(u for s in allv for u in s); top=set(u for u,_ in freq.most_common(150))
vocab=[s for s in allv if all(u in top for u in s)]; rng0.shuffle(vocab); vocab=vocab[:3000]
wts=[1/(r+1) for r in range(len(vocab))]
logos={tuple(vocab[i]):f"L{i:02d}" for i in range(60)}   # the 60 most frequent words get word-signs
units=collections.Counter(u for s in vocab for u in s); code={u:f"S{i:03d}" for i,(u,_) in enumerate(units.most_common())}
sec={v:k for k,v in code.items()}
base=C.unique_texts(C.load_corpus()); texts=[]
for segs in base:
    t=[]
    for seg in segs:
        out=[]
        while len(out)<len(seg):
            w=tuple(rng0.choices(vocab,wts)[0])
            out+= [logos[w]] if w in logos else [code[u] for u in w]
        t.append(out)
    texts.append(t)
tr,he=C.split_texts(texts)
MIXED=os.environ.get('MIXED')=='1'
syl=[s for s,_ in C.sign_freq(tr).most_common() if MIXED or s.startswith('S')]; N=len(syl)
W,_=X.index_texts(tr,syl); prob=S.Problem(W,N)
small=set(''.join(v) for v in vocab); hit=S.string_hit_fn(small,5)
if os.environ.get('BIGLEX')=='1': hit=S.string_hit_fn(C.forms(lex,'full'),5)
true=[sec.get(s,'#') for s in syl]; allunits=sorted(top)+(['#']*70 if MIXED else [])   # word-signs have no syllable value: '#'-values never match
share=sum(1 for t in tr for s in t for x in s if x.startswith('L'))/sum(len(s) for t in tr for s in t)
def run(k,seed=0,iters=150000):
    rng=random.Random(seed); pool=list(allunits)
    for v in true[:k]:
        if v in pool: pool.remove(v)
    rng.shuffle(pool)
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
    return dict(anchors=k,syllable_signs=N,word_sign_token_share=round(share,2),free=N-k,correct_free=sum(bk[i]==true[i] for i in free),found=best,true=sum(hit(tuple(true[x] for x in w)) for w in W))
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[0,20,40]): print(json.dumps(r),flush=True)
