"""Strategy 8b: more cross-language baselines for the Sumerian lexicon (Indus scored +8.02, planted Tamil +5.75).
Planted Sanskrit, English, Akkadian corpora searched with Sumerian; Indus repeated with 2 more split seeds."""
from multiprocessing import Pool
import indus_core as C, run_all as RA, synth
texts=C.unique_texts(C.load_corpus())
RA.log(f"## Strategy 8b: Sumerian cross-language baselines ({RA.now()})")
with Pool(4) as pool:
    runs=[(f'cross-planted-{L}',synth.planted_corpus(texts,C.load_lexicon(L),seed=7,noise=0.2)[0],0) for L in ('sanskrit','english','akkadian')]
    runs+=[(f'indus-seed{s}',texts,s) for s in (1,2)]
    for tag,tx,seed in runs:
        tr,he=C.split_texts(tx,seed=seed); top=[s for s,_ in C.sign_freq(tr).most_common()]
        RA.run_condition(pool,f"s8b-sumerian-{tag}-full-80",'sumerian','full',80,tr,he,top,20,30,32000,control=True)
