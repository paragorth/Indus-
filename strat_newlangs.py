"""Strategy 8: Sumerian (candidate) and Akkadian (control) through the calibrated test:
power (planted L searched with L), Indus searched with L, cross baseline (planted Tamil searched with L)."""
import sys
from multiprocessing import Pool
import indus_core as C, run_all as RA, synth
langs=sys.argv[1:] or ['sumerian','akkadian']
texts=C.unique_texts(C.load_corpus())
RA.log(f"## Strategy 8: new candidate languages via CDLI lexicons ({RA.now()})")
with Pool(4) as pool:
    for L in langs:
        lex=C.load_lexicon(L)
        runs=[('power-planted-'+L, synth.planted_corpus(texts,lex,seed=7,noise=0.2)[0]),
              ('indus', texts),
              ('cross-planted-tamil', synth.planted_corpus(texts,C.load_lexicon('tamil'),seed=7,noise=0.2)[0])]
        for tag,tx in runs:
            tr,he=C.split_texts(tx); top=[s for s,_ in C.sign_freq(tr).most_common()]
            RA.run_condition(pool,f"s8-{L}-{tag}-full-80",L,'full',80,tr,he,top,20,30,32000,control=True)
