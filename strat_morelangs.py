"""Strategy 55: more candidate lexicons (Wiktionary via kaikki.org, romanised and reduced):
Telugu, Malayalam (Dravidian), Pali (Middle Indo-Aryan), Santali (Munda).
Calibrated test per language: power (planted L searched with L), Indus, cross (planted English searched with L)."""
import sys
from multiprocessing import Pool
import indus_core as C, run_all as RA, synth
langs=sys.argv[1:] or ['telugu','malayalam','pali','santali']
texts=C.unique_texts(C.load_corpus())
RA.log(f"## Strategy 55: Telugu, Malayalam, Pali, Santali lexicons ({RA.now()})")
with Pool(4) as pool:
    for L in langs:
        lex=C.load_lexicon(L)
        runs=[('power-planted-'+L, synth.planted_corpus(texts,lex,seed=7,noise=0.2)[0]),
              ('indus', texts),
              ('cross-planted-english', synth.planted_corpus(texts,C.load_lexicon('english'),seed=7,noise=0.2)[0])]
        for tag,tx in runs:
            tr,he=C.split_texts(tx); top=[s for s,_ in C.sign_freq(tr).most_common()]
            RA.run_condition(pool,f"s55-{L}-{tag}-full-80",L,'full',80,tr,he,top,20,30,32000,control=True)
