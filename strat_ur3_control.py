"""Strategy 107: can the calibrated key search decode a REAL formulaic seal corpus in a KNOWN language?
Ur III seal legends (CDLI), each legend = one text, sign readings reduced (lu2 -> lu) and replaced by opaque
codes, searched with the Sumerian lexicon. Cross baseline: the same coded corpus searched with Akkadian.
If the Sumerian search passes and the Akkadian one does not, the method can in principle read a seal corpus
of this size; if not, the Indus null says little about the language."""
import re, random
from multiprocessing import Pool
import indus_core as C, run_all as RA, strat_repcal as S
G=S.groups()
legs=list({tuple(s) for s in G[('seal','Ur III (ca. 2100-2000 BC)')] if 2<=len(s)<=14})
random.Random(0).shuffle(legs); legs=legs[:3200]
red=lambda x: re.sub(r'[^a-z]','',x.lower().replace('sz','s').replace('g~','g'))
signs=sorted({red(x) for t in legs for x in t if red(x)})
code={s:f"U{i:04d}" for i,s in enumerate(signs)}
texts=[[[code[red(x)] for x in t if red(x)]] for t in legs]
texts=[t for t in texts if len(t[0])>=2]
if __name__=='__main__':
    RA.log(f"## Strategy 107: Ur III seal legends as a known-language positive control, {len(texts)} texts ({RA.now()})")
    with Pool(4) as pool:
        tr,he=C.split_texts(texts); top=[s for s,_ in C.sign_freq(tr).most_common()]
        for L in ('sumerian','akkadian'):
            RA.run_condition(pool,f"s107-ur3legends-searched-with-{L}-full-80",L,'full',80,tr,he,top,20,30,32000,control=True)
