"""Strategy 31: syllabic key search on the unique middles only (frame stripped, S18/S24/S30).
If names are spelt phonetically and the frame is logographic, whole-text search is swamped by the
frame; searching only middles gives the phonetic hypothesis its best chance.
Per language: power (planted L on middle-shaped texts), Indus middles, cross (planted other language)."""
import sys
from multiprocessing import Pool
import indus_core as C, run_all as RA, synth
OPEN={'817','861','820'}; JAR='740'
def middle(seg):
    s=list(seg)
    if s and s[0] in OPEN: s=s[2:] if len(s)>1 and s[1]=='002' else s[1:]
    if len(s)>=2 and s[-2]==JAR: s=s[:-2]          # post-jar suffix
    elif s and s[-1]==JAR: s=s[:-2]               # X + jar closer
    return s
texts=C.unique_texts(C.load_corpus())
mids=[]
for segs in texts:
    m=[middle(s) for s in segs]; m=[x for x in m if len(x)>=2]
    if m: mids.append(m)
mids=C.unique_texts([(i,m) for i,m in enumerate(mids)])
if __name__=='__main__':
    langs=sys.argv[1:] or ['tamil','sanskrit']
    RA.log(f"## Strategy 31: key search on seal-text middles only, {len(mids)} texts ({RA.now()})")
    with Pool(4) as pool:
        for L in langs:
            other='sanskrit' if L=='tamil' else 'tamil'
            runs=[('power-planted-'+L, synth.planted_corpus(mids,C.load_lexicon(L),seed=7,noise=0.2)[0]),
                  ('indus-middles', mids),
                  ('cross-planted-'+other, synth.planted_corpus(mids,C.load_lexicon(other),seed=7,noise=0.2)[0])]
            for tag,tx in runs:
                tr,he=C.split_texts(tx); top=[s for s,_ in C.sign_freq(tr).most_common()]
                RA.run_condition(pool,f"s31-{L}-{tag}-full-80",L,'full',80,tr,he,top,20,30,32000,control=True)
