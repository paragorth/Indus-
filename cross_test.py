"""Cross-language negative controls (letter-shuffle null, as in the main run).

A corpus with a known language planted in it is searched with the lexicon of a
*different* language.  If the Tamil lexicon scores as high on planted Sanskrit
or planted English as on the Indus corpus, the Indus result shows only that the
texts are language-like and formulaic, not that they are Tamil.
    python cross_test.py            # all pairs below
"""
import sys
from multiprocessing import Pool

import indus_core as C
import run_all as RA
import synth

PAIRS = [("sanskrit", "tamil"), ("english", "tamil"), ("tamil", "sanskrit"), ("tamil", "english")]


def main(noise=0.2):
    assert C.FAKE_KIND == "shuffle"
    RA.log(f"## Cross-language controls ({RA.now()}): planted language X, searched with lexicon Y "
           f"(noise {noise}, letter-shuffle null)")
    texts = C.unique_texts(C.load_corpus())
    with Pool(4) as pool:
        for planted, search in PAIRS:
            pt, _ = synth.planted_corpus(texts, C.load_lexicon(planted), seed=7, noise=noise)
            tr, he = C.split_texts(pt)
            top = [s for s, _ in C.sign_freq(tr).most_common()]
            RA.run_condition(pool, f"cross-planted-{planted}-searched-{search}-full-80", search, "full",
                             80, tr, he, top, 20, 30, 32_000, control=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.2)
