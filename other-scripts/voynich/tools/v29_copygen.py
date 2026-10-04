"""v29 kill control: a shape-NEUTRAL copy-and-edit (self-citation) generator.
Each word is, with prob `rate`, a copy of a word from a window of the previous `win` words (same or
previous lines), with Poisson(`edits`) glyph substitutions drawn from the unigram glyph distribution
(no stroke similarity); otherwise a word drawn from the corpus word unigram.  Line lengths are kept."""
import random
import numpy as np
from collections import Counter


def gen(lines, rate=0.5, edits=1.0, win=12, seed=0):
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    words = [w for L in lines for w in L]
    gc = Counter(g for w in words for g in w)
    gk = list(gc); gp = np.cumsum([gc[k] for k in gk]); gp = gp / gp[-1]
    out, hist = [], []
    for L in lines:
        nl = []
        for _ in L:
            if hist and rng.random() < rate:
                w = list(rng.choice(hist[-win:]))
                for _ in range(nrng.poisson(edits)):
                    i = rng.randrange(len(w)); w[i] = gk[int(np.searchsorted(gp, rng.random()))]
                w = tuple(w)
            else:
                w = rng.choice(words)
            nl.append(w); hist.append(w)
        out.append(nl)
    return out
