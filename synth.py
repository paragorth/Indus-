"""Positive control: a corpus with a known language planted under a known key.

Texts copy the real corpus's segment-length distribution.  Each segment is
filled with a stream of lexicon words split into V / CV / CVC syllables, each
syllable written with its own sign (the secret key), and then cut to length at
an arbitrary point, as real texts would be.  `noise` replaces that fraction of
signs with a random sign, to measure how much signal the pipeline can detect.
"""
import random
from collections import Counter

import indus_core as C


def syllabify(w):
    """(C)V(C) syllables with the usual VC.CV split; None if the word has an
    onset or coda cluster the syllabary cannot write."""
    V = C.VOWELS
    units, i, n = [], 0, len(w)
    while i < n:
        u = ""
        if w[i] not in V:
            u += w[i]; i += 1
        if i >= n or w[i] not in V:
            return None
        u += w[i]; i += 1
        # take a coda if the next consonant is followed by another consonant or the end
        if i < n and w[i] not in V and (i + 1 == n or w[i + 1] not in V):
            u += w[i]; i += 1
        units.append(u)
    return units


def planted_corpus(real_texts, lex, seed=0, noise=0.0, max_syll=3):
    rng = random.Random(seed)
    vocab = [s for s in (syllabify(w) for w in sorted(lex)) if s and len(s) <= max_syll]
    # Zipfian word use, like any real text stream
    rng.shuffle(vocab)
    vocab = vocab[:3000]
    wts = [1 / (r + 1) for r in range(len(vocab))]
    units = Counter(u for s in vocab for u in s)
    code = {u: f"S{i:03d}" for i, (u, _) in enumerate(units.most_common())}
    signs = list(code.values())

    def stream(n):
        out = []
        while len(out) < n + 3:
            out += [code[u] for u in rng.choices(vocab, wts)[0]]
        k = rng.randrange(0, 3)       # start mid-word sometimes
        s = out[k:k + n]
        return [x if rng.random() >= noise else rng.choice(signs) for x in s]

    texts = [[stream(len(seg)) for seg in segs] for segs in real_texts]
    secret = {v: u for u, v in code.items()}
    return texts, secret


def planted_logo_corpus(real_texts, lex, seed=0, noise=0.0, n_roots=300):
    """Logographic positive control: each sign is a Tamil root, and each text
    is a walk along attested compounds (a, b), (b, c), ... of those roots."""
    import experiment as X
    rng = random.Random(seed)
    roots, pairs = X.compound_graph(lex)
    deg = Counter()
    for a, b in pairs:
        deg[a] += 1; deg[b] += 1
    keep = {r for r, _ in deg.most_common(n_roots)}
    pairs = sorted((a, b) for a, b in pairs if a in keep and b in keep)
    nxt = {}
    for a, b in pairs:
        nxt.setdefault(a, []).append(b)
    code = {r: f"L{i:03d}" for i, r in enumerate(sorted(keep))}
    signs = list(code.values())

    def walk(n):
        a, b = rng.choice(pairs)
        out = [a, b]
        while len(out) < n:
            out.append(rng.choice(nxt[out[-1]]) if out[-1] in nxt else rng.choice(pairs)[0])
        return [code[x] if rng.random() >= noise else rng.choice(signs) for x in out[:n]]

    texts = [[walk(len(seg)) for seg in segs] for segs in real_texts]
    return texts, {v: k for k, v in code.items()}
