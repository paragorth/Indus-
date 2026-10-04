"""v5 cycle 4b -- order-free transposition: parallel substitution across a word pair.

A melody repeated at another pitch changes every note by the same rule. The order-free trace of that is a
pair of adjacent words (a b) recurring as (a' b') where a -> a' and b -> b' are the SAME single-glyph change
x -> y (e.g. qokedy qokeey ~ qotedy qoteey under k -> t). Count distinct bigram-type pairs related this way
(PS), against the same count after within-line word-order shuffles (20; keeps the word inventory and each
line's words, removes the adjacency). Ratio obs/null and z. Same for chant (syllable = word), verbose
Latin/Italian, trigram-regenerated Voynich.
"""
import sys, os, random
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib


def ps_count(lines):
    big = set()
    for L in lines:
        for a, b in zip(L, L[1:]):
            if a != b:
                big.add((a, b))
    groups = defaultdict(set)
    for a, b in big:
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    groups[(a[:i], a[i + 1:], b[:j], b[j + 1:])].add(x)
    return sum(len(v) * (len(v) - 1) // 2 for v in groups.values()), len(big)


def run(units, R=20, seed=0):
    rng = random.Random(seed)
    lines = [L for u in units for L in u['lines']]
    obs, nb = ps_count(lines)
    nn = [ps_count([rng.sample(L, len(L)) for L in lines])[0] for _ in range(R)]
    return {'bigram_types': nb, 'obs': obs, 'null': float(np.mean(nn)), 'ratio': obs / max(np.mean(nn), 1e-9),
            'z': float((obs - np.mean(nn)) / (np.std(nn) + 1e-12))}


if __name__ == '__main__':
    C = vc.all_corpora()
    res = {}
    for k, u in C.items():
        if k == 'V-ZL-gshuf':
            continue
        r = run(u); res[k] = r
        print(f"{k:22s} bigram types {r['bigram_types']:6d}  PS obs {r['obs']:6d} null {r['null']:8.1f} ratio {r['ratio']:.2f} z {r['z']:+.1f}")
    vlib.save('v5_cycle4b', res)
