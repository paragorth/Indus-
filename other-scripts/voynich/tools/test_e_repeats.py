#!/usr/bin/env python3
"""Test (e): adjacent repetition.
For adjacent word pairs inside a line: rate (per 1000 pairs) of identical
pairs, of near-identical pairs (edit distance 1), and of runs of 3 identical.
Expected rate = same text with words shuffled across the whole text (mean of
10 shuffles); report observed/expected. Same for real languages and generators.
Also: do near-identical neighbours differ mostly at the start of the word?"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
import gen
from test_a_language import unitize

def ed1(a, b):
    if a == b or abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b): a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a: return True
    return False

def rates(lines):
    n = same = near = trip = 0; where = Counter()
    for L in lines:
        ws = L['words']
        for i in range(len(ws) - 1):
            a, b = ws[i], ws[i + 1]; n += 1
            if a == b:
                same += 1
                if i + 2 < len(ws) and ws[i + 2] == a: trip += 1
            elif ed1(a, b):
                near += 1
                if len(a) == len(b):
                    k = [j for j in range(len(a)) if a[j] != b[j]][0]
                    where['first' if k == 0 else ('last' if k == len(a) - 1 else 'inner')] += 1
                else:
                    s, l = (a, b) if len(a) < len(b) else (b, a)
                    where['first' if l[1:] == s else ('last' if l[:-1] == s else 'inner')] += 1
    return {'pairs': n, 'same_per1000': 1000 * same / n, 'near_per1000': 1000 * near / n,
            'triples_per1000': 1000 * trip / n, 'near_where': dict(where), 'same_n': same}

def analyse(name, lines, out, reps=10):
    o = rates(lines)
    sh = [rates(gen.word_shuffle(lines, seed=s)) for s in range(reps)]
    for k in ('same_per1000', 'near_per1000', 'triples_per1000'):
        e = sum(x[k] for x in sh) / reps
        o[k.replace('per1000', 'exp')] = e
        o[k.replace('per1000', 'ratio')] = o[k] / e if e > 0 else None
    out[name] = o
    print(f"{name:34s} same={o['same_per1000']:.2f}/1000 (exp {o['same_exp']:.2f}, x{o['same_ratio']:.2f})  "
          f"near={o['near_per1000']:.1f} (exp {o['near_exp']:.1f}, x{o['near_ratio']:.2f})  triples={o['triples_per1000']:.2f}  where={o['near_where']}")

def run():
    out = {}
    vz = unitize(load_voynich('ZL3b', ('P',), True))
    vt = unitize(load_voynich('IT2a', ('P',), True))
    analyse('Voynich-ZL3b-glyph', vz, out)
    analyse('Voynich-IT2a-glyph', vt, out)
    analyse('Voynich-ZL-A', [L for L in vz if L['lang'] == 'A'], out)
    analyse('Voynich-ZL-B', [L for L in vz if L['lang'] == 'B'], out)
    for g in ('char_trigram_markov', 'table_grille', 'self_citation'):
        analyse('CTRL-' + g, gen.GENERATORS[g](vz, seed=7), out)
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        analyse(k, load_ref(k, max_words=35000, skip_frac=skip), out)
    # most common identical pairs in Voynich
    c = Counter()
    for L in load_voynich('ZL3b', ('P',), True):
        ws = L['words']
        for a, b in zip(ws, ws[1:]):
            if a == b: c[a] += 1
    out['Voynich_top_identical_pairs_EVA'] = c.most_common(20)
    print(c.most_common(20))
    save('test_e_repeats', out)

if __name__ == '__main__':
    run()
