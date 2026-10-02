#!/usr/bin/env python3
"""Test (c): word-internal slot structure.
1. Order-rigidity index: for every pair of distinct glyph units (a,b) that occur
   together in a word token, f = share of co-occurrences where a comes first;
   index = co-occurrence-weighted mean of max(f, 1-f). 1.0 = fixed order.
2. Learned slot grammar: one total order of glyph units is learned on the even
   pages / first half (hill-climbing to maximise tokens whose units are in
   non-decreasing slot order); coverage is measured on held-out text
   (tokens and types, and types unseen in training).
Same procedure for real-language letters and for controls (char shuffle,
trigram Markov, table-grille)."""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
import gen
from test_a_language import unitize

def rigidity(words):
    pair = Counter()
    for w in words:
        seen = set()
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                a, b = w[i], w[j]
                if a != b and (a, b, i, j) not in seen:
                    pair[a, b] += 1
    tot = 0; acc = 0
    done = set()
    for (a, b), v in pair.items():
        if (b, a) in done or (a, b) in done: continue
        u = pair.get((b, a), 0); n = v + u
        acc += max(v, u); tot += n; done.add((a, b))
    return acc / tot

def conforms(w, rank):
    r = [rank.get(c, 999) for c in w]
    return all(x <= y for x, y in zip(r, r[1:]))

def learn_order(words, iters=4, seed=0):
    tc = Counter(words)
    units = sorted({c for w in tc for c in w})
    pos = defaultdict(list)
    for w, k in tc.items():
        for i, c in enumerate(w):
            pos[c].append(i / (len(w) - 1) if len(w) > 1 else 0.5)
    order = sorted(units, key=lambda c: sum(pos[c]) / len(pos[c]))
    def score(o):
        rank = {c: i for i, c in enumerate(o)}
        return sum(k for w, k in tc.items() if conforms(w, rank))
    best = score(order)
    for _ in range(iters):
        improved = False
        for i in range(len(order)):
            c = order[i]
            rest = order[:i] + order[i + 1:]
            for j in range(len(rest) + 1):
                if j == i: continue
                cand = rest[:j] + [c] + rest[j:]
                s = score(cand)
                if s > best:
                    best, order, improved = s, cand, True
                    break
        if not improved:
            break
    return order

def coverage(order, words, train_types=None):
    rank = {c: i for i, c in enumerate(order)}
    tc = Counter(words)
    tok = sum(k for w, k in tc.items() if conforms(w, rank)) / sum(tc.values())
    typ = sum(1 for w in tc if conforms(w, rank)) / len(tc)
    r = {'token_cov': tok, 'type_cov': typ}
    if train_types is not None:
        new = [w for w in tc if w not in train_types]
        r['new_type_cov'] = sum(1 for w in new if conforms(w, rank)) / len(new) if new else None
        r['n_new_types'] = len(new)
    return r

def run_one(name, words, out):
    half = len(words) // 2
    tr, te = words[:half], words[half:]
    order = learn_order(tr)
    r = {'rigidity': rigidity(words), 'order': ''.join(order),
         'train': coverage(order, tr), 'test': coverage(order, te, set(tr))}
    out[name] = r
    print(f"{name:34s} rigidity={r['rigidity']:.3f}  train tok={r['train']['token_cov']:.3f}  "
          f"test tok={r['test']['token_cov']:.3f} type={r['test']['type_cov']:.3f} newtype={r['test']['new_type_cov']:.3f}  order={r['order']}")

def run():
    out = {}
    N = 20000
    vz = unitize(load_voynich('ZL3b', ('P',), True))
    vt = unitize(load_voynich('IT2a', ('P',), True))
    # interleave pages so train/test halves both span sections: use alternate lines
    def alt(lines):
        ws_even = [w for i, L in enumerate(lines) if i % 2 == 0 for w in L['words']]
        ws_odd = [w for i, L in enumerate(lines) if i % 2 == 1 for w in L['words']]
        return ws_even + ws_odd
    run_one('Voynich-ZL3b-glyph', alt(vz), out)
    run_one('Voynich-IT2a-glyph', alt(vt), out)
    vze = load_voynich('ZL3b', ('P',), True)
    run_one('Voynich-ZL3b-EVA-letters', alt(vze), out)
    for g in ('char_shuffle', 'char_trigram_markov', 'table_grille'):
        run_one('CTRL-' + g, alt(gen.GENERATORS[g](vz, seed=7)), out)
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        L = load_ref(k, max_words=N, skip_frac=skip)
        run_one(k, alt(L), out)
        if k == 'Latin-Caesar':
            run_one('CTRL-char_shuffle(Latin-Caesar)', alt(gen.char_shuffle(L, seed=7)), out)
    save('test_c_slots', out)

if __name__ == '__main__':
    run()
