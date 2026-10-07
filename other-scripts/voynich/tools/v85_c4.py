"""v85 cycle 4: DOES THE WORD STOCK FOLLOW THE SUBJECT OR THE SCRIBE?
Target group (hand h, section s).  Trainers of equal size N (random whole pages, 20 draws):
  SH = same hand, other section;  SS = other hand, same section;  XX = other hand, other section.
For a trainer T: glyph model M_T = order-2 glyph Markov (add-0.1) on T; word model p_T(w) = (c_T(w) + 5 p_M(w)) / (n_T + 5).
  GLYPH score  = mean log2 p_M(target token)                (how alike the glyph moods are)
  LEX gain     = mean [log2 p_T - log2 p_M] over target tokens (what the trainer's actual words add beyond its glyph
                 grammar: the lexicon).
Frozen 7 Oct 2026.  Prediction under 'one dictionary about subjects': LEX(SS) > LEX(SH); under 'private habit':
LEX(SH) > LEX(SS).  Calibration (planted, canon level): MEANING world = hand 1 writes Culpeper part 1 and Cury part 1
(surface profile A), hand 2 writes Culpeper part 2 and Cury part 2 (profile B), one payload code; HABIT world = each
Voynich hand group replaced by a private random-rank dictionary of its HAND (same dictionary in both of its sections).
Usage: VOY_MODE=glyph python3 v85_c4.py [ZL3b|IT2a]"""
import sys, random, math
from collections import Counter
from multiprocessing import Pool
import numpy as np
import v85_lib as L
from v85_c3 import RefM

NAME = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'
R = 20


def take(pages, N, rng):
    w, _ = L.sample(pages, N, rng); return w


def score(train, test):
    M = RefM(train); c = Counter(train); n = len(train)
    g, lx = [], []
    cache = {}
    for w in test:
        if w not in cache:
            lm = M.lp(w); pw = (c[w] + 5 * 2 ** lm) / (n + 5); cache[w] = (lm, math.log2(pw) - lm)
        a, b = cache[w]; g.append(a); lx.append(b)
    return float(np.mean(g)), float(np.mean(lx))


def triad(target, SH, SS, XX, N, seed):
    rng = random.Random(seed); out = {k: [] for k in ('SH', 'SS', 'XX')}
    test = [w for p in target for l in p['lines'] for w in l['w']]
    for _ in range(R):
        for k, G in (('SH', SH), ('SS', SS), ('XX', XX)):
            if G: out[k].append(score(take(G, N, rng), test))
    return {k: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])), float(np.std([b for _, b in v])))
            for k, v in out.items() if v}


def voy_design(name):
    P, G = L.voy_groups(name)
    C = {k: L.canon(v) for k, v in G.items()}
    g = lambda s, h, l='B': C.get((s, h, l), [])
    D = [  # label, target, SH, SS, XX, N
        ('h2 herbal', g('H', '2'), g('B', '2'), g('H', '3') + g('H', '5'), g('S', '3'), 1000),
        ('h3 herbal', g('H', '3'), g('S', '3'), g('H', '2'), g('B', '2'), 1000),
        ('h5 herbal', g('H', '5'), g('T', '5'), g('H', '2'), g('B', '2'), 280),
        ('h2 text-only', g('T', '2'), g('H', '2'), g('T', '5'), g('S', '3'), 280),
        ('h3+h5 herbal', g('H', '3') + g('H', '5'), g('S', '3'), g('H', '2'), g('B', '2'), 1000),
        ('h2 bio', g('B', '2'), g('H', '2'), [], g('S', '3'), 1000),
        ('h1 herbal (A)', g('H', '1', 'A'), g('P', '1', 'A'), g('H', '2'), g('B', '2'), 1000),
        ('h1 pharma (A)', g('P', '1', 'A'), g('H', '1', 'A'), [], g('B', '2'), 1000),
    ]
    return D, G


def meaning_world():
    C = L.ctrl_entries()
    cu, cy = C['CULP'], C['CURY']
    import v72_lib as V
    code = V.payload_code([w for e in cu + cy for w in e], seed=860, mode='merge')
    def mk(E, pre, sec):
        P = L.make_pages(E, pre, nsec=1, cap=12000)
        for p in P: p['sec'] = sec
        P = V.encode_payload(P, code)
        for p in P:
            for l in p['lines']: l.pop('orig', None)
        return P
    h = len(cu) // 2; k = len(cy) // 2
    W = {('H', 1): L.surface_prof(mk(cu[:h], 'a', 'H'), 1, 'A'), ('R', 1): L.surface_prof(mk(cy[:k], 'b', 'R'), 2, 'A'),
         ('H', 2): L.surface_prof(mk(cu[h:], 'c', 'H'), 3, 'B'), ('R', 2): L.surface_prof(mk(cy[k:], 'd', 'R'), 4, 'B')}
    W = {k: L.canon(v) for k, v in W.items()}
    return [('MEANING h1 herb', W[('H', 1)], W[('R', 1)], W[('H', 2)], W[('R', 2)], 1000),
            ('MEANING h2 recipe', W[('R', 2)], W[('H', 2)], W[('R', 1)], W[('H', 1)], 1000)]


def habit_world(G):
    """private random-rank dictionary per HAND, used in all of that hand's sections."""
    C = {k: L.canon(v) for k, v in G.items()}
    hands = {}
    for (s, h, l), P in C.items():
        hands.setdefault(h, []).extend(P)
    dic = {}
    for h, P in hands.items():
        _, d = L.priv_gen(P, 100 + ord(h[0]), 2); random.Random(7).shuffle(d); dic[h] = d
    W = {}
    for (s, h, l), P in C.items():
        W[(s, h, l)], _ = L.priv_gen(P, 200 + len(W), 2, shared_dict=dic[h])
    g = lambda s, h, l='B': W.get((s, h, l), [])
    return [('HABIT h2 herbal', g('H', '2'), g('B', '2'), g('H', '3') + g('H', '5'), g('S', '3'), 1000),
            ('HABIT h3 herbal', g('H', '3'), g('S', '3'), g('H', '2'), g('B', '2'), 1000),
            ('HABIT h2 text-only', g('T', '2'), g('H', '2'), g('T', '5'), g('S', '3'), 280)]


def run(job):
    lab, t, sh, ss, xx, N = job
    return lab, N, triad(t, sh, ss, xx, N, 864)


def fmt(lab, N, d):
    s = '%-18s N%-5d' % (lab, N)
    for k in ('SH', 'SS', 'XX'):
        if k in d: s += ' | %s glyph %.3f lex %+.3f+-%.3f' % (k, d[k][0], d[k][1], d[k][2])
    if 'SH' in d and 'SS' in d: s += ' || LEX SS-SH %+.3f, GLYPH SS-SH %+.3f' % (d['SS'][1] - d['SH'][1], d['SS'][0] - d['SH'][0])
    return s


if __name__ == '__main__':
    D, G = voy_design(NAME)
    jobs = D[:]
    if NAME == 'ZL3b':
        jobs += meaning_world() + habit_world(G)
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=1)
    L.psave('c4_%s.pkl' % NAME, res)
    for lab, N, d in res:
        print(fmt(lab, N, d), flush=True)
