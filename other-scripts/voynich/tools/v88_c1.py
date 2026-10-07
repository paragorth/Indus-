"""v88 cycle 1: do lines continue across the gutter, through the leaf or across the flat sheet?
Corpora: Voynich ZL3b / IT2a (single-panel sides), planted controls in the same layout
(Isidore XVI and Forme of Cury, opaque padded code, written either ACROSS each opening row by
row or in normal page flow), word-bigram Markov per stratum, self-citation generator.
Statistics per pair class (GUT, GUTR, LEAF, FLAT): row-offset profile of held-out junction PMI
(model trained on the other vocabulary half) and of line glyph-profile cosine; contrast
d=0 minus mean(d=+-1,+-2); z against replacing Q by a random side of the same
section x language x hand stratum (the hand/section-only model). Blind recovery: Hungarian
assignment of versos to rectos by the row-aligned junction contrast.
"""
import sys, os, json, math, random
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v88_lib as G, v65_lib as V

NREP = int(os.environ.get('NREP', 300))


def make_f(pages):
    Jm = [G.Junction(pages, half=h) for h in (0, 1)]
    def fJ(A, B):
        a, b = A[-1], B[0]
        return Jm[1 - G.wclass(a)].pmi(a, b)
    alpha = {}
    for L in pages.values():
        for line in L:
            for w in line:
                for g in w:
                    alpha.setdefault(g, len(alpha))
    cache = {}
    def vec(line):
        key = id(line)
        if key not in cache:
            cache[key] = G.glyph_vec(line, alpha)
        return cache[key]
    def fS(A, B):
        return float(vec(A) @ vec(B))
    # held-out within-line MI
    wl = []
    for L in pages.values():
        for line in L:
            for a, b in zip(line, line[1:]):
                wl.append(Jm[1 - G.wclass(a)].pmi(a, b))
    wrap = []
    for L in pages.values():
        for A, B in zip(L, L[1:]):
            wrap.append(fJ(A, B))
    return fJ, fS, float(np.mean(wl)), float(np.mean(wrap))


def strata(pages, meta):
    s = defaultdict(list)
    for k in pages:
        m = meta.get(k, {})
        s[(m.get('sec'), m.get('lang'), m.get('hand'))].append(k)
    return s


def analyse(name, pages, meta, quires, seed=0):
    rng = random.Random(seed)
    fJ, fS, wl, wrap = make_f(pages)
    pc = G.pair_classes(quires, pages)
    st = strata(pages, meta)
    res = {'name': name, 'within_line': wl, 'wrap': wrap, 'classes': {}}
    for cl, pairs in pc.items():
        r = {'n': len(pairs)}
        for fn, f in (('J', fJ), ('S', fS)):
            prof = G.row_profile(pages, pairs, f)
            T0, D = prof[0], G.contrast(prof)
            nT, nD = [], []
            for _ in range(NREP):
                fake = []
                for P, Q in pairs:
                    m = meta.get(Q, {})
                    pool = [k for k in st[(m.get('sec'), m.get('lang'), m.get('hand'))] if k != Q and k != P and k[1] == Q[1]]
                    if not pool:
                        pool = [k for k in pages if k != Q and k != P and k[1] == Q[1]]
                    fake.append((P, rng.choice(pool)))
                fp = G.row_profile(pages, fake, f)
                nT.append(fp[0]); nD.append(G.contrast(fp))
            r[fn] = {'prof': prof, 'T0': T0, 'D': D,
                     'zT0': (T0 - np.mean(nT)) / (np.std(nT) + 1e-12),
                     'zD': (D - np.mean(nD)) / (np.std(nD) + 1e-12)}
        res['classes'][cl] = r
    # blind recovery of facing pairs by the row-aligned junction contrast
    gut = pc.get('GUT', [])
    vs = [p for p, q in gut]; rs = [q for p, q in gut]
    M = np.zeros((len(vs), len(rs)))
    for i, P in enumerate(vs):
        for j, Q in enumerate(rs):
            prof = G.row_profile(pages, [(P, Q)], fJ, D=(-1, 0, 1))
            M[i, j] = (prof[0] - np.nanmean([prof[-1], prof[1]])) if not math.isnan(prof[0]) else 0
    ri, ci = linear_sum_assignment(-M)
    acc = float(np.mean(ri == ci))
    # rank of the true recto per verso
    ranks = [int((M[i] > M[i, i]).sum()) + 1 for i in range(len(vs))]
    res['blind'] = {'n': len(vs), 'hungarian_acc': acc, 'chance': 1 / max(len(vs), 1),
                    'median_rank': float(np.median(ranks)), 'top1': float(np.mean([r == 1 for r in ranks]))}
    return res


def main():
    out = []
    quires, _ = V.structure('ZL3b')
    tmpl, meta = G.vpages('ZL3b')
    corpora = []
    for name in ('ZL3b', 'IT2a'):
        p, m = G.vpages(name)
        corpora.append((name, p, m))
    isi = V.isidore_words(); cury = G.cury_words()
    corpora.append(('Isidore-SPREAD', G.pour_spread(tmpl, isi, 11, quires), meta))
    corpora.append(('Isidore-FLOW', G.pour_flow(tmpl, isi, 11, quires), meta))
    corpora.append(('Cury-SPREAD', G.pour_spread(tmpl, cury, 12, quires), meta))
    corpora.append(('Cury-FLOW', G.pour_flow(tmpl, cury, 12, quires), meta))
    corpora.append(('Markov-ZL', V.markov_pages(tmpl, meta, 5), meta))
    corpora.append(('SelfCit-ZL', V.selfcit_pages(tmpl, quires, 5), meta))
    for name, p, m in corpora:
        r = analyse(name, p, m, quires)
        out.append(r)
        c = r['classes']
        print(name, 'WL %.3f WRAP %.3f' % (r['within_line'], r['wrap']),
              ' '.join('%s J0 %.3f zT %.1f zD %.1f | S zD %.1f' % (cl, c[cl]['J']['T0'], c[cl]['J']['zT0'], c[cl]['J']['zD'], c[cl]['S']['zD']) for cl in ('GUT', 'GUTR', 'LEAF', 'FLAT')),
              'blind', r['blind'], flush=True)
    json.dump(out, open(os.path.join(G.CK, 'c1.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
