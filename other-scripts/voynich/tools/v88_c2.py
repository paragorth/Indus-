"""v88 cycle 2: massive random guessing over physical reading relations.
Hypothesis = (page relation, row mapping, feature). Physical relations come from the binding
structure; the null bank is thousands of random page pairings (same side types, same
section x language x hand stratum where possible). Score = mean row-pair feature at the mapped
row minus the mean over all rows of the partner page (removes page-level similarity).
Selection on vocabulary half 0, re-test on half 1 (and the reverse)."""
import sys, os, json, math, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v88_lib as G, v65_lib as V

NRAND = int(os.environ.get('NRAND', 1500))
FEATS = ('J', 'S', 'W', 'O')   # junction, glyph mood, shared word types, same first glyph of first word


def line_table(pages):
    keys = sorted(pages); idx = {}; lines = []
    for k in keys:
        idx[k] = []
        for L in pages[k]:
            idx[k].append(len(lines)); lines.append(L)
    return keys, idx, lines


def feature_mats(pages, lines, half):
    n = len(lines)
    alpha = {}
    for L in lines:
        for w in L:
            for g in w:
                alpha.setdefault(g, len(alpha))
    X = np.zeros((n, len(alpha)), np.float32)
    vocab = {}
    rows, cols = [], []
    for i, L in enumerate(lines):
        for w in L:
            if G.wclass(w) != half:
                continue
            for g in w:
                X[i, alpha[g]] += 1
            j = vocab.setdefault(w, len(vocab)); rows.append(i); cols.append(j)
    nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr == 0] = 1
    X /= nr
    S = X @ X.T
    B = np.zeros((n, max(len(vocab), 1)), np.float32)
    B[rows, cols] = 1
    inter = B @ B.T
    sz = B.sum(1)
    W = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    # junction trained on within-line pairs whose left word is in the other half; applied to rows whose last word is in `half`
    Jm = G.Junction(pages, half=1 - half)
    lastg = [L[-1][-1:] for L in lines]; firstg = [L[0][:1] for L in lines]
    ug = sorted(set(lastg) | set(firstg)); gi = {g: i for i, g in enumerate(ug)}
    T = np.array([[Jm.pmi(a, b) for b in ug] for a in ug], np.float32)
    li = np.array([gi[g] for g in lastg]); fi = np.array([gi[g] for g in firstg])
    J = T[li][:, fi]
    mask = np.array([G.wclass(L[-1]) == half for L in lines])
    J[~mask, :] = np.nan
    fg = np.array([gi[L[0][:1]] for L in lines])
    O = (fg[:, None] == fg[None, :]).astype(np.float32)
    mO = np.array([G.wclass(L[0]) == half for L in lines])
    O[~mO, :] = np.nan
    return {'J': J, 'S': S, 'W': W, 'O': O}


MAPS = [('d%+d' % d, d) for d in (-2, -1, 0, 1, 2)] + [('rev', 'rev'), ('prop', 'prop')]


def map_rows(m, nP, nQ):
    out = []
    for k in range(nP):
        if m == 'rev':
            j = nQ - 1 - k
        elif m == 'prop':
            j = int(round(k * (nQ - 1) / max(nP - 1, 1)))
        else:
            j = k + m
        if 0 <= j < nQ:
            out.append((k, j))
    return out


def build_index(pairs, idx):
    """-> per map: (I, Jx) aligned row indices; baseline index lists (I repeated over all rows of Q)."""
    res = {}
    bi, bj = [], []
    for P, Q in pairs:
        for i in idx[P]:
            for j in idx[Q]:
                bi.append(i); bj.append(j)
    for name, m in MAPS:
        I, Jx = [], []
        for P, Q in pairs:
            for k, j in map_rows(m, len(idx[P]), len(idx[Q])):
                I.append(idx[P][k]); Jx.append(idx[Q][j])
        res[name] = (np.array(I, int), np.array(Jx, int))
    return res, (np.array(bi, int), np.array(bj, int))


def score(F, ind):
    res, (bi, bj) = ind
    base = np.nanmean(F[bi, bj]) if len(bi) else np.nan
    return {name: float(np.nanmean(F[I, Jx]) - base) if len(I) else np.nan for name, (I, Jx) in res.items()}


def physical_relations(quires, pages):
    pc = G.pair_classes(quires, pages)
    rel = dict(pc)
    seq = [k for k in G.phys_seq(quires)]
    rel['RV_PREV'] = [(b, a) for a, b in pc['GUT']]           # recto -> previous verso (same as GUTR)
    rel.pop('GUTR', None)
    same = []; two = []
    sides = [k for k in seq if k != G.GAP]
    for s in ('r', 'v'):
        lst = [k for k in seq if k == G.GAP or k[1] == s]
        for a, b in zip(lst, lst[1:]):
            if a != G.GAP and b != G.GAP and a in pages and b in pages:
                same.append((a, b))
        for a, b in zip(lst, lst[2:]):
            if a != G.GAP and b != G.GAP and a in pages and b in pages:
                two.append((a, b))
    rel['NEXT_SAMESIDE'] = same; rel['TWO_SAMESIDE'] = two
    opp = []
    for qn, bifs in quires:
        for B, A, Bl in bifs:
            if A is None or Bl is None:
                continue
            for s in ('r', 'v'):
                if (A, s) in pages and (Bl, s) in pages:
                    opp.append(((A, s), (Bl, s)))
    rel['CONJ_SAMESIDE'] = opp
    return rel


def random_relation(rel_pairs, pages, meta, rng, strat):
    out = []
    for P, Q in rel_pairs:
        m = meta.get(Q, {})
        pool = [k for k in strat[(m.get('sec'), m.get('lang'), m.get('hand'))] if k[1] == Q[1] and k not in (P, Q)]
        if len(pool) < 2:
            pool = [k for k in pages if k[1] == Q[1] and k not in (P, Q)]
        out.append((P, rng.choice(pool)))
    return out


def run(name, pages, meta, quires, seed=0):
    rng = random.Random(seed)
    keys, idx, lines = line_table(pages)
    rel = physical_relations(quires, pages)
    strat = defaultdict(list)
    for k in pages:
        m = meta.get(k, {}); strat[(m.get('sec'), m.get('lang'), m.get('hand'))].append(k)
    out = {'name': name, 'halves': {}}
    for half in (0, 1):
        Fm = feature_mats(pages, lines, half)
        phys = {}
        for rn, pairs in rel.items():
            ind = build_index(pairs, idx)
            phys[rn] = {f: score(Fm[f], ind) for f in FEATS}
        # null bank: for each relation, NRAND random pairings with the same P list and Q side/stratum
        null = {rn: {f: defaultdict(list) for f in FEATS} for rn in rel}
        for rn, pairs in rel.items():
            for t in range(NRAND):
                ind = build_index(random_relation(pairs, pages, meta, rng, strat), idx)
                for f in FEATS:
                    for mname, v in score(Fm[f], ind).items():
                        null[rn][f][mname].append(v)
        z = {}
        for rn in rel:
            for f in FEATS:
                for mname, v in phys[rn][f].items():
                    a = np.array(null[rn][f][mname]); a = a[~np.isnan(a)]
                    z['%s|%s|%s' % (rn, f, mname)] = {'v': v, 'z': float((v - a.mean()) / (a.std() + 1e-12)),
                                                      'p': float((np.sum(a >= v) + 1) / (len(a) + 1))}
        # how often does a RANDOM hypothesis from the bank reach a given z? (max-z calibration)
        maxz = []
        for t in range(200):
            rn = rng.choice(list(rel)); f = rng.choice(FEATS); mname = rng.choice([m for m, _ in MAPS])
            a = np.array(null[rn][f][mname]); a = a[~np.isnan(a)]
            j = rng.randrange(len(a)); b = np.delete(a, j)
            maxz.append(float((a[j] - b.mean()) / (b.std() + 1e-12)))
        out['halves'][half] = {'z': z, 'null_z_sd': float(np.std(maxz)), 'null_z_p99': float(np.percentile(np.abs(maxz), 99))}
        print(name, 'half', half, 'done', flush=True)
    # selection on half 0, re-test on half 1 and vice versa
    sel = {}
    for a, b in ((0, 1), (1, 0)):
        za, zb = out['halves'][a]['z'], out['halves'][b]['z']
        surv = sorted([h for h in za if za[h]['z'] >= 2.5], key=lambda h: -za[h]['z'])
        sel['%d->%d' % (a, b)] = [(h, round(za[h]['z'], 2), round(zb[h]['z'], 2)) for h in surv]
    out['selection'] = sel
    return out


def main():
    quires, _ = V.structure('ZL3b')
    tmpl, meta = G.vpages('ZL3b')
    stream = [w for k in G.phys_seq(quires) if k != G.GAP and k in tmpl for L in tmpl[k] for w in L]
    which = sys.argv[1:] or ['ZL3b', 'IT2a', 'Voy-SPREAD', 'Markov-ZL']
    res = []
    for name in which:
        if name in ('ZL3b', 'IT2a'):
            p, m = G.vpages(name)
        elif name == 'Voy-SPREAD':
            p, m = G.pour_spread(tmpl, stream, 0, quires, encode=False), meta
        elif name == 'Markov-ZL':
            p, m = V.markov_pages(tmpl, meta, 5), meta
        elif name == 'SelfCit-ZL':
            p, m = V.selfcit_pages(tmpl, quires, 5), meta
        r = run(name, p, m, quires)
        res.append(r)
        json.dump(r, open(os.path.join(G.CK, 'c2_%s.json' % name), 'w'), indent=1, default=float)
        print(name, 'selection', json.dumps(r['selection']), flush=True)


if __name__ == '__main__':
    main()
