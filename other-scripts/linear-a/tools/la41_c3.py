#!/usr/bin/env python3
"""LA-41 cycle 3: copy-made sign substitutions and amount-edit ratios.

(2) Spelling edits between matched entries of near-copy families (cycle 1 robust families, plus
    every pair called by >= 1 scheme as a looser set) give sign substitutions a scribe made while
    copying. Scored against the blind la21 same-row probabilities (LA rows for LA, LB rows for LB)
    and, for LB only, against LB consonants. Null: partners permuted among the substitution list
    (each sign's count kept), 20,000 reps. Planted control: substitutions planted in copies
    (cycle-1 generator) with a same-row table must be recovered.
(3) Amount edits: does one ratio (1/2, 2, 4/5, 5/4, 2/3, 3/2, 1/3, 3, 3/4, 4/3; exact, floor,
    round, ceil) explain >= 2 changed amounts in a family? Null: the second list's amounts redrawn
    from its own site's amount pool, 5,000 reps.
"""
import sys, os, json, pickle, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la41_common import *
import la32_common as L32

OUT = os.path.join(HERE, '..', 'loops', 'la41_cycle3.txt')
RAT = [0.5, 2.0, 0.8, 1.25, 2 / 3, 1.5, 1 / 3, 3.0, 0.75, 4 / 3]


def fam_sets(tag):
    J = json.load(open(os.path.join(CK, f'c1_fam_{tag}.json')))
    rob = [(r[0], r[1]) for r in J['rob']]
    P = pickle.load(open(os.path.join(CK, 'c1.pkl'), 'rb'))
    real = P['real' + tag]; T = J['T']
    loose = set()
    for s, lst in real.items():
        if T.get(s) is None: continue
        for a, b, sc, n, cv in lst:
            if sc >= T[s][0]: loose.add((a, b))
    return rob, sorted(loose)


def subs_of(docs, fams):
    out = []
    for a, b in fams:
        A, B = docs[a]['items'], docs[b]['items']
        for i, j in set_match(A, B):
            e = sign_edit(A[i]['w'], B[j]['w'])
            if e and e[0] == 'sub': out.append((e[2], e[3], a, b, '-'.join(A[i]['w']), '-'.join(B[j]['w'])))
    return out


def score_subs(subs, rows, rng, R=20000):
    signs, M, _ = rows
    ix = {s: k for k, s in enumerate(signs)}
    E = [(x, y) for x, y, *_ in subs if x in ix and y in ix and x != y]
    if not E: return None
    xs = [x for x, _ in E]; ys = [y for _, y in E]
    obs = np.mean([M[ix[x], ix[y]] for x, y in E])
    null = np.empty(R)
    for r in range(R):
        yy = ys[:]; rng.shuffle(yy)
        null[r] = np.mean([M[ix[x], ix[y]] if x != y else np.nan for x, y in zip(xs, yy)]) if True else 0
    null = null[~np.isnan(null)]
    return len(E), float(obs), float(null.mean()), float((1 + (null >= obs).sum()) / (len(null) + 1))


def lbcons(subs, rng, R=20000):
    E = []
    for x, y, *_ in subs:
        cx, cy = L32.lb_cv(x.lower()), L32.lb_cv(y.lower())
        if cx and cy: E.append((cx, cy))
    if not E: return None
    obs = np.mean([a[0] == b[0] for a, b in E])
    ys = [b for _, b in E]; null = []
    for r in range(R):
        rng.shuffle(ys); null.append(np.mean([a[0] == b[0] for (a, _), b in zip(E, ys)]))
    null = np.array(null)
    return len(E), float(obs), float(null.mean()), float((1 + (null >= obs).sum()) / (R + 1))


def ratio_hit(x, y):
    best = 0
    for r in RAT:
        for f in (lambda v: v, math.floor, round, math.ceil):
            hits = sum(1 for a, b in zip(x, y) if a != b and abs(f(a * r) - b) < 1e-6)
            best = max(best, hits)
    return best


def amount_test(docs, fams, rng, R=5000):
    pools = defaultdict(list)
    for d in docs:
        for it in docs[d]['items']:
            if it['val']: pools[docs[d]['site']].append(it['val'])
    units = []
    for a, b in fams:
        A, B = docs[a]['items'], docs[b]['items']
        mp = [(i, j) for i, j in set_match(A, B) if A[i]['val'] and B[j]['val']]
        if len(mp) >= 2: units.append((a, b, [A[i]['val'] for i, _ in mp], [B[j]['val'] for _, j in mp]))
    obs = [ratio_hit(x, y) for _, _, x, y in units]
    k_obs = sum(1 for h in obs if h >= 2)
    null = []
    for r in range(R):
        k = 0
        for a, b, x, y in units:
            yy = [rng.choice(pools[docs[b]['site']]) for _ in y]
            k += ratio_hit(x, yy) >= 2
        null.append(k)
    null = np.array(null)
    det = [(a, b, h, list(zip(x, y))) for (a, b, x, y), h in zip(units, obs) if h >= 2]
    return len(units), k_obs, float(null.mean()), float((1 + (null >= k_obs).sum()) / (R + 1)), det


def planted_subs(rng, rows, reps=10):
    """plant copies whose spelling substitutions follow the blind LA rows (same-row partner)."""
    signs, M, _ = rows
    docs0 = la_docs()
    res = []
    for r in range(reps):
        rr = random.Random(500 + r)
        tab = {}
        for k, s in enumerate(signs[:40]):
            order = np.argsort(-M[k]); cand = [signs[o] for o in order if signs[o] != s][:2]
            tab[s] = rr.choice(cand)
        new, truth = plant_copies(docs0, 25, rr, tab, p_spell=0.4)
        subs = subs_of(new, truth['pairs'])
        res.append(score_subs(subs, rows, rr, R=2000))
    return res


def main():
    rng = random.Random(4141)
    R21 = L32.la21_rows()
    rows = []; out = {}
    for tag in ('LA', 'LB'):
        docs = la_docs() if tag == 'LA' else lb_docs()
        rob, loose = fam_sets(tag)
        for nm, F in (('robust', rob), ('loose', loose)):
            S = subs_of(docs, F)
            out[f'{tag}_{nm}_subs'] = S
            sc = score_subs(S, R21[tag], rng)
            cons = lbcons(S, rng) if tag == 'LB' else None
            listing = '; '.join(f"{x}->{y} ({w1}/{w2}, {a}~{b})" for x, y, a, b, w1, w2 in S[:40])
            rows.append(f"| LA-41.3{'a' if tag == 'LA' else 'b'}-{nm} | {tag} {nm} families ({len(F)} pairs): one-sign substitutions between matched entries; blind la21 {tag} same-row probability vs partner permutation (20,000); {'LB consonant identity vs permutation' if tag == 'LB' else 'no LB values'} | {len(S)} substitutions: {listing}. la21 rows: {'n %d obs %.3f null %.3f P %.4f' % sc if sc else 'none scorable'}{('; LB consonant: n %d obs %.2f null %.2f P %.4f' % cons) if cons else ''}. | see verdict |")
        for nm, F in (('robust', rob), ('loose', loose)):
            n, k, nm_, p, det = amount_test(docs, F, rng)
            d = '; '.join(f"{a}~{b} hits {h}: {pairs}" for a, b, h, pairs in det[:12])
            rows.append(f"| LA-41.3{'c' if tag == 'LA' else 'd'}-{nm} | {tag} {nm}: amount edits follow one fixed ratio (>= 2 changed entries, 10 ratios x 4 roundings) vs second list's amounts redrawn from its site pool (5,000). | {n} families with >= 2 matched amounts; {k} ratio families vs null {nm_:.2f}, P {p:.4f}. {d} | see verdict |")
    pl = planted_subs(rng, R21['LA'])
    pls = [x for x in pl if x]
    rows.append(f"| LA-41.3e | Planted control: 25 copies per rep with spelling substitutions drawn from each sign's two most same-row partners (p_spell 0.4), 10 reps; same scorer. | P per rep: {', '.join('%.3f' % x[3] for x in pls)}; n subs {', '.join(str(x[0]) for x in pls)}; detected (P<0.05) {sum(1 for x in pls if x[3] < 0.05)}/{len(pls)}. | {'PASS' if sum(1 for x in pls if x[3] < 0.05) >= 7 else 'WEAK'} |")
    json.dump({k: v for k, v in out.items()}, open(os.path.join(CK, 'c3.json'), 'w'))
    for r in rows: wlog(OUT, r)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
