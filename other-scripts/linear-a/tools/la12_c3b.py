#!/usr/bin/env python3
"""LA-12 cycle 3b: word roles from the summary-line joins, against the magnitude-keeping null.

In la12_c3.py the word lists were compared with N1 (numbers shuffled across tablets), which
makes fewer joins overall, so every word looked enriched. Here each word's count among the
summary-line (JS) joins is compared with N3 (each entry redrawn from its own tablet: keeps
each tablet's magnitude and roundness) and also as a SHARE of all JS joins, 200 runs.
Two roles: 'line-label' = the word on a line whose value equals a multi-item sum of another
tablet (candidate total / transfer); 'donor-first-word' = the heading word of the tablet whose
sum is reproduced elsewhere (candidate source ledger). Holm-Bonferroni over all words tested, on a normal-approximation P for the share (empirical
P floors at 1/(n+1)); the empirical count and share P are reported beside it.
Also a planted control: the word TEST-PLANT is given to 10 random lines whose value is then
set to the whole sum of another random tablet; it must come out on top.
"""
import sys, random, math
from multiprocessing import Pool
from la12_common import *
import la12_c2 as C2
from la12_c3 import roles


def js(recs, mode, cval):
    H, D, T = C2.hits(recs, mode, cval, kmax=1)
    return [('JS', T[ti]['rid'], rs[0], T[ti]['label']) for ti, rs, w, k, ds in H]


def _job(a):
    seed, recs, mode, cval = a
    R2 = bootstrap_within(recs, random.Random(seed))
    J = js(R2, mode, cval)
    tl, dw = roles(J, recs)
    return len(J), tl, dw


def plant_word(recs, mode, cval, rng, n=10):
    out = [dict(r, items=[dict(i) for i in r['items']]) for r in recs]
    multi = [i for i, r in enumerate(out) if sum(1 for x in r['items'] if not x['tot']) >= 2]
    lines = [(ri, j) for ri, r in enumerate(out) for j, x in enumerate(r['items']) if not x['tot']]
    for ri, j in rng.sample(lines, n):
        src = rng.choice([m for m in multi if m != ri])
        q = Q()
        for x in out[src]['items']:
            if not x['tot']:
                q = q + x['q']
        tags = {x['tag'] for x in out[src]['items'] if not x['tot']}
        out[ri]['items'][j] = dict(out[ri]['items'][j], q=q, label='TEST-PLANT', tag=tags.pop() if len(tags) == 1 else '*')
    return out


def test(P, recs, mode, cval, nnull, seed0, out, label):
    J = js(recs, mode, cval)
    rtl, rdw = roles(J, recs)
    res = P.map(_job, [(seed0 + i, recs, mode, cval) for i in range(nnull)], chunksize=5)
    out.write('[%s] real JS joins %d; N3 mean %.1f\n' % (label, len(J), sum(r[0] for r in res) / len(res)))
    rows = []
    for role, real, ix in (('line-label', rtl, 1), ('donor-first-word', rdw, 2)):
        for w, c in real.items():
            if c < 3:
                continue
            xs = [r[ix].get(w, 0) for r in res]
            sh = c / len(J)
            shx = [r[ix].get(w, 0) / max(1, r[0]) for r in res]
            p = (1 + sum(x >= c for x in xs)) / (1 + len(xs))
            ps = (1 + sum(x >= sh for x in shx)) / (1 + len(shx))
            mu = sum(shx) / len(shx); sd = (sum((x - mu) ** 2 for x in shx) / len(shx)) ** .5
            sd = max(sd, math.sqrt(max(sum(xs) / len(xs), 0.5)) / max(1, len(J)))   # Poisson floor
            pz = 0.5 * math.erfc((sh - mu) / sd / math.sqrt(2))
            rows.append((pz, role, w, c, sum(xs) / len(xs), p, ps))
    rows.sort()
    m = len(rows)
    for k, (pm, role, w, c, mu, p, ps) in enumerate(rows):
        holm = pm * (m - k)
        if k < 25 or w == 'TEST-PLANT':
            out.write('  %-16s %-14s real %3d  N3 mean %6.2f  P(count) %.3f  P(share) %.3f  Holm(normal) %.4f\n' % (role, w, c, mu, p, ps, min(1, holm)))
    out.write('  words tested %d; min Holm-adjusted P %.3f\n' % (m, min(1, rows[0][0] * m) if rows else 1))
    out.flush()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'V'
    nnull = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out = open(os.path.join(OUT, 'c3b_%s.txt' % mode), 'w')
    R = la_records('HT')
    with Pool(2) as P:
        test(P, R, mode, LA_CVAL, nnull, 20000, out, 'HT real')
        Rp = plant_word(R, mode, LA_CVAL, random.Random(77))
        test(P, Rp, mode, LA_CVAL, nnull // 2, 30000, out, 'HT + 10 planted TEST-PLANT summary lines')


if __name__ == '__main__':
    main()
