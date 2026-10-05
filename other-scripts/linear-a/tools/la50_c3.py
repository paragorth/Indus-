#!/usr/bin/env python3
"""la50 cycle 3: place unidentified goods in the farming year; massive random calendar guessing.

(A) Placement. Known goods fixed at their Aegean months, year start fitted on known goods.
    Each unidentified item u gets the month (24 half-month grid) that maximises the concordance
    of the written-order pairs involving u; 1,000 basket bootstraps give the spread.
    Calibration: leave-one-out on Linear B known goods (and on LA known goods): mask a known good,
    predict its month the same way, circular error against the calendar month (random 3.0).
    Planted: an LA unknown given a hidden month, baskets containing it re-ordered by the calendar.
(B) Massive guessing. 5x10^4 random calendars (monthly year starts) over every item with >= 3 basket occurrences,
    scored on a random half of tablets; the top 1 % re-scored on the other half. Null: the same
    pipeline on data whose half-A orders are shuffled. Then: are surviving calendars more
    'Aegean' (known goods) than random ones? 10 splits. Linear B as control.
(C) Season of each tablet under the best alignment (descriptive).
"""
import json
from collections import Counter
from la50_common import *

rng = np.random.default_rng(503)
OUT = os.path.join(CK, 'c3.log'); open(OUT, 'w').close()
GRID = np.arange(24) / 2.0


def fit_start(us, cal):
    kn = sorted({x for _, o in us for x in o if x in cal})
    a, b, w, _ = pairs(us, kn)
    f, s = conc(cal_vec(cal, kn)[None], a, b, w, return_start=True)
    return float(s[0]), float(f[0])


def place(us, cal, u, s0):
    """profile over GRID of concordance of pairs involving u (others at cal months)."""
    kn = [x for x in sorted(cal) if x != u]
    sc = np.zeros(len(GRID)); W = 0.0
    for _, o in us:
        o2 = [x for x in o if x in kn or x == u]
        if u not in o2 or len(o2) < 2: continue
        n = len(o2); w = 1.0 / (n - 1); pu = o2.index(u)
        for p, x in enumerate(o2):
            if x == u: continue
            kx = (cal[x] - s0) % 12; ku = (GRID - s0) % 12
            ok = (ku < kx) if pu < p else (ku > kx)
            sc += w * (ok + 0.5 * (ku == kx)); W += w
    return sc / W if W else None, W


def predict(us, cal, u, s0):
    prof, W = place(us, cal, u, s0)
    if prof is None: return None, None, 0
    best = np.flatnonzero(prof >= prof.max() - 1e-9)
    # circular mean of the best arc
    ang = np.angle(np.exp(2j * np.pi * GRID[best] / 12).mean()) * 12 / (2 * np.pi) % 12
    return float(ang), prof, W


def boot_pred(us, cal, u, s0, nb=1000):
    rel = [x for x in us if u in x[1]]
    out = []
    for k in range(nb):
        smp = [rel[i] for i in rng.integers(0, len(rel), len(rel))]
        m, _, _ = predict(smp, cal, u, s0)
        if m is not None: out.append(m)
    out = np.array(out)
    c = np.angle(np.exp(2j * np.pi * out / 12).mean()) * 12 / (2 * np.pi) % 12
    d = circ_d(out, c)
    return float(c), float(np.quantile(d, 0.8))


def mname(m):
    return MONTHS[int(m) % 12] + ('' if (m % 1) < 0.5 else '+')


def part_A(la, lb):
    log(OUT, '== (A) placement')
    res = {}
    for nm, docs in [('LB', lb), ('LA', la)]:
        us = units(docs, 'basket')
        cal = {k: v for k, v in AEGEAN.items() if any(k in o for _, o in us)}
        s0, f0 = fit_start(us, cal)
        errs = []
        for c in sorted(cal):
            m, prof, W = predict(us, cal, c, s0)
            if m is None: continue
            e = circ_d(m, cal[c]); errs.append(e)
            log(OUT, '%s leave-one-out %s: true %s predicted %s (weight %.1f) error %.1f mo' % (nm, c, mname(cal[c]), mname(m), W, e))
        # random-month baseline: errors of uniform guesses = 3.0; permutation baseline below
        permerr = []
        ks = sorted(cal)
        for r in range(500):
            pv = rng.permutation([cal[k] for k in ks]); cp = dict(zip(ks, pv))
            s1, _ = fit_start(us, cp)
            ee = []
            for c in ks:
                m, _, _ = predict(us, cp, c, s1)
                if m is not None: ee.append(circ_d(m, cp[c]))
            permerr.append(np.mean(ee))
        permerr = np.array(permerr)
        log(OUT, '%s LOO mean error %.2f mo (uniform guess 3.0; permuted calendars mean %.2f, P(perm <= real) %s); start %s'
            % (nm, np.mean(errs), permerr.mean(), fmt((permerr <= np.mean(errs)).mean()), mname(s0)))
        res[nm + '_loo'] = dict(err=float(np.mean(errs)), perm=float(permerr.mean()), P=float((permerr <= np.mean(errs)).mean()))
    # LA unknowns
    us = units(la, 'basket')
    cal = {k: v for k, v in AEGEAN.items() if any(k in o for _, o in us)}
    s0, _ = fit_start(us, cal)
    cnt = Counter(x for _, o in us for x in o)
    unk = [x for x, n in cnt.most_common() if x not in AEGEAN and n >= 3]
    for u in unk:
        m, prof, W = predict(us, cal, u, s0)
        c, spread = boot_pred(us, cal, u, s0)
        log(OUT, 'LA unknown %s (%d baskets): predicted %s, bootstrap centre %s, 80%% within +-%.1f mo; profile max %.2f min %.2f'
            % (u, cnt[u], mname(m), mname(c), spread, prof.max(), prof.min()))
        res['LA_' + u] = dict(month=m, boot=c, spread=spread, n=cnt[u])
    # LB CYP etc as a check (no confident calendar month; reported only)
    usb = units(lb, 'basket'); calb = {k: v for k, v in AEGEAN.items() if any(k in o for _, o in usb)}
    sb, _ = fit_start(usb, calb)
    for u in ('CYP', 'KO', 'AROM', 'PE', 'TU'):
        m, prof, W = predict(usb, calb, u, sb)
        if m is not None:
            log(OUT, 'LB %s: predicted %s (weight %.1f)' % (u, mname(m), W))
    # planted: hidden month for each LA unknown, baskets re-ordered by calendar w.p. 0.8
    perr = []
    for r in range(200):
        u = unk[r % len(unk)]; hm = rng.uniform(0, 12)
        cal2 = dict(cal); cal2[u] = hm
        us2 = []
        for i, o in us:
            if u in o and rng.random() < 0.8:
                inc = [x for x in o if x in cal2]
                srt = iter(sorted(inc, key=lambda x: (cal2[x] - s0) % 12))
                us2.append((i, [next(srt) if x in cal2 else x for x in o]))
            else:
                us2.append((i, o))
        m, _, _ = predict(us2, cal, u, s0)
        perr.append(circ_d(m, hm))
    log(OUT, 'PLANTED placement (200, LA basket structure, 80%% calendar-ordered): mean error %.2f mo, within 2 mo %.2f (uniform 3.0 / 0.33)'
        % (np.mean(perr), np.mean(np.array(perr) <= 2)))
    res['planted'] = dict(err=float(np.mean(perr)), within2=float(np.mean(np.array(perr) <= 2)))
    return res, s0, cal


def aegean_agreement(M, items):
    """for calendars M over items, agreement of the known goods' cyclic order with Aegean:
    concordance of all known pairs under best common rotation (1 = same cyclic order)."""
    kn = [i for i, x in enumerate(items) if x in AEGEAN]
    if len(kn) < 3: return None
    # pseudo 'baskets': Aegean order from each rotation start -> compare cyclic triples
    tri = []
    for i in range(len(kn)):
        for j in range(i + 1, len(kn)):
            for k in range(j + 1, len(kn)):
                tri.append((kn[i], kn[j], kn[k]))
    tri = np.array(tri)

    def orient(x):
        a, b, c = x[..., 0], x[..., 1], x[..., 2]
        return np.sign(np.sin(2 * np.pi * (b - a) / 12) + np.sin(2 * np.pi * (c - b) / 12) + np.sin(2 * np.pi * (a - c) / 12))
    aeg = np.array([AEGEAN.get(x, 0) for x in items])
    oa = orient(aeg[tri])
    om = orient(M[:, tri])
    return (om == oa).mean(1)


MSTARTS = np.arange(12.0)


def part_B(docs, name, n_cal=50000, splits=10, top=0.01):
    us_all = units(docs, 'basket')
    cnt = Counter(x for _, o in us_all for x in o)
    items = sorted(x for x, n in cnt.items() if n >= 3)
    rows = []
    for sp in range(splits):
        ids = sorted({i for i, _ in us_all}); rng.shuffle(ids)
        A = set(ids[: len(ids) // 2])
        uA = [(i, o) for i, o in us_all if i in A]; uB = [(i, o) for i, o in us_all if i not in A]
        uAs = [(i, list(rng.permutation(o))) for i, o in uA]
        aA, bA, wA, _ = pairs(uA, items); aB, bB, wB, _ = pairs(uB, items); aS, bS, wS, _ = pairs(uAs, items)
        M = rng.uniform(0, 12, (n_cal, len(items)))
        fA = np.concatenate([conc(M[i:i + 10000], aA, bA, wA, MSTARTS) for i in range(0, n_cal, 10000)])
        fS = np.concatenate([conc(M[i:i + 10000], aS, bS, wS, MSTARTS) for i in range(0, n_cal, 10000)])
        k = int(n_cal * top)
        sv = np.argsort(-fA)[:k]; svS = np.argsort(-fS)[:k]
        hB = conc(M[sv], aB, bB, wB, MSTARTS).mean(); hS = conc(M[svS], aB, bB, wB, MSTARTS).mean()
        hR = conc(M[:2000], aB, bB, wB, MSTARTS).mean()
        ag = aegean_agreement(M[sv], items); agR = aegean_agreement(M[:20000], items); agS = aegean_agreement(M[svS], items)
        rows.append((hB, hS, hR, ag.mean(), agS.mean(), agR.mean()))
    r = np.array(rows)
    log(OUT, '%s massive guessing (%d splits x %d calendars, top %.1f%%, items %s): held-out fit survivors %.3f vs shuffled-trained survivors %.3f vs random %.3f '
             '(survivors > shuffled in %d/%d splits); Aegean triple agreement survivors %.3f vs shuffled %.3f vs random %.3f (> random in %d/%d)'
        % (name, splits, n_cal, 100 * top, ','.join(items), r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), (r[:, 0] > r[:, 1]).sum(), splits,
           r[:, 3].mean(), r[:, 4].mean(), r[:, 5].mean(), (r[:, 3] > r[:, 5]).sum(), splits))
    return r.tolist()


def part_B_plant(la, reps=10):
    """plant Aegean order in LA baskets (p 0.8) and check survivors become Aegean."""
    out = []
    for k in range(reps):
        D2 = []
        for d in la:
            nb = []
            for b in d['baskets']:
                if rng.random() < 0.8:
                    kn = [x for x in b if x in AEGEAN]
                    it = iter(sorted(kn, key=lambda x: (AEGEAN[x] - 4.0) % 12))
                    nb.append([next(it) if x in AEGEAN else x for x in b])
                else:
                    nb.append(b)
            D2.append(dict(d, baskets=nb))
        out.append(part_B(D2, 'PLANT', n_cal=20000, splits=1)[0])
    return out


def part_C(la, s0, cal):
    log(OUT, '== (C) tablet seasons under the Aegean calendar (circular mean of known goods; descriptive)')
    seasons = Counter(); rows = []
    for d in la:
        kn = [x for x in d['flat'] if x in cal]
        if len(kn) < 2: continue
        z = np.exp(2j * np.pi * cal_vec(cal, kn) / 12).mean()
        m = np.angle(z) * 12 / (2 * np.pi) % 12; R = abs(z)
        sea = ['winter', 'spring', 'summer', 'autumn'][int(((m + 1) % 12) // 3)]
        seasons[sea] += 1; rows.append((d['id'], mname(m), round(R, 2), sea))
    log(OUT, 'season counts: %s; mean resultant length %.2f' % (dict(seasons), np.mean([r[2] for r in rows])))
    log(OUT, 'tablets: ' + '; '.join('%s %s(R %.2f)' % (a, b, c) for a, b, c, _ in rows))
    return rows


if __name__ == '__main__':
    la = load_la(); lb = load_lb(('KN', 'PY'))
    resA, s0, cal = part_A(la, lb)
    log(OUT, '== (B) massive random guessing with held-out halves')
    rB = {'LA': part_B(la, 'LA'), 'LB': part_B(lb, 'LB')}
    rB['PLANT'] = part_B_plant(la)
    pl = np.array(rB['PLANT'])
    log(OUT, 'PLANTED massive guessing (10 reps): held-out survivors %.3f vs shuffled-trained %.3f; Aegean agreement survivors %.3f vs random %.3f (> random in %d/10)'
        % (pl[:, 0].mean(), pl[:, 1].mean(), pl[:, 3].mean(), pl[:, 5].mean(), (pl[:, 3] > pl[:, 5]).sum()))
    rC = part_C(la, s0, cal)
    json.dump({'A': resA, 'B': rB, 'C': rC}, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
