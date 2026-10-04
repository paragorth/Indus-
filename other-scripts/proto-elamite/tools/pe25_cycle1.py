"""pe25 cycle 1: is the reverse written by the obverse's hand?

Real PE: hand index H for (obverse, reverse) vs (obverse first half, second half),
stratum nulls (same header base x publication series), tablet bootstrap for
pi = 1 - H_rev / H_half.  Planted controls: variants re-drawn from synthetic
hands (Dirichlet around corpus frequencies, concentration alpha):
  S1 one hand per tablet; T2 reverse by one of 5 checkers; MIX 30% two-hand;
  SEM single hand but 70% of variant choices fixed by the entry (meaning).
Single-hand plants must give pi ~ 0; two-hand ~ 1; mix ~ 0.3.
"""
import sys, random, hashlib, json
import numpy as np
from pe25_common import *

R_NULL = 10
NBOOT = 300
rng = random.Random(25)
d = load()
vb = variant_bases(d)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'loops', 'pe25_cycle1.txt')

keyof = {}
recs = []
for i, t in enumerate(d):
    obv, rev, edge = faces(t)
    o1, o2 = split_halves(obv)
    recs.append(dict(id=t['id'], OBV=tokens(obv, vb), REV=tokens(rev, vb), O1=tokens(o1, vb), O2=tokens(o2, vb)))
    keyof[i] = (header_class(t), series(t))
groups = strata(d, lambda t: (header_class(t), series(t)))
groups_s = strata(d, lambda t: series(t))
keyof_s = {i: series(t) for i, t in enumerate(d)}


def has(r, X):
    return len(r[X]) > 0


def partners(recs, X, Y, rr):
    """for each tablet with X and Y tokens sharing a base: real pair and R_NULL null pairs."""
    elig = [i for i, r in enumerate(recs) if r[X] and r[Y] and cells(r[X], r[Y])]
    nulls = {}
    for i in elig:
        js = []
        for _ in range(R_NULL):
            j = null_partner(i, groups, keyof, rr, need=lambda j: bool(recs[j][Y]))
            if j is None:
                j = null_partner(i, groups_s, keyof_s, rr, need=lambda j: bool(recs[j][Y]))
            if j is not None:
                js.append(j)
        nulls[i] = js
    return elig, nulls


def stat(recs, elig, nulls, X, Y, ctx, sample=None):
    idx = sample if sample is not None else elig
    real = [cells(recs[i][X], recs[i][Y], ctx) for i in idx]
    nul = [cells(recs[i][X], recs[j][Y], ctx) for i in idx for j in nulls[i]]
    a, n = pooled(real)
    a0, _ = pooled(nul)
    return a, a0, (a - a0) / (1 - a0), n


def analyse(recs, label, ctx=False, boot=True, rr=None):
    rr = rr or random.Random(7)
    eA, nA = partners(recs, 'OBV', 'REV', rr)
    eW, nW = partners(recs, 'O1', 'O2', rr)
    A = stat(recs, eA, nA, 'OBV', 'REV', ctx)
    W = stat(recs, eW, nW, 'O1', 'O2', ctx)
    pi = 1 - A[2] / W[2] if W[2] > 0 else float('nan')
    res = dict(label=label, ctx=ctx, nA=len(eA), nW=len(eW), A=A, W=W, pi=pi)
    if boot:
        pis = []
        for _ in range(NBOOT):
            sa = [rr.choice(eA) for _ in eA]
            sw = [rr.choice(eW) for _ in eW]
            a = stat(recs, eA, nA, 'OBV', 'REV', ctx, sa)[2]
            w = stat(recs, eW, nW, 'O1', 'O2', ctx, sw)[2]
            pis.append(1 - a / w if w > 0 else np.nan)
        pis = np.array(pis)
        res['pi_ci'] = [float(np.nanpercentile(pis, 5)), float(np.nanpercentile(pis, 95))]
        res['p_pi_le0'] = float(np.mean(pis <= 0))
    return res


# ---------- planted corpora ----------
def profile(alpha, r):
    out = {}
    for b, f in vb.items():
        vs = list(f)
        g = np.array([r.gammavariate(max(alpha * f[v], 1e-3), 1) for v in vs]) + 1e-12
        out[b] = (vs, g / g.sum())
    return out


def draw(prof, b, r):
    vs, p = prof[b]
    x = r.random()
    c = 0
    for v, q in zip(vs, p):
        c += q
        if x <= c:
            return v
    return vs[-1]


def plant(kind, alpha, seed, mix=0.3, sem=0.7):
    r = random.Random(seed)
    checkers = [profile(alpha, r) for _ in range(5)]
    out = []
    for rec in recs:
        hand = profile(alpha, r)
        second = (kind == 'T2') or (kind == 'MIX' and r.random() < mix)
        chk = r.choice(checkers)
        cache = {}

        def redo(toks, h):
            o = []
            for b, v, k in toks:
                if kind == 'SEM' and r.random() < sem:
                    hsh = int(hashlib.md5((b + k).encode()).hexdigest(), 16) % 10 ** 6 / 1e6
                    vs, _ = profile_freq[b]
                    c = 0
                    nv = vs[-1]
                    for vv in vs:
                        c += vb[b][vv]
                        if hsh <= c:
                            nv = vv
                            break
                    o.append((b, nv, k))
                else:
                    o.append((b, draw(h, b, r), k))
            return o
        # obverse tokens drawn once; halves are its split, so redraw per line identity
        obv_new = redo(rec['OBV'], hand)
        n1 = len(rec['O1'])
        rev_new = redo(rec['REV'], chk if second else hand)
        out.append(dict(id=rec['id'], OBV=obv_new, O1=obv_new[:n1], O2=obv_new[n1:], REV=rev_new))
    return out


profile_freq = {b: (list(f), None) for b, f in vb.items()}

if __name__ == '__main__':
    results = []
    real = analyse(recs, 'REAL')
    realc = analyse(recs, 'REAL', ctx=True)
    results += [real, realc]
    print(json.dumps(real), flush=True)
    print(json.dumps(realc), flush=True)
    # calibrate alpha on single-hand plant: match real H_W
    cal = {}
    for alpha in (0.3, 1, 3, 10):
        s = analyse(plant('S1', alpha, 1), 'S1', boot=False)
        cal[alpha] = s['W'][2]
        print('alpha', alpha, 'H_W', round(s['W'][2], 3), 'pi', round(s['pi'], 3), flush=True)
    alpha = min(cal, key=lambda a: abs(cal[a] - real['W'][2]))
    plants = {}
    for kind in ('S1', 'T2', 'MIX', 'SEM'):
        pis = []
        for seed in range(3):
            s = analyse(plant(kind, alpha, 100 + seed), kind, boot=(seed == 0))
            pis.append(s['pi'])
            if seed == 0:
                results.append(s)
        plants[kind] = pis
        print(kind, [round(p, 3) for p in pis], flush=True)
    dump('cycle1.json', dict(results=results, alpha=alpha, cal=cal, plants=plants, vb=sorted(vb)))
    f = lambda x: '%.3f' % x
    row(OUT, 'PE-25.1a', 'Hand index H = (variant agreement - same-stratum cross-tablet null)/(1-null), obverse vs reverse (n %d tablets) and obverse first half vs second half (n %d); %d variant-bearing bases; pi = 1 - H_rev/H_half, tablet bootstrap' % (real['nA'], real['nW'], len(vb)),
        'H_rev %s (agree %s vs null %s), H_half %s (agree %s vs null %s); pi %s, 90%% CI %s-%s, P(pi<=0) %.2f; context-differing pairs only: H_rev %s, H_half %s, pi %s CI %s-%s' % (
            f(real['A'][2]), f(real['A'][0]), f(real['A'][1]), f(real['W'][2]), f(real['W'][0]), f(real['W'][1]), f(real['pi']), f(real['pi_ci'][0]), f(real['pi_ci'][1]), real['p_pi_le0'],
            f(realc['A'][2]), f(realc['W'][2]), f(realc['pi']), f(realc['pi_ci'][0]), f(realc['pi_ci'][1])), 'see 1b')
    row(OUT, 'PE-25.1b', 'PLANTS (alpha %s chosen so planted H_half matches real; grid %s): S1 one hand per tablet, T2 reverse by 1 of 5 checkers, MIX 30%% two-hand, SEM single hand with 70%% meaning-fixed variants; 3 seeds each' % (alpha, {k: round(v, 3) for k, v in cal.items()}),
        '; '.join('%s pi %s' % (k, [round(p, 2) for p in v]) for k, v in plants.items()),
        'calibration (single-hand ~0, two-hand ~1 required)')
