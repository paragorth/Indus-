"""v46 cycle 2: kill tests for cycle-1 leads.
(a) DRIFT-PRESERVING NULL: counts rotated (and reflected) along manuscript order inside each section
    (2N-1 replicates); family-wise max-z over all patterns; same for the held-out aggregate.
(b) STRUCTURE PARTIALLED: log count residualised on log tokens, log lines, log paragraphs (ZL paragraph
    starts + circular/radial lines) inside each section; then S1 vs n with the stratified permutation null.
(c) S2 (exact repetition) held-out: top 10 by S2 on a half, S2 excess over null mean on the other half.
(d) SUB-FIGURE TEST, stars: each starred paragraph carries its own star (7 or 8 points); every pattern's
    per-paragraph presence vs 8-point, permutation of star types WITHIN pages (page drift cancels).
    Positive control: planted opa- in 8-point paragraphs at 30 / 50%.
Controls: Hyginus faithful counts under (a)-(c); planted rate word under the rotation null."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import rankdata
import v46_lib as L

log = []
def out(s):
    print(s, flush=True); log.append(s)

def order_units(units, name='ZL3b'):
    recs = json.load(open(os.path.join(L.DATA, 'derived', name + '_lines.json')))
    first = {}
    for i, r in enumerate(recs): first.setdefault(r['folio'], i)
    return sorted(units, key=lambda u: (u['sec'], first.get(u['id'], 1e9)))

def structure(units, name='ZL3b'):
    recs = json.load(open(os.path.join(L.DATA, 'derived', name + '_lines.json')))
    par, lin = Counter(), Counter()
    for r in recs:
        if r['ltype'] in ('P', 'C', 'R'):
            lin[r['folio']] += 1
            par[r['folio']] += 1 if (r['ltype'] != 'P' or r['para_start']) else 0
    return np.array([[max(1, lin[u['id']]), max(1, par[u['id']])] for u in units], float)

def rotations(units):
    """index arrays: n[perm] for every rotation/reflection inside each section (sections combined by
    matching replicate number modulo section size)."""
    secs = defaultdict(list)
    for i, u in enumerate(units): secs[u['sec']].append(i)
    K = max(2 * len(v) - 1 for v in secs.values())
    P = []
    for k in range(1, K + 1):
        perm = np.arange(len(units))
        for s, ii in secs.items():
            ii = np.array(ii); N = len(ii)
            reps = [np.roll(ii, r) for r in range(1, N)] + [np.roll(ii[::-1], r) for r in range(N)]
            perm[ii] = reps[(k - 1) % len(reps)]
        P.append(perm)
    return np.array(P)

def fw_with(C, T, n, secs, P, extra=None):
    if extra is not None:
        Y = np.log(C + 0.5); X0 = np.column_stack([np.log(T), np.log(extra)]); R = np.zeros_like(Y)
        sa = np.array(secs)
        for s in set(secs):
            m = sa == s; X = np.column_stack([np.ones(m.sum()), X0[m]])
            b, *_ = np.linalg.lstsq(X, Y[m], rcond=None); R[m] = Y[m] - X @ b
        Zr = L.zrank_within(R, secs)
    else:
        Zr = L.zrank_within(np.log((C + 0.5) / T[:, None]), secs)
    zn = L.zrank_within(n, secs)[:, 0]
    obs = (Zr * zn[:, None]).mean(0)
    null = np.abs(np.stack([(Zr * zn[p][:, None]).mean(0) for p in P]))
    mu, sd = null.mean(0), null.std(0) + 1e-9
    zo = (np.abs(obs) - mu) / sd; mx = ((null - mu) / sd).max(1)
    j = int(np.argmax(zo))
    return obs, zo, j, (1 + (mx >= zo[j]).sum()) / (1 + len(P)), Zr, zn

def heldout_P(Zr, zn, secs, P, nsplit=10, K=10, seed=0):
    secs = np.array(secs)
    def run(znv, rs):
        sc = []
        for _ in range(nsplit):
            fit = np.zeros(len(znv), bool)
            for s in set(secs):
                ii = np.nonzero(secs == s)[0]; ii = ii[rs.permutation(len(ii))]; fit[ii[:len(ii) // 2]] = True
            a = (Zr[fit] * znv[fit][:, None]).mean(0); top = np.argsort(-np.abs(a))[:K]
            b = (Zr[~fit][:, top] * znv[~fit][:, None]).mean(0); sc.append(np.mean(np.sign(a[top]) * b))
        return np.mean(sc)
    real = run(zn, np.random.default_rng(seed))
    null = np.array([run(zn[p], np.random.default_rng(seed + 1 + k)) for k, p in enumerate(P)])
    return real, null

def part_ab(tag, units, nkey='n'):
    units = order_units(units)
    M, types, tot = L.type_matrix(units)
    pats = L.make_patterns(types, tot, n_random=4000, seed=0)
    C = L.counts_for(M, pats); names = [p[0] for p in pats]
    T = np.array([len(u['toks']) for u in units], float); n = np.array([u[nkey] for u in units], float)
    secs = [u['sec'] for u in units]
    Prot = rotations(units)
    obs, zo, j, pfw, Zr, zn = fw_with(C, T, n, secs, Prot)
    real, null = heldout_P(Zr, zn, secs, Prot)
    out('%s ROTATION null (%d replicates): best %s S1 %.3f z %.1f pFW %.3f | held-out %.3f vs rot-null %.3f +- %.3f, p %.3f' % (
        tag, len(Prot), names[j], obs[j], zo[j], pfw, real, null.mean(), null.std(), (1 + (null >= real).sum()) / (1 + len(null))))
    for nm in ('inf:opa', 'pre:okal'):
        if nm in names:
            k = names.index(nm); out('   %s under rotation: S1 %.3f z %.1f' % (nm, obs[k], zo[k]))
    S = structure(units)
    Pst = L.perms_within(L.strata(units), 300, 1)
    obs2, zo2, j2, pfw2, Zr2, zn2 = fw_with(C, T, n, secs, Pst, extra=S)
    real2, null2 = heldout_P(Zr2, zn2, secs, Pst[:100])
    out('%s STRUCTURE-partialled (tokens, lines, paragraphs), stratified null: best %s S1 %.3f z %.1f pFW %.3f | held-out %.3f vs %.3f +- %.3f, p %.3f' % (
        tag, names[j2], obs2[j2], zo2[j2], pfw2, real2, null2.mean(), null2.std(), (1 + (null2 >= real2).sum()) / (1 + len(null2))))
    # (c) S2 held-out
    secs_a = np.array(secs); rs = np.random.default_rng(3)
    def s2_held(nv, rs):
        vals = []
        for _ in range(10):
            fit = np.zeros(len(nv), bool)
            for s in set(secs):
                ii = np.nonzero(secs_a == s)[0]; ii = ii[rs.permutation(len(ii))]; fit[ii[:len(ii) // 2]] = True
            a = L.s2_vector(C[fit], nv[fit]); top = np.argsort(-a)[:10]
            vals.append(L.s2_vector(C[~fit][:, top], nv[~fit]).mean())
        return np.mean(vals)
    r2 = s2_held(n, np.random.default_rng(3))
    n2 = np.array([s2_held(n[p], np.random.default_rng(4 + k)) for k, p in enumerate(Pst[:100])])
    out('%s S2 exact-repetition held-out: %.4f vs null %.4f +- %.4f, p %.3f' % (tag, r2, n2.mean(), n2.std(), (1 + (n2 >= r2).sum()) / 101))

def part_d():
    from v32_lib import load_q20
    U = [u for u in load_q20() if u['star'] and u['star']['pts'] in (7, 8)]
    def run(U, tag, nperm=1000):
        toks = [[w for l in u['lines'] for w in l] for u in U]
        tot = Counter(w for t in toks for w in t); types = sorted(tot)
        ti = {w: i for i, w in enumerate(types)}
        M = np.zeros((len(U), len(types)), np.float32)
        for r, t in enumerate(toks):
            for w in t: M[r, ti[w]] += 1
        pats = L.make_patterns(types, tot, n_random=3000, seed=0, min_tok=10)
        C = (L.counts_for(M, pats) > 0).astype(float)      # presence per paragraph
        T = np.array([len(t) for t in toks], float)
        y = np.array([u['star']['pts'] == 8 for u in U], float)
        pages = np.array([u['f'] for u in U])
        rng = np.random.default_rng(0)
        idx = {p: np.nonzero(pages == p)[0] for p in set(pages)}
        def stat(yv):
            # difference in presence rate 8 vs 7, after removing page means (within-page contrast)
            d = np.zeros(C.shape[1]); 
            for p, ii in idx.items():
                if yv[ii].min() == yv[ii].max(): continue
                d += C[ii][yv[ii] == 1].sum(0) * (1 - yv[ii].mean()) - C[ii][yv[ii] == 0].sum(0) * yv[ii].mean()
            return d
        obs = stat(y)
        null = []
        for _ in range(nperm):
            yp = y.copy()
            for p, ii in idx.items(): yp[ii] = y[ii][rng.permutation(len(ii))]
            null.append(stat(yp))
        null = np.abs(np.array(null)); mu, sd = null.mean(0), null.std(0) + 1e-9
        zo = (np.abs(obs) - mu) / sd; mx = ((null - mu) / sd).max(1)
        o = np.argsort(-zo)[:5]
        names = [p[0] for p in pats]
        rows = ['%s d %.2f z %.1f pFW %.3f' % (names[j], obs[j], zo[j], (1 + (mx >= zo[j]).sum()) / (1 + nperm)) for j in o]
        k = names.index('inf:opa') if 'inf:opa' in names else None
        extra = ' | inf:opa d %.2f z %.1f' % (obs[k], zo[k]) if k is not None else ''
        out('%s: %d paragraphs (%d with 8-point stars), %d patterns; top: %s%s' % (tag, len(U), int(y.sum()), len(pats), ' ; '.join(rows), extra))
    run(U, 'PARAGRAPH-LEVEL 8- vs 7-point star, within-page permutation')
    rng = np.random.default_rng(9)
    for share in (0.3, 0.5):
        Up = []
        for u in U:
            v = dict(u); v['lines'] = [list(l) for l in u['lines']]
            if u['star']['pts'] == 8 and rng.random() < share:
                l = v['lines'][rng.integers(len(v['lines']))]; l[rng.integers(len(l))] = 'qopchal'
            Up.append(v)
        run(Up, 'PLANT qopchal in %d%% of 8-point paragraphs' % int(share * 100))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'd'):
        part_d()
    if which in ('all', 'ab'):
        V = L.voy_units()
        Ve = [u for u in V if u['sec'] != 'herbal']
        part_ab('EYE-COUNTED SECTIONS (bio, zodiac, pharma, stars)', Ve)
        for s in ('stars', 'bio', 'zodiac'):
            part_ab(s.upper(), [u for u in V if u['sec'] == s])
        part_ab('STARS n = 8-point', [u for u in V if u['sec'] == 'stars'], 'n8')
        part_ab('HERBAL (auto leaves)', [u for u in V if u['sec'] == 'herbal'])
        # positive control under the rotation null: planted rate word
        part_ab('PLANT okchey 0.05/object, eye-counted sections', L.plant(Ve, 'okchey', 0.05, seed=3))
    if which == 'rest':
        V = L.voy_units(); Ve = [u for u in V if u['sec'] != 'herbal']
        part_ab('HERBAL (auto leaves)', [u for u in V if u['sec'] == 'herbal'])
        part_ab('PLANT okchey 0.05/object, eye-counted sections', L.plant(Ve, 'okchey', 0.05, seed=3))
    if which in ('all', 'hyg', 'rest'):
        H = L.hyginus_units()
        part_ab('HYGINUS faithful', [dict(u, n=u['stated']) for u in H])
        part_ab('HYGINUS Ptolemy', H)
    open(os.path.join(L.CK, 'c2_%s.log' % which), 'w').write('\n'.join(log))
