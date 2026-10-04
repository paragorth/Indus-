"""pe34 cycle 3: factorise compound behaviour (simple signs included as cells
(base, none)). Models fitted leave-one-compound-out:
  ADD   f = mu + a_b + b_m                      (ridge ALS)
  SCALE f = mu + a_b * s_m + b_m                (modifier scales and shifts the base)
  CP    f = mu + a_b + b_m + sum_r u_br v_mr w_r (rank-2 interaction, 6 random restarts)
Score: held-out squared error vs base-only (mu + a_b). Null: modifier labels
shuffled among compounds (ADD 300x, SCALE/CP 30x). Then modifier effect
profiles in raw units with sign agreement across bases."""
import json, os, random, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe34_common import tokens, prep, CK, parse, strip_var

EXC = ('X', 'xX')
LAM = 2.0


def cells(P, mods):
    comp = P['comp']
    bases = sorted({b for _, b, _ in comp})
    rows = [(b, '-', b) for b in bases] + [(c, b, m) for (c, b, _), m in zip(comp, mods)]
    Y = np.array([P['mean'][r[0]] for r in rows])
    W = np.array([min(P['n'][r[0]], 10) for r in rows], float)
    return rows, Y, W


def fit(rows, Y, W, mask, model, rng):
    bl = sorted({r[1] if r[1] != '-' else r[0] for r in rows})
    ml = sorted({r[2] for r in rows if r[1] != '-'})
    bi = {b: i for i, b in enumerate(bl)}
    mi = {m: i for i, m in enumerate(ml)}
    B = np.array([bi[r[1] if r[1] != '-' else r[0]] for r in rows])
    M = np.array([mi[r[2]] + 1 if r[1] != '-' else 0 for r in rows])  # 0 = none
    K = Y.shape[1]
    w = W * mask
    mu = (w[:, None] * Y).sum(0) / w.sum()
    a = np.zeros((len(bl), K))
    bm = np.zeros((len(ml) + 1, K))
    s = np.ones((len(ml) + 1, K))
    U = V = Wr = None
    if model == 'CP':
        U = rng.normal(0, .3, (len(bl), 2)); V = rng.normal(0, .3, (len(ml) + 1, 2)); Wr = rng.normal(0, .3, (2, K))
    for it in range(40 if model != 'CP' else 15):
        inter = 0 if model != 'CP' else (U[B][:, :, None] * V[M][:, :, None] * Wr[None]).sum(1)
        sc = s[M] if model == 'SCALE' else 1.0
        R = Y - mu - bm[M] - inter
        num = np.zeros_like(a); den = np.zeros((len(bl), K)) + LAM
        np.add.at(num, B, w[:, None] * R * sc)
        np.add.at(den, B, w[:, None] * (sc ** 2 if model == 'SCALE' else np.ones_like(R)))
        a = num / den
        if model == 'SCALE':
            A = a[B]
            R2 = Y - mu
            for m in range(1, len(ml) + 1):
                idx = np.where((M == m) & (w > 0))[0]
                if not len(idx):
                    continue
                ww = w[idx][:, None]
                x, y = A[idx], R2[idx]
                # ridge: s -> 1, b -> 0 ; solve 2x2 per feature
                sxx = (ww * x * x).sum(0) + LAM; sx = (ww * x).sum(0); sw = ww.sum() + LAM
                sxy = (ww * x * y).sum(0) + LAM * 1.0; sy = (ww * y).sum(0)
                det = sxx * sw - sx * sx
                s[m] = (sxy * sw - sx * sy) / det
                bm[m] = (sxx * sy - sx * sxy) / det
        else:
            R = Y - mu - a[B] - inter
            num = np.zeros_like(bm); den = np.zeros(len(ml) + 1) + LAM
            np.add.at(num, M, w[:, None] * R); np.add.at(den, M, w)
            bm = num / den[:, None]; bm[0] = 0
        if model == 'CP':
            for _ in range(5):
                inter = (U[B][:, :, None] * V[M][:, :, None] * Wr[None]).sum(1)
                E = (Y - mu - a[B] - bm[M] - inter) * w[:, None] / (w.sum() / len(B)) / len(B)
                gU = np.zeros_like(U); gV = np.zeros_like(V)
                for r in range(2):
                    t = (E * Wr[r]).sum(1)
                    np.add.at(gU[:, r], B, t * V[M][:, r]); np.add.at(gV[:, r], M, t * U[B][:, r])
                gW = np.einsum('n,nr,nk->rk', np.ones(len(B)), U[B] * V[M], E)
                U += np.clip(2.0 * (gU - 0.01 * U), -.1, .1); V += np.clip(2.0 * (gV - 0.01 * V), -.1, .1); Wr += np.clip(0.5 * (gW - 0.01 * Wr), -.1, .1)
    def pred(i, with_mod=True):
        p = mu + (a[B[i]] * (s[M[i]] if model == 'SCALE' and with_mod else 1)) + (bm[M[i]] if with_mod else 0)
        if model == 'CP' and with_mod:
            p = p + (U[B[i]] * V[M[i]]) @ Wr
        return p
    loss = ((Y - np.array([pred(i) for i in range(len(rows))])) ** 2 * w[:, None]).sum()
    return pred, loss


def loo(rows, Y, W, model, restarts=1, seed=0):
    rng = np.random.default_rng(seed)
    mb = defaultdict(set)
    for r in rows:
        if r[1] != '-' and r[2] not in EXC:
            mb[r[2]].add(strip_var(r[1]))
    eb, em = [], []
    for i, r in enumerate(rows):
        if r[1] == '-' or r[2] in EXC or len(mb[r[2]] - {strip_var(r[1])}) == 0:
            continue
        mask = np.ones(len(rows)); mask[i] = 0
        best = None
        for _ in range(restarts):
            pr, l = fit(rows, Y, W, mask, model, rng)
            if best is None or l < best[1]:
                best = (pr, l)
        pr = best[0]
        eb.append(((pr(i, False) - Y[i]) ** 2).mean()); em.append(((pr(i) - Y[i]) ** 2).mean())
    return float(np.mean(eb)), float(np.mean(em)), len(em)


T = tokens()
res = {}
QUICK = os.environ.get('QUICK')
for c in (['LINB'] if QUICK else ['PE', 'LINB', 'ARCH']):
    P = prep(T[c], c)
    mods = [m for _, _, m in P['comp']]
    res[c] = {}
    for model, nperm, rs in ((('ADD', 3, 1), ('SCALE', 2, 1), ('CP', 2, 2)) if QUICK else (('ADD', 300, 1), ('SCALE', 30, 1), ('CP', 30, 6))):
        if c == 'ARCH' and model != 'ADD':
            continue
        rows, Y, W = cells(P, mods)
        eb, em, n = loo(rows, Y, W, model, rs)
        gain = eb - em
        rng = random.Random(9)
        null = []
        for k in range((100 if c == 'ARCH' else nperm)):
            keep = [i for i, m in enumerate(mods) if m not in EXC]
            vals = [mods[i] for i in keep]; rng.shuffle(vals)
            sh = mods[:]
            for i, v in zip(keep, vals):
                sh[i] = v
            rows2, Y2, W2 = cells(P, sh)
            b2, m2, _ = loo(rows2, Y2, W2, model, rs, seed=k)
            null.append(b2 - m2)
        null = np.array(null)
        p = float((1 + (null >= gain).sum()) / (1 + len(null)))
        res[c][model] = {'n': n, 'err_base': eb, 'err_model': em, 'gain': gain,
                         'null_gain': [float(null.mean()), float(null.std())], 'p': p}
        print(c, model, res[c][model], flush=True)

# quantity-family transfer: leave-one-modifier-out (does one modifier carry it?)
from pe34_common import perm_test, sub, transfer
res['qty_jack'] = {}
for c in (['LINB'] if QUICK else ['PE', 'ARCH', 'LINB']):
    P = prep(T[c], c)
    cols = [i for i, k in enumerate(P['names']) if k in ('logq', 'num')]
    Q = sub(P, cols)
    full = perm_test(Q, 1000, seed=5, exclude=EXC, stat='dot', disjoint=True)
    out = {'all': [full[0]['n'], full[0]['dot'], full[3]]}
    if c == 'PE':
        mods0 = sorted({m for _, _, m in P['comp']} - set(EXC))
        for m in mods0:
            R = dict(Q)
            R['comp'] = [x for x in Q['comp'] if x[2] != m]
            r = perm_test(R, 500, seed=6, exclude=EXC, stat='dot', disjoint=True)
            if r[0]['n'] != full[0]['n']:
                out['drop ' + m] = [r[0]['n'], r[0]['dot'], r[3]]
    res['qty_jack'][c] = out
    print('QTY', c, out, flush=True)

# modifier profiles in raw units
prof = {}
for c in ['PE', 'LINB']:
    P = prep(T[c], c)
    toks = T[c]
    names = P['names']
    raw = defaultdict(list)
    for r in toks:
        raw[r['s']].append([r['f'].get(k, 0.0) for k in names])
    rm = {s: np.array(v).mean(0) for s, v in raw.items()}
    sd = np.array([[r['f'].get(k, 0.0) for k in names] for r in toks]).std(0) + 1e-9
    by = defaultdict(list)
    for comp, b, m in P['comp']:
        if m not in EXC:
            by[m].append((comp, b))
    out = {}
    for m, lst in by.items():
        if len({strip_var(b) for _, b in lst}) < 3:
            continue
        D = np.array([(rm[cc] - rm[b]) for cc, b in lst])
        z = D.mean(0) / sd
        agree = (np.sign(D) == np.sign(D.mean(0))).mean(0)
        order = np.argsort(-np.abs(z))[:6]
        out[m] = {'bases': [b for _, b in lst], 'n_tokens': [P['n'][cc] for cc, _ in lst],
                  'top': [(names[j], round(float(D.mean(0)[j]), 3), round(float(z[j]), 2), round(float(agree[j]), 2)) for j in order]}
        print('PROFILE', c, m, out[m], flush=True)
    prof[c] = out
res['profiles'] = prof
json.dump(res, open(os.path.join(CK, 'c3q.json' if QUICK else 'c3.json'), 'w'), indent=1)
