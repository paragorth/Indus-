#!/usr/bin/env python3
"""LA-9 approximate Bayesian computation: simulate scribes adding the real entries with random
error mechanisms and random hidden letter values, write the total with the letters (greedy,
largest first, <= 4 signs), and compare value-free summaries of the written totals with the
observed ones. Millions of prior draws; rejection ABC on a scaled Euclidean distance.

Prior: letter values uniform on GRID (49 values), mechanism weights Dirichlet(1) over
MECH_SIM, lacuna sign q+ ~ U(0,1).
"""
import os, json, time
import numpy as np
from la9_common import U, GRID, GRID_LABEL, OUT
from la9_engine import prep

MECH_SIM = ['exact', 'skip', 'double', 'carry', 'slip', 'tally', 'round', 'misread', 'lacuna']
NMS = len(MECH_SIM)
NSUM = 10
SUM_NAMES = ['Dint=0', '|Dint|=1', '|Dint|=10^k', 'Dint=+-entry', 'Dint<0', '2<=|Dint|<=9 other',
             'big gap', 'frac-sec: total no letters', 'frac-sec: total letters within entry letters',
             'int-sec: total has letters']


def setup(P, L):
    for s in P:
        if 'carries' not in s: prep(s, L)
    pool_a = np.concatenate([s['a'] for s in P]); pool_C = np.vstack([s['C'] for s in P])
    return pool_a, pool_C


def summaries(P, tint, tcnt):
    """tint: (S, B) written integer parts; tcnt: (S, B, L) written letter counts."""
    S, B = tint.shape
    out = np.zeros((B, NSUM))
    nfrac = 0; nint = 0
    for k, s in enumerate(P):
        A = s['asum']; Dint = tint[k] - A
        ad = np.abs(Dint)
        out[:, 0] += Dint == 0
        out[:, 1] += ad == 1
        out[:, 2] += np.isin(ad, [10, 100, 1000, 10000])
        ent = np.isin(ad, s['a'][s['a'] > 0]) & (Dint != 0)
        out[:, 3] += ent
        out[:, 4] += Dint < 0
        out[:, 5] += (ad >= 2) & (ad <= 9) & ~ent
        out[:, 6] += (ad >= 0.25 * np.maximum(A, tint[k])) & (Dint != 0) & (ad > 9)
        uni = s['C'].sum(0)
        if uni.sum() > 0:
            nfrac += 1
            has = tcnt[k].sum(1) > 0
            out[:, 7] += ~has
            out[:, 8] += has & np.all(tcnt[k] <= uni[None, :], 1)
        else:
            nint += 1
            out[:, 9] += tcnt[k].sum(1) > 0
    out[:, :7] /= len(P)
    if nfrac: out[:, 7:9] /= nfrac
    if nint: out[:, 9] /= nint
    return out


def observed_summary(P, L):
    tint = np.array([[s['ta']] for s in P]); tcnt = np.array([[s['ct']] for s in P])
    return summaries(P, tint, tcnt)[0]


def write_total(T, V, maxsign=4):
    """Greedy written form: integer part and up to maxsign letters, largest first."""
    tint = T // U; r = T - tint * U
    B, L = V.shape
    cnt = np.zeros((B, L), dtype=np.int64)
    for _ in range(maxsign):
        ok = (V <= r[:, None]) & (r[:, None] > 0)
        cand = np.where(ok, V, -1)
        j = cand.argmax(1); has = cand.max(1) > 0
        cnt[np.arange(B)[has], j[has]] += 1
        r = r - np.where(has, V[np.arange(B), j], 0)
    return tint, cnt


def simulate(P, L, pool, vidx, p, qplus, rng, raw=False):
    pool_a, pool_C = pool
    B = vidx.shape[0]
    V = GRID[vidx]
    S_ = len(P)
    tint = np.zeros((S_, B), dtype=np.int64); tcnt = np.zeros((S_, B, L), dtype=np.int64)
    cum = p.cumsum(1)
    for k, s in enumerate(P):
        a = s['a']; n = len(a)
        e = a[:, None] * U + s['C'] @ V.T                      # (n, B)
        S = e.sum(0)
        m = (rng.random(B)[:, None] > cum).sum(1)
        m = np.minimum(m, NMS - 1)
        T = S.copy()
        ii = rng.integers(0, n, B); ei = e[ii, np.arange(B)]
        T = np.where(m == 1, S - ei, T)
        T = np.where(m == 2, S + ei, T)
        # carry
        c0 = (S - s['asum'] * U) // U
        opts = [(c * pl * U) for c, pl in s['carries']]
        nopt = len(opts) + (c0 > 0)
        pick = (rng.random(B) * np.maximum(nopt, 1)).astype(int)
        drop = np.zeros(B, dtype=np.int64)
        for oi, val in enumerate(opts): drop = np.where(pick == oi, val, drop)
        drop = np.where((c0 > 0) & (pick == len(opts)), c0 * U, drop)
        T = np.where(m == 3, S - drop, T)
        # slip
        mag = np.maximum(S // U, 1)
        okp = np.array([(pl <= mag) for pl in s['places']])       # (K, B)
        K = okp.sum(0)
        kk = (rng.random(B) * K).astype(int)
        plv = s['places'][kk] * U
        sg = np.where(rng.random(B) < 0.5, 1, -1)
        T = np.where(m == 4, S + sg * plv, T)
        # tally
        j = rng.geometric(0.5, B)
        T = np.where(m == 5, S + sg * j * U, T)
        # round
        fl = (S // U) * U
        rr = rng.integers(0, 4, B)
        rv = np.select([rr == 0, rr == 1, rr == 2], [fl, fl + U, fl + U * ((S - fl) * 2 >= U)], s['asum'] * U)
        T = np.where((m == 6) & (S % U != 0), rv, T)
        # misread
        if s['occ']:
            w = np.array([o[2] for o in s['occ']], float); w /= w.sum()
            oi = rng.choice(len(s['occ']), B, p=w)
            ls = np.array([o[1] for o in s['occ']])[oi]
            l2 = rng.integers(0, L, B); l2 = np.where(l2 >= ls, l2 + 1, l2)    # 0..L, skip own; L = none
            alt = np.where(l2 < L, V[np.arange(B), np.minimum(l2, L - 1)], 0)
            T = np.where(m == 7, S - V[np.arange(B), ls] + alt, T)
        # lacuna / wrong cut: +- sum of 1..5 random entries from the corpus pool
        nl = np.minimum(rng.geometric(0.5, B), 5)
        X = np.zeros(B, dtype=np.int64)
        for t in range(5):
            idx = rng.integers(0, len(pool_a), B)
            val = pool_a[idx] * U + (pool_C[idx] * V).sum(1)
            X += np.where(nl > t, val, 0)
        sgn = np.where(rng.random(B) < qplus, 1, -1)
        T = np.where(m == 8, S + sgn * X, T)
        T = np.maximum(T, 0)
        ti, tc = write_total(T, V)
        tint[k] = ti; tcnt[k] = tc
    if raw: return tint, tcnt
    return summaries(P, tint, tcnt)


def abc(P, L, ndraws, seed, chunk=20000, keep_frac=0.0005, tag='x', verbose=False):
    fn = os.path.join(OUT, f'abc_{tag}.npz')
    if os.path.exists(fn):
        z = np.load(fn); return {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    pool = setup(P, L)
    ck = os.path.join(OUT, f'abc_{tag}.ckpt.npz')
    obs = observed_summary(P, L)
    # pilot for scale
    vid = rng.integers(0, len(GRID), (chunk, L)); pp = rng.dirichlet(np.ones(NMS), chunk); qp = rng.random(chunk)
    sp = simulate(P, L, pool, vid, pp, qp, rng)
    sd = sp.std(0); sd[sd == 0] = 1
    keep = []
    done = 0; t0 = time.time()
    if os.path.exists(ck):
        z = np.load(ck); sd = z['sd']; done = int(z['done'])
        keep = [(z['d'], z['vid'], z['p'], z['q'])]
        rng = np.random.default_rng(seed + done)
    while done < ndraws:
        vid = rng.integers(0, len(GRID), (chunk, L)).astype(np.int8)
        pp = rng.dirichlet(np.ones(NMS), chunk); qp = rng.random(chunk)
        sm = simulate(P, L, pool, vid.astype(int), pp, qp, rng)
        d = np.sqrt((((sm - obs) / sd) ** 2).sum(1))
        thr = np.quantile(d, keep_frac * 10)
        sel = d <= thr
        keep.append((d[sel], vid[sel], pp[sel], qp[sel]))
        done += chunk
        if done % (chunk * 10) == 0:
            kd = np.concatenate([k[0] for k in keep]); kv = np.concatenate([k[1] for k in keep])
            kp = np.concatenate([k[2] for k in keep]); kq = np.concatenate([k[3] for k in keep])
            keep = [(kd, kv, kp, kq)]
            np.savez(ck, d=kd, vid=kv, p=kp, q=kq, sd=sd, done=done)
        if verbose and done % (chunk * 25) == 0: print(tag, done, time.time() - t0, flush=True)
    d = np.concatenate([k[0] for k in keep]); vid = np.concatenate([k[1] for k in keep])
    pp = np.concatenate([k[2] for k in keep]); qp = np.concatenate([k[3] for k in keep])
    nacc = int(keep_frac * ndraws)
    o = np.argsort(d)[:nacc]
    res = {'d': d[o], 'vid': vid[o], 'p': pp[o], 'q': qp[o], 'obs': obs, 'sd': sd}
    np.savez(fn, **res)
    if os.path.exists(ck): os.remove(ck)
    return res


def describe(res, letters, truth=None):
    out = {'n_acc': int(len(res['d'])), 'eps': float(res['d'].max()),
           'p_mech_post': dict(zip(MECH_SIM, res['p'].mean(0).round(3).tolist())),
           'obs_summary': dict(zip(SUM_NAMES, np.round(res['obs'], 3).tolist())), 'letters': {}}
    G = len(GRID)
    for i, l in enumerate(letters):
        h = np.bincount(res['vid'][:, i].astype(int), minlength=G) / len(res['d'])
        o = np.argsort(-h)[:4]
        d = {'top': [(GRID_LABEL[g], round(float(h[g]), 3)) for g in o], 'max_over_prior': round(float(h.max() * G), 2)}
        if truth:
            t = GRID_LABEL.index('%d/%d' % truth[l]); d['truth'] = GRID_LABEL[t]
            d['rank_truth'] = int((h > h[t]).sum()) + 1; d['mass_truth_x_prior'] = round(float(h[t] * G), 2)
        out['letters'][l] = d
    return out
