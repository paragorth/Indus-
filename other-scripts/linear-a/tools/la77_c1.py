"""LA-77 cycle 1: random mutation models x random trees over mutant families.

Hypothesis h = (w: 6 log-rates for {sub,indel} x {initial, medial, final}; sigma: tree noise;
p_net: share of non-tree edit edges kept as reticulations). For each family the tree is the
min-cost spanning tree of its edit-1 graph under cost -w[class] + sigma*N(0,1). Score = sum over
tree (+ network) edges of log p(class), p = softmax(w + log opportunities). 200,000 random h
scored on a train half of families; the top 0.5 % re-scored on held-out families.
Controls: signs shuffled within site (5 draws); Linear B (DAMOS, KN, LA-sized type samples) as
a calibration of what an inflected script looks like.
"""
import sys, json, collections, time
import numpy as np
from numba import njit
from la77_common import *

NH = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
KI = {k: i for i, k in enumerate(KCLS)}


def fam_arrays(types, edges, fams):
    """flatten per-family edge lists: (fam_id, local u, local v, class)."""
    loc = {}
    for f, mem in enumerate(fams):
        for a, i in enumerate(mem):
            loc[i] = (f, a)
    rows = []
    for i, j, op, pos in edges:
        if i in loc:
            f, a = loc[i]; b = loc[j][1]
            rows.append((f, a, b, KI[(op, pos)]))
    rows.sort()
    R = np.array(rows, np.int64).reshape(-1, 4)
    sizes = np.array([len(m) for m in fams], np.int64)
    starts = np.searchsorted(R[:, 0], np.arange(len(fams) + 1))
    return R, sizes, starts


@njit(cache=True)
def score_all(R, sizes, starts, famsel, W, SIG, PNET, logO, seed):
    np.random.seed(seed)
    H = W.shape[0]
    out = np.zeros(H)
    cnt = np.zeros((H, 6))
    par = np.zeros(64, np.int64)
    for h in range(H):
        w = W[h]
        lp = w + logO
        m = lp.max(); z = np.log(np.exp(lp - m).sum()) + m
        lp = lp - z
        tot = 0.0
        for f in famsel:
            s0, s1 = starts[f], starts[f + 1]
            ne = s1 - s0
            cost = np.empty(ne)
            for e in range(ne):
                cost[e] = -w[R[s0 + e, 3]] + SIG[h] * np.random.randn()
            order = np.argsort(cost)
            n = sizes[f]
            if n > par.shape[0]:
                par = np.zeros(n, np.int64)
            for a in range(n):
                par[a] = a
            for e in order:
                u = R[s0 + e, 1]; v = R[s0 + e, 2]; k = R[s0 + e, 3]
                while par[u] != u:
                    u = par[u]
                while par[v] != v:
                    v = par[v]
                if u != v:
                    par[u] = v
                    tot += lp[k]; cnt[h, k] += 1
                elif np.random.rand() < PNET[h]:
                    tot += lp[k]; cnt[h, k] += 1
        out[h] = tot
    return out, cnt


def total_opp(types):
    O = np.zeros(6)
    for t in types:
        if len(t) >= 2:
            o = opportunities(t)
            if len(t) < 3:
                o[:3] = 0           # 2-sign substitutions excluded by design
            O += o
    return np.log(O)


def run(types, label, rng, nsplit=5):
    E, F = families(types)
    R, sizes, starts = fam_arrays(types, E, F)
    logO = total_opp(types)
    obs = np.bincount(R[:, 3], minlength=6).astype(float)
    W = rng.normal(0, 1.5, (NH, 6)); SIG = rng.uniform(0, 2, NH); PNET = rng.uniform(0, 0.3, NH)
    res = []
    for sp in range(nsplit):
        perm = rng.permutation(len(F)); tr = np.sort(perm[:len(F) // 2]); te = np.sort(perm[len(F) // 2:])
        s_tr, _ = score_all(R, sizes, starts, tr, W, SIG, PNET, logO, int(rng.integers(1 << 30)))
        top = np.argsort(-s_tr)[:max(NH // 200, 10)]
        s_te, c_te = score_all(R, sizes, starts, te, W, SIG, PNET, logO, int(rng.integers(1 << 30)))
        # baseline: uniform w (opportunity only), same trees noise-free
        z = np.zeros((1, 6))
        b_te, bc = score_all(R, sizes, starts, te, z, np.zeros(1), np.zeros(1), logO, 1)
        ne_te = bc[0].sum()
        gain = (np.mean(s_te[top]) - b_te[0]) / max(ne_te, 1)
        pct = float(np.mean(s_te < np.mean(s_te[top])))
        lp = W[top] + logO
        P = np.exp(lp - lp.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        res.append(dict(gain=gain, pct=pct, P=P.mean(0).tolist(), nte=float(ne_te)))
    Pbar = np.mean([r['P'] for r in res], 0)
    base = np.exp(logO) / np.exp(logO).sum()
    return dict(label=label, ntypes=len(types), nfam=len(F), nedges=int(R.shape[0]),
                obs=obs.tolist(), obs_share=(obs / obs.sum()).tolist(), base=base.tolist(),
                enrich_obs=((obs / obs.sum()) / base).tolist(), P_surv=Pbar.tolist(),
                heldout_gain_nats=float(np.mean([r['gain'] for r in res])),
                heldout_gain_sd=float(np.std([r['gain'] for r in res])),
                heldout_pct=float(np.mean([r['pct'] for r in res])))


def lb_types(rng, n):
    from la15_common import load_lb
    D = [d for d in load_lb() if d['id'].startswith('KN')]
    ty = sorted({tuple(w.split('-')) for d in D for w in d['words']})
    ty = [t for t in ty if len(t) >= 2 and not any('*' in s for s in t)]
    idx = rng.choice(len(ty), min(n, len(ty)), replace=False)
    return [ty[i] for i in idx]


if __name__ == '__main__':
    rng = np.random.default_rng(77)
    t0 = time.time()
    docs = load_docs()
    out = []
    T, _ = type_table(docs)
    out.append(run(T, 'LA real', rng))
    print(json.dumps(out[-1]), time.time() - t0, flush=True)
    adm = [d for d in docs if d['support'] in ('Tablet', 'Nodule', 'Roundel', 'Lames (short thin tablet)', 'Sealing')]
    out.append(run(type_table(adm)[0], 'LA admin only', rng))
    print(json.dumps(out[-1]), flush=True)
    for s in range(5):
        Ts, _ = type_table(shuffle_within_site(docs, np.random.default_rng(1000 + s)))
        out.append(run(Ts, f'LA shuffled-within-site {s}', rng, nsplit=2))
        print(json.dumps(out[-1]), flush=True)
    for s in range(3):
        out.append(run(lb_types(np.random.default_rng(2000 + s), len(T)), f'LB KN LA-sized {s}', rng, nsplit=2))
        print(json.dumps(out[-1]), flush=True)
    # class-profile test: real edit-class shares vs 200 within-site sign shuffles (no hypotheses)
    def shares(ty):
        E, F = families(ty)
        c = np.zeros(6)
        for i, j, op, pos in E:
            c[KI[(op, pos)]] += 1
        return c, len(F)
    real, nf = shares(T)
    sh = np.array([shares(type_table(shuffle_within_site(docs, np.random.default_rng(5000 + s)))[0])[0]
                   for s in range(200)])
    rs = real / real.sum(); ss = sh / sh.sum(1, keepdims=True)
    z = (rs - ss.mean(0)) / ss.std(0)
    pz = [float(min(1, 2 * min(np.mean(ss[:, k] >= rs[k]), np.mean(ss[:, k] <= rs[k])) + 1 / 201)) for k in range(6)]
    lbs = np.array([shares(lb_types(np.random.default_rng(3000 + s), len(T)))[0] for s in range(10)])
    lbs = lbs / lbs.sum(1, keepdims=True)
    out.append(dict(label='class test', classes=[f'{a}-{b}' for a, b in KCLS], real=rs.tolist(),
                    shuf_mean=ss.mean(0).tolist(), shuf_sd=ss.std(0).tolist(), z=z.tolist(), p=pz,
                    real_edges_total=float(real.sum()), shuf_edges_mean=float(sh.sum(1).mean()),
                    lb_mean=lbs.mean(0).tolist(), lb_sd=lbs.std(0).tolist()))
    print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(os.path.join(CKPT, 'c1.json'), 'w'), indent=1)
    print('done', time.time() - t0)
