"""la75 cycle 2: number habits as handwriting. Random feature subsets define a per-document 'number-habit profile';
subsets are selected blind by split-half reliability (entries of each document split in two) against the stratum
permutation null; the selected profile and document distance matrix are frozen (sha256) and only then compared with
outside data the numbers never saw: HT findspot and scribe (LA), hand (LB KN/PY). Planted rooms control."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from la75_common import *
from la75_c1 import FastPanel, dedupe

NS = int(os.environ.get('NS', 3000)); K = 12; NPERM = int(os.environ.get('NPERM', 200))


def profiles(P, R):
    G = P.M @ R
    return G / np.sqrt(P.nd)[:, None]


def split_rel(P, R, rng, reps=4):
    """Split-half reliability per feature subset is computed later; here return two half-profiles per rep."""
    out = []
    for _ in range(reps):
        h = np.zeros(len(P.rows), bool)
        for d in range(len(P.docs)):
            ix = np.where(P.g == d)[0]; rng.shuffle(ix); h[ix[: len(ix) // 2]] = True
        A = P.M @ (R * h[:, None]); B = P.M @ (R * (~h)[:, None])
        out.append((A, B))
    return out


def subset_scores(halves, subsets, nd):
    """reliability of document distances: corr(dist_A, dist_B) over doc pairs, for each subset."""
    iu = np.triu_indices(len(nd), 1); sc = np.zeros(len(subsets))
    for A, B in halves:
        for j, S in enumerate(subsets):
            a = A[:, S]; b = B[:, S]
            da = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)[iu]; db = ((b[:, None, :] - b[None, :, :]) ** 2).sum(-1)[iu]
            sc[j] += np.corrcoef(da, db)[0, 1] if da.std() > 0 and db.std() > 0 else 0
    return sc / len(halves)


def run(rows, feats, rng, label, nsub=NS, nperm=NPERM):
    F = fmatrix(rows, feats); P = FastPanel(rows)
    keep = np.where((P.resid(F) ** 2).sum(0) > 2)[0]; keep = keep[dedupe(F[:, keep])]
    R = P.resid(F[:, keep]).astype(np.float64); R /= (R.std(0) + 1e-9)
    subsets = [rng.choice(len(keep), K, replace=False) for _ in range(nsub)]
    halves = split_rel(P, R, np.random.default_rng(1))
    real = subset_scores(halves, subsets, P.nd)
    # null: same subsets, entries permuted within strata (only the top-scoring statistic matters)
    null_max = []; null_top = []
    for _ in range(nperm):
        Rp = R[P.perm(rng)]; hp = split_rel(P, Rp, np.random.default_rng(1), reps=1)
        s = subset_scores(hp, subsets[:300], P.nd); null_max.append(s.max()); null_top.append(np.sort(s)[-15:].mean())
    real300 = np.sort(real[:300])[-15:].mean()
    p_top = (1 + sum(x >= real300 for x in null_top)) / (1 + nperm)
    best = np.argsort(-real)[:50]
    union = sorted(set(int(i) for b in best for i in subsets[b]))
    prof = profiles(P, R[:, union])
    return dict(label=label, n_docs=len(P.docs), n_feat=len(keep), real_top15_300=float(real300),
                null_top15=float(np.mean(null_top)), null_top15_95=float(np.percentile(null_top, 95)), p_top=float(p_top),
                best_rel=float(real[best[0]]), union=[feats[keep[i]] for i in union], docs=P.docs, prof=prof)


def round_score(rows, feat=('last', (0, 5))):
    F = fmatrix(rows, [feat]); P = FastPanel(rows); R = P.resid(F)[:, 0].astype(float)
    G = np.bincount(P.g, weights=R, minlength=len(P.docs)); V = np.bincount(P.g, weights=np.full(len(R), R.var()), minlength=len(P.docs))
    return P.docs, G / np.sqrt(V + 1e-12)


def outside_anova(docs, z, attr, rng, nperm=20000):
    lab = [attr.get(d, '') for d in docs]; ix = [i for i, l in enumerate(lab) if l]
    if len(ix) < 6: return None
    zz = z[ix]; L = np.array([lab[i] for i in ix]); u, inv = np.unique(L, return_inverse=True)
    def stat(inv):
        n = np.bincount(inv, minlength=len(u)); m = np.bincount(inv, weights=zz, minlength=len(u)) / np.maximum(n, 1)
        return (n * (m - zz.mean()) ** 2).sum()
    obs = stat(inv); ge = 1 + sum(stat(rng.permutation(inv)) >= obs for _ in range(nperm))
    n = np.bincount(inv); m = np.bincount(inv, weights=zz) / n
    top = sorted(zip(u, n, m), key=lambda t: -abs(t[2]) * np.sqrt(t[1]))[:5]
    return dict(n=len(ix), n_labels=len(u), stat=float(obs), p=ge / (nperm + 1), top=[(a, int(b), round(float(c), 2)) for a, b, c in top])


def outside(docs, prof, attr, rng, nperm=20000):
    """mean profile distance between docs sharing a non-empty attribute vs not; permute labels among labelled docs."""
    lab = [attr.get(d, '') for d in docs]; ix = [i for i, l in enumerate(lab) if l]
    if len(ix) < 6: return None
    X = prof[ix]; L = np.array([lab[i] for i in ix])
    Dm = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); iu = np.triu_indices(len(ix), 1)
    d = Dm[iu]
    def stat(L):
        same = (L[:, None] == L[None, :])[iu]
        return d[same].mean() - d[~same].mean() if same.any() and (~same).any() else 0
    obs = stat(L); ge = 1
    for _ in range(nperm):
        ge += stat(rng.permutation(L)) <= obs
    nsame = int((L[:, None] == L[None, :])[iu].sum())
    return dict(n=len(ix), n_labels=len(set(L)), same_pairs=nsame, diff=float(obs), p=ge / (nperm + 1))


if __name__ == '__main__':
    t0 = time.time(); rng = np.random.default_rng(75002)
    c1 = pickle.load(open(os.path.join(CK, 'c1.pkl'), 'rb')); feats = c1['feats']
    rows, docs = la_entries(); rows = prepare(rows)
    out = {}
    # ---- LA: select blind, freeze, then outside
    r = run(rows, feats, rng, 'LA'); print('LA run', time.time() - t0, flush=True)
    frozen = dict(docs=r['docs'], union=r['union'], prof=np.round(r['prof'], 6).tolist())
    h = sha(frozen); json.dump(dict(sha256=h, **frozen), open(os.path.join(D, 'la75_frozen_profiles.json'), 'w'))
    open(os.path.join(D, 'la75_frozen_profiles.sha256'), 'w').write(h + '\n')
    r['sha256'] = h
    fs = {d: docs[d]['findspot'] for d in r['docs'] if docs[d]['site'] == 'Haghia Triada'}
    sc = {d: docs[d]['scribe'] for d in r['docs'] if docs[d]['site'] == 'Haghia Triada'}
    site = {d: docs[d]['site'] for d in r['docs']}
    ctx = {d: docs[d]['context'] for d in r['docs']}
    r['out_findspot'] = outside(r['docs'], r['prof'], fs, rng)
    r['out_scribe'] = outside(r['docs'], r['prof'], sc, rng)
    r['out_site'] = outside(r['docs'], r['prof'], site, rng)
    r['out_context'] = outside(r['docs'], r['prof'], ctx, rng)
    zd, z = round_score(rows)
    r['round_findspot'] = outside_anova(zd, z, fs, rng); r['round_scribe'] = outside_anova(zd, z, sc, rng)
    r['round_site'] = outside_anova(zd, z, site, rng)
    out['LA'] = r
    # ---- planted rooms: round5 planted on 50 % of docs (full rounding), 'room' label = planted or not, plus noise label
    from la75_c1 import plant
    for s in range(2):
        pr, pl = plant(rows, np.random.default_rng(7700 + s), 'round5', 0.5, 1.0)
        rp = run(pr, feats, rng, f'plant rooms s{s}', nperm=50)
        room = {d: ('R1' if d in pl else 'R2') if rng.random() < 0.8 else rng.choice(['R1', 'R2']) for d in rp['docs']}
        rp['out_room'] = outside(rp['docs'], rp['prof'], room, rng, 5000)
        zd, z = round_score(pr); rp['round_room'] = outside_anova(zd, z, room, rng, 5000)
        # realistic planted rooms: 25 % docs at p 0.7, label tracks plant 80 %
        pr2, pl2 = plant(rows, np.random.default_rng(7750 + s), 'round5', 0.25, 0.7)
        room2 = {d: ('R1' if d in pl2 else 'R2') if rng.random() < 0.8 else rng.choice(['R1', 'R2']) for d in rp['docs']}
        zd, z = round_score(pr2); rp['round_room_weak'] = outside_anova(zd, z, room2, rng, 5000)
        out[f'plant_{s}'] = rp; print('plant', s, time.time() - t0, flush=True)
    # ---- LB controls: hand
    for p in ('KN', 'PY'):
        b = prepare(lb_entries(p)); rb = run(b, feats, rng, 'LB ' + p, nperm=50)
        hand = {}
        for x in b: hand[x['doc']] = x['scribe'] if x['scribe'] not in ('', '-') else ''
        rb['out_hand'] = outside(rb['docs'], rb['prof'], hand, rng, 5000)
        zd, z = round_score(b); rb['round_hand'] = outside_anova(zd, z, hand, rng, 5000)
        out['LB_' + p] = rb; print(p, time.time() - t0, flush=True)
    pickle.dump(out, open(os.path.join(CK, 'c2.pkl'), 'wb'))
    for k, o in out.items():
        print(k, {x: o[x] for x in o if x not in ('prof', 'docs', 'union')})
        print('   union', len(o['union']), o['union'][:8])
