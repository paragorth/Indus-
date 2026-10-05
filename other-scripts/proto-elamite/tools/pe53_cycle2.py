"""pe53 cycle 2: posterior over assignments, order in the year, seasonal signs.
 2a PE: per-sign class marginals from all 6,050 assignment scores (weights exp(S1)),
    and the same on planted archives (does the marginal point at the truth?).
 2b Seasonal-sign test: each record gets a phase (posterior mean month under the best
    assignment); for every co-occurring sign, mean resultant length of its records'
    phases against 5,000 permutations of phases across records; family-wise via the max.
    Controls: Drehem with the TRUE Y/F terms but blind phases (do month names come out?),
    Drehem oracle (true months), planted archives with a season marker sign.
"""
import time, re
from collections import defaultdict
from multiprocessing import Pool
from pe53_common import *
from common import base, is_sign
from pe53_cycle1 import A8, TR8

HERD = set(PE_SIGNS) | {'M362', 'M367', 'M346', 'M006'}


def pe_record_signs(R):
    """Signs (base forms, herd class signs excluded) on the lines of each record plus the
    tablet's header lines before the first record."""
    T = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
    by_tab = defaultdict(list)
    for i, (tid, lab, _) in enumerate(R):
        by_tab[tid].append((i, lab))
    out = [set() for _ in R]
    for tid, recs in by_tab.items():
        lines = T[tid]['lines']
        keys = [(l['surface'][0] + l['label']) for l in lines]
        starts = []
        for i, lab in recs:
            if tid == 'P008294':
                m = re.match(r'([or])(\d+)', lab)
                idx = [j for j, l in enumerate(lines) if l['surface'][0] == m.group(1)
                       and re.match(r'^%s(\.|$|[a-z])' % m.group(2), l['label'].replace('?', ''))]
                for j in idx:
                    out[i] |= {base(s) for s in lines[j]['signs'] if is_sign(s)}
                continue
            j = keys.index(lab) if lab in keys else None
            starts.append((j, i))
        starts = sorted([s for s in starts if s[0] is not None])
        hdr = set()
        if starts:
            for l in lines[:starts[0][0]]:
                hdr |= {base(s) for s in l['signs'] if is_sign(s)}
        for k, (j, i) in enumerate(starts):
            j2 = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
            for l in lines[max(0, j - 1):j2]:
                out[i] |= {base(s) for s in l['signs'] if is_sign(s)}
            out[i] |= hdr
    return [{s for s in o if s not in HERD and s.split('~')[0] not in HERD} for o in out]


def seasonal_signs(phase, sets, rng, nperm=5000, minn=3):
    phase = np.asarray(phase)
    vocab = sorted({s for x in sets for s in x})
    cnt = {s: sum(s in x for x in sets) for s in vocab}
    vocab = [s for s in vocab if minn <= cnt[s] <= len(sets) - 2]
    if not vocab:
        return []
    Mx = np.array([[s in x for s in vocab] for x in sets], float)    # R x V
    z = np.exp(1j * phase / 12 * 2 * np.pi)

    def stat(zz):
        return np.abs(zz @ Mx) / Mx.sum(0)
    obs = stat(z)
    ge = np.zeros(len(vocab))
    mx = np.zeros(nperm)
    for b in range(nperm):
        st = stat(rng.permutation(z))
        ge += st >= obs
        mx[b] = st.max()
    mean_ph = (np.angle(z @ Mx) % (2 * np.pi)) / (2 * np.pi) * 12
    res = [dict(sign=s, n=int(Mx[:, v].sum()), R=float(obs[v]), p=float((ge[v] + 1) / (nperm + 1)),
                fwer=float((np.sum(mx >= obs[v]) + 1) / (nperm + 1)), mean_phase=float(mean_ph[v]))
           for v, s in enumerate(vocab)]
    return sorted(res, key=lambda r: r['p'])


def marginals(S, A, col=0):
    v = S[:, col]
    ok = np.isfinite(v)
    w = np.exp(v[ok] - v[ok].max())
    w /= w.sum()
    Aok = A[ok]
    return np.array([[np.sum(w * (Aok[:, j] == c)) for c in (0, 1, 2)] for j in range(A.shape[1])])


def plant_job(seed):
    """Planted seasonal archive with a season marker on records written in months 0-3."""
    rng = np.random.default_rng(seed)
    R, Mpe = pe_matrix()
    perm = rng.permutation(8)
    truth = np.zeros(8, np.int8)
    truth[perm[:2]] = 1
    truth[perm[2:4]] = 2
    th = THETAS[rng.integers(len(THETAS))]
    pc = curve(*th)
    k = KAPPAS[rng.integers(1, 4)]
    M = np.zeros_like(Mpe)
    ts = []
    for r in range(len(Mpe)):
        tot = Mpe[r].sum()
        nf = max(1, int(round(tot * rng.uniform(0.3, 0.7))))
        ti = rng.integers(len(TGRID))
        ts.append(TGRID[ti])
        p = np.clip(pc[ti], 0.01, 0.99)
        q = rng.beta(p * k, (1 - p) * k)
        ny = rng.binomial(int(nf / max(1 - q, 0.05)), q)
        no = int(rng.integers(0, max(2, tot // 2)))
        for cls, cnt in ((1, ny), (2, nf), (0, no)):
            idx = np.where(truth == cls)[0]
            if len(idx) and cnt:
                M[r, idx] += rng.multinomial(cnt, rng.dirichlet(np.ones(len(idx))))
    ts = np.array(ts)
    sets = [set(rng.choice(['x%d' % i for i in range(25)], 3, replace=False)) for _ in range(len(M))]
    for r in range(len(M)):
        if ts[r] < 3 and rng.random() < 0.8:
            sets[r].add('MARK')
    S = score_all(M, A8)
    mg = marginals(S, A8)
    top = A8[int(np.nanargmax(S[:, 0]))]
    y, n = yn_from(M, top)
    r = scores(y, n, True)
    ph, _ = circ_phase(r['post_t'])
    kept = np.where(r['keep'])[0]
    ss = seasonal_signs(ph, [sets[i] for i in kept], rng, nperm=2000)
    mk = [x for x in ss if x['sign'] == 'MARK']
    # correlation of inferred phase with truth (best rotation, forward time)
    hit, _ = best_rotation_hits(ph, ts[kept] + 1, tol=1.5)
    nullhit = np.mean([best_rotation_hits(ph, rng.permutation(ts[kept]) + 1, tol=1.5)[0] for _ in range(50)])
    return dict(seed=seed, truth=truth.tolist(), top=top.tolist(),
                marg_Y_on_trueY=float(mg[truth == 1, 1].mean()), marg_Y_on_other=float(mg[truth != 1, 1].mean()),
                marg_F_on_trueF=float(mg[truth == 2, 2].mean()), marg_F_on_other=float(mg[truth != 2, 2].mean()),
                phase_hit=float(hit), phase_hit_null=float(nullhit),
                mark_rank=(ss.index(mk[0]) + 1) if mk else None, mark_p=mk[0]['p'] if mk else None,
                mark_fwer=mk[0]['fwer'] if mk else None)


def drehem_job(args):
    mode, seed = args
    rng = np.random.default_rng(seed)
    d = json.load(open(os.path.join(DATA, 'pe47_ckpt', 'ur3_drehem.json')))
    ctx = {x['id']: set(x['ctx']) for x in d}
    X, mo, ids = drehem_matrix()
    X = X[:, :8]
    y, n = yn_from(X, TR8)
    pool = np.where(n > 0)[0]
    pick = rng.choice(pool, 600, replace=False)
    if mode == 'oracle':
        ph = (mo[pick] - 1).astype(float)
        kept = pick
    else:
        r = scores(y[pick], n[pick], True)
        ph, _ = circ_phase(r['post_t'])
        kept = pick[r['keep']]
    sets = [ctx[ids[i]] for i in kept]
    ss = seasonal_signs(ph, sets, rng, nperm=2000, minn=8)
    months = [x for x in ss if x['sign'].startswith('iti:') and not re.search(r'\d|ba-zal|la2|u4|szu-esz', x['sign'])]
    hit, _ = best_rotation_hits(ph, mo[kept], tol=1.5)
    nullhit = np.mean([best_rotation_hits(ph, rng.permutation(mo[kept]), tol=1.5)[0] for _ in range(50)])
    return dict(mode=mode, seed=seed, n=len(kept), phase_hit=float(hit), phase_hit_null=float(nullhit),
                top10=[(x['sign'], round(x['p'], 4), round(x['fwer'], 3)) for x in ss[:10]],
                month_names_fwer05=sum(x['fwer'] < 0.05 for x in months), month_names=len(months),
                month_names_p05=sum(x['p'] < 0.05 for x in months))


def pe_part():
    rng = np.random.default_rng(11)
    R, M = pe_matrix()
    S = np.load(os.path.join(CK, 'c1_pe_scores.npy'))
    out = {}
    for col, nm in ((0, 'S1'), (1, 'S2')):
        mg = marginals(S, A8, col)
        order = np.argsort(-np.nan_to_num(S[:, col], nan=-1e9))[:10]
        out[nm] = dict(marg={s: dict(O=round(m[0], 3), Y=round(m[1], 3), F=round(m[2], 3))
                             for s, m in zip(PE_SIGNS, mg)},
                       top=[(''.join('OYF'[c] for c in A8[i]), round(S[i, 0], 2), round(S[i, 1], 2)) for i in order])
    best = A8[int(np.nanargmax(S[:, 0]))]
    y, n = yn_from(M, best)
    r = scores(y, n, True)
    ph, conc = circ_phase(r['post_t'])
    kept = np.where(r['keep'])[0]
    sets = pe_record_signs(R)
    ss = seasonal_signs(ph, [sets[i] for i in kept], rng)
    out['best'] = ''.join('OYF'[c] for c in best)
    out['theta'] = r['theta']
    out['order'] = sorted([(R[i][0], R[i][1], round(float(ph[j]), 2), round(float(conc[j]), 2),
                            int(y[i]), int(n[i])) for j, i in enumerate(kept)], key=lambda x: x[2])
    out['seasonal_signs'] = ss[:12]
    out['n_signs_tested'] = len(ss)
    return out


if __name__ == '__main__':
    t0 = time.time()
    res = {'pe': pe_part()}
    print(json.dumps(res['pe'], default=str)[:3000], flush=True)
    jobs_d = [('blind', s) for s in range(6)] + [('oracle', 50 + s) for s in range(2)]
    with Pool(2) as P:
        res['drehem'] = []
        for r in P.imap_unordered(drehem_job, jobs_d):
            res['drehem'].append(r)
            print(round(time.time() - t0), r, flush=True)
        res['plant'] = []
        for r in P.imap_unordered(plant_job, range(700, 716)):
            res['plant'].append(r)
            print(round(time.time() - t0), r, flush=True)
            dump(res, os.path.join(CK, 'c2.json'))
    dump(res, os.path.join(DATA, 'pe53_cycle2.json'))
