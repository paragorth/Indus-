"""pe37 cycle 2: does the bone ruler transfer to held-out tablets?
  H  PE: random 50/50 tablet splits. Train: all 7^8 assignments scored by the Malyan
     Banesh target on train totals -> MAP. Test: rank of that MAP among all assignments
     on the test totals (same target). Null: the same with random faunal profiles.
     Positive control: Ur III oracle (true composition as target), 24-tablet halves.
  X  Extended sign set with cattle: 20 most-counted final signs of herd tablets,
     classes none/sheep F M Y/goat F M Y/cattle; Metropolis posterior at beta 1;
     null: random faunal profiles.
Usage: python3 pe37_cycle2.py H|X n
"""
import time
from pe37_common import *
import pe20_common as P20

part = sys.argv[1]
N = int(sys.argv[2])
rng = np.random.default_rng(372 if part == 'H' else 373)
TG = json.load(open(TARGETS))['banesh_all']
SD = {'sheep': TG['sheep_share_nisp'][1], 'young': TG['living_young_share'][1],
      'adF': TG['living_adultF_share'][1]}
REAL = {'sheep': TG['sheep_share_nisp'][0], 'young': TG['living_young_share'][0],
        'adF': TG['living_adultF_share'][0]}


def target_of(m):
    return {k: beta_from(m[k], SD[k]) for k in m}


def rnd_profile():
    return {'sheep': rng.uniform(0.05, 0.95), 'young': rng.uniform(0.1, 0.5), 'adF': rng.uniform(0.4, 0.95)}


def heldout(recs, signs, tg, A):
    tabs = sorted({r[0] for r in recs})
    tr = set(rng.choice(tabs, len(tabs) // 2, replace=False))
    ttr = np.nansum(P20.to_matrix([r for r in recs if r[0] in tr], signs), 0)
    tte = np.nansum(P20.to_matrix([r for r in recs if r[0] not in tr], signs), 0)
    _, s1 = enumerate_scores(ttr, tg, A=A)
    best = int(np.nanargmax(np.where(np.isfinite(s1), s1, -1e18)))
    _, s2 = enumerate_scores(tte, tg, A=A)
    v = s2[best]
    fin = s2[np.isfinite(s2)]
    return float(np.mean(fin > v)) if np.isfinite(v) else 1.0


out = {}
if part == 'H':
    pe = P20.pe_records()
    A8 = all_assignments(8, range(7))
    res = {'bone': [], 'random': []}
    t0 = time.time()
    for i in range(N):
        res['bone'].append(heldout(pe, P20.PE_SIGNS, target_of(REAL), A8))
        res['random'].append(heldout(pe, P20.PE_SIGNS, target_of(rnd_profile()), A8))
        if i % 5 == 4:
            print(i + 1, 'bone rank %.3f random %.3f (%.0fs)' % (np.mean(res['bone']), np.mean(res['random']), time.time() - t0), flush=True)
    ur = P20.ur_herd_records()
    US = [s for s in P20.UR_SIGNS if s != 'asz2-gar3']
    tU = np.nansum(P20.to_matrix(ur, US), 0)
    S = class_sums(np.array([[1, 2, 3, 3, 4, 5, 6]]), tU)
    p = proportions(S)
    tp = {'sheep': float(p[0][0]), 'young': float(p[1][0]), 'adF': float(p[2][0])}
    A7 = all_assignments(7, range(7))
    pids = sorted({r[0] for r in ur})
    res['ur_oracle'], res['ur_random'] = [], []
    for i in range(N):
        keep = set(rng.choice(pids, 24, replace=False))
        sub = [r for r in ur if r[0] in keep]
        res['ur_oracle'].append(heldout(sub, US, target_of(tp), A7))
        res['ur_random'].append(heldout(sub, US, target_of(rnd_profile()), A7))
    for k, v in res.items():
        out[k] = {'mean_rank': float(np.mean(v)), 'median': float(np.median(v)), 'vals': v}
    from scipy.stats import mannwhitneyu
    out['p_bone_lt_random'] = float(mannwhitneyu(res['bone'], res['random'], alternative='less').pvalue)
    out['p_ur_oracle_lt_random'] = float(mannwhitneyu(res['ur_oracle'], res['ur_random'], alternative='less').pvalue)
    print({k: (v['mean_rank'] if isinstance(v, dict) else v) for k, v in out.items()}, flush=True)
    dump(out, os.path.join(DATA, 'pe37_cycle2_H.json'))

if part == 'X':
    T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    pe = P20.pe_records()
    herd = {r[0] for r in pe} | P20.LIST_TABS
    tot = Counter()
    per_tab = {}
    for t in T:
        if t['id'] not in herd:
            continue
        for l in t['lines']:
            if not l['signs']:
                continue
            v = P20.numval(l)
            if v is None:
                continue
            s = P20.canon(l['signs']) or P20.clean(l['signs'][-1])
            if s in ('x', 'n', 'X'):
                continue
            tot[s] += v
            per_tab.setdefault(t['id'], Counter())[s] += v
    signs = [s for s, _ in tot.most_common(20)]
    t = np.array([tot[s] for s in signs], float)
    print('extended signs', list(zip(signs, t.tolist())), flush=True)
    tg = target_of(REAL)
    tg['cat'] = beta_from(TG['cattle_share_nisp'][0], TG['cattle_share_nisp'][1])
    USE = ('sheep', 'young', 'adF', 'cat')

    def sc1(a, tt):
        return float(score_props(proportions(class_sums(a[None, :], tt)), tg, USE)[0])

    def mcmc(tt, sweeps, beta=1.0):
        K = len(tt)
        # anneal to a valid start
        a = rng.integers(0, NCL, K)
        while not np.isfinite(sc1(a, tt)):
            a = rng.integers(0, NCL, K)
        cur = sc1(a, tt)
        M = np.zeros((K, NCL))
        best = (cur, a.copy())
        for sw in range(sweeps):
            b = beta if sw > sweeps // 4 else beta * (0.2 + 0.8 * sw / (sweeps // 4))
            for j in rng.permutation(K):
                cand = np.repeat(a[None, :], NCL, 0)
                cand[:, j] = np.arange(NCL)
                s = score_props(proportions(class_sums(cand, tt)), tg, USE)
                p = np.exp(b * (s - np.max(s)))
                p /= p.sum()
                a[j] = rng.choice(NCL, p=p)
                cur = s[a[j]]
                if cur > best[0]:
                    best = (cur, a.copy())
            if sw > sweeps // 4:
                M[np.arange(K), a] += 1
        return M / M.sum(1, keepdims=True), best
    M, best = mcmc(t, N)
    out['signs'] = signs
    out['totals'] = t
    out['marg'] = {s: {CLASSES[c]: float(M[j, c]) for c in range(NCL)} for j, s in enumerate(signs)}
    out['best'] = ({s: CLASSES[c] for s, c in zip(signs, best[1])}, best[0])
    for s, d in out['marg'].items():
        print('  %-14s' % s, ' '.join('%s %.2f' % (k, v) for k, v in d.items()), flush=True)
    print('best', out['best'], flush=True)
    # cattle: is P(cat) just a function of smallness? rank correlation with total
    pc = np.array([M[j, 7] for j in range(len(signs))])
    from scipy.stats import spearmanr
    out['cat_vs_total_rho'] = float(spearmanr(pc, t).correlation)
    # null: random faunal profiles (incl. cattle share 0-0.2) -> marginal concentration
    conc = float(M.max(1).mean())
    nul = []
    real_tg = dict(tg)
    for i in range(10):
        tg.clear()
        tg.update(target_of(rnd_profile()))
        tg['cat'] = beta_from(rng.uniform(0.002, 0.2), TG['cattle_share_nisp'][1])
        Mn, _ = mcmc(t, max(20, N // 3))
        nul.append(float(Mn.max(1).mean()))
    tg.clear(); tg.update(real_tg)
    out['conc'] = conc
    out['null_conc'] = nul
    print('cat-vs-total rho %.2f conc %.3f null %s' % (out['cat_vs_total_rho'], conc, np.round(nul, 3)), flush=True)
    dump(out, os.path.join(DATA, 'pe37_cycle2_X.json'))
