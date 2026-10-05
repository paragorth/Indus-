"""v66 cycle 2: is the match SPECIFIC to the medical concept graph?

Cycle 1 showed the best-match score S* mostly measures how clustered a text's keyword graph is.
The QAP score is invariant to relabelling the concept graph, so the right null is a different
concept graph with the same edge-weight values and (nearly) the same node strengths:
value-swap rewiring of C under a Metropolis constraint on node strengths.
  z_C = (S*(C_real, B) - mean S*(C_null, B)) / sd.
A real herbal (opaque + padding) must give z_C > 0 against a leave-one-out reference graph;
a chronicle/novel must not. Run at two concept resolutions: 47 classes and 12 macro classes.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v66_lib import *
from multiprocessing import Pool

NU = 227
NCN = 16          # concept-graph nulls per text
R = 24            # restarts per match

MACRO = {
    'HEADFACE': ['HEAD', 'EYE', 'EAR', 'MOUTH'], 'TRUNK': ['CHEST', 'STOMACH', 'LIVER', 'HEART'],
    'URO_WOMB': ['URINE', 'WOMB'], 'SKIN_BLOOD': ['SKIN', 'BLOOD', 'JOINT'],
    'ILLNESS': ['FEVER', 'COUGH', 'PAIN', 'WORM', 'MIND', 'POISON'],
    'VEHICLE': ['WINE', 'HONEY', 'VINEGAR', 'OIL', 'WATER', 'JUICE'],
    'PREP': ['POWDER', 'COOK', 'DRINK', 'SALVE', 'SYRUP', 'SUGAR_SPICE'],
    'UNDERGROUND': ['ROOT', 'STALK'], 'ABOVE': ['LEAF', 'FLOWER', 'FRUIT', 'SEED'],
    'QUALITY': ['HOT', 'COLD', 'DRY', 'MOIST'], 'SKY_TIME': ['SUN', 'MOON', 'SEASON', 'MONTH', 'DAY_NIGHT'],
    'HABITAT': ['GARDEN', 'WILD'],
}
MNAMES = list(MACRO)
C2M = {c: m for m, cs in MACRO.items() for c in cs}


def macro_graph(T, names):
    Ms = []
    for k in names:
        X = class_units(T[k]['units'])
        Y = np.zeros((X.shape[0], len(MNAMES)), np.uint8)
        for j, c in enumerate(CLASS_NAMES):
            Y[:, MNAMES.index(C2M[c])] |= X[:, j]
        Ms.append(np.arctanh(np.clip(phi(Y), -0.99, 0.99)))
    M = np.tanh(np.mean(Ms, 0)); np.fill_diagonal(M, 0)
    return M


def rewire_values(C, seed, sweeps=60, temp=None):
    """permute edge values among pairs (multiset kept) under a Metropolis penalty on node-strength drift."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]; iu = np.triu_indices(n, 1)
    v = C[iu].copy(); E = len(v)
    a, b = iu
    s0 = C.sum(1); M = C.copy()
    s = s0.copy()
    temp = temp or (np.std(v) ** 2) * 8.0
    for _ in range(sweeps * E):
        e, f = rng.integers(E, size=2)
        if e == f: continue
        d = v[f] - v[e]
        ds = np.zeros(n)
        ds[a[e]] += d; ds[b[e]] += d; ds[a[f]] -= d; ds[b[f]] -= d
        idx = [a[e], b[e], a[f], b[f]]
        old = sum((s[i] - s0[i]) ** 2 for i in set(idx))
        new = sum((s[i] + ds[i] - s0[i]) ** 2 for i in set(idx))
        if new <= old or rng.random() < np.exp(-(new - old) / temp):
            v[e], v[f] = v[f], v[e]
            for i in set(idx): s[i] += ds[i]
    N = np.zeros_like(C); N[iu] = v; N = N + N.T
    return N


def job(args):
    name, kind, seed, level = args
    t0 = time.time()
    T = load_texts()
    if kind == 'voynich':
        _, units, _ = voynich_pages(name); back = {}; ref = REF
    else:
        units = sample_units(T[name]['units'], NU, seed)
        units, back = opaque(units, seed)
        ref = [r for r in REF if r != name and r != TWIN.get(name)]
    C = concept_graph(T, ref) if level == 47 else macro_graph(T, ref)
    Cz = zoff(C)
    K = 80 if level == 47 else 40
    kw, X, B = keyword_graph(units, K=K)
    S, m, _, _ = qap(Cz, B, restarts=R * 2, seed=seed)
    nul = []
    for r in range(NCN):
        Cn = zoff(rewire_values(C, seed * 500 + r, sweeps=30))
        nul.append(qap(Cn, B, restarts=R, seed=seed * 500 + r)[0])
    out = {'name': name, 'kind': kind, 'seed': seed, 'level': level, 'S': S, 'cn_mu': float(np.mean(nul)),
           'cn_sd': float(np.std(nul) + 1e-9), 'cn': nul}
    out['zC'] = (S - out['cn_mu']) / out['cn_sd']
    out['pct'] = float(np.mean(np.array(nul) < S))
    # held-out: concept-specificity on the other half (map fit on half A with real C vs null Cs, scored on half B with real C)
    A, Bu = units[0::2], units[1::2]
    BA, BB = zoff(phi(incidence(A, kw))), zoff(phi(incidence(Bu, kw)))
    _, mA, _, _ = qap(Cz, BA, restarts=R, seed=seed + 3)
    ho = score(Cz, BB, mA)
    hon = []
    for r in range(8):
        Cn = zoff(rewire_values(C, seed * 900 + r, sweeps=30))
        _, mn, _, _ = qap(Cn, BA, restarts=R // 2, seed=seed * 900 + r)
        hon.append(score(Cz, BB, mn))
    out['ho'] = ho; out['ho_null'] = hon
    out['zHO'] = (ho - np.mean(hon)) / (np.std(hon) + 1e-9)
    if level == 47:
        out['map'] = {CLASS_NAMES[i]: kw[k] for i, k in enumerate(m)}
        if back: out['rec'] = recovery(m, kw, back)
    else:
        out['map'] = {MNAMES[i]: kw[k] for i, k in enumerate(m)}
        if back:
            cls = [classify(back.get(w, w)) for w in kw]
            mc = [C2M[c] if c else None for c in cls]
            hit = sum(1 for i, k in enumerate(m) if mc[k] == MNAMES[i])
            exp = sum(1 for c in mc if c) / len(kw)
            out['rec'] = (hit, exp, sum(1 for c in mc if c))
    out['sec'] = time.time() - t0
    print('%-14s L%d s%d S %.3f nullC %.3f+-%.3f zC %.2f pct %.2f zHO %.2f rec %s %.0fs' % (name, level, seed, S, out['cn_mu'], out['cn_sd'], out['zC'], out['pct'], out['zHO'], out.get('rec'), out['sec']), flush=True)
    return out


if __name__ == '__main__':
    jobs = []
    for level in (12, 47):
        jobs += [('ZL3b', 'voynich', 11, level), ('IT2a', 'voynich', 12, level)]
        for s in (1, 2):
            jobs += [(n, 'control', s, level) for n in ['culpeper', 'konrad_plants', 'macer', 'circa_fr', 'celsus_lat', 'v21_IT']]
            jobs += [(n, 'unrelated', s, level) for n in UNRELATED]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1, default=float)
