#!/usr/bin/env python3
"""LA-32 cycle 3: massive random guessing.
Each hypothesis = (random shape matrix: random positive weights over 4 metrics, random glyph blur;
                   random sound matrix: blind la21 rows from a random subset of runs, thresholded or not;
                   random pair filter: min length, position class, locality class, rare-only).
Score on the discovery half of LA pairs (split by word-type hash), keep the top 1% by |z_sound - z_shape|,
re-test on the held-out half. Controls: the same search on LA pairs with partners shuffled (degree kept),
and on Linear B pairs subsampled to LA size. Then pair-level guesses: LA confusions with high blind-row
co-assignment that are not same-consonant under LB values (outside check only)."""
import sys, os, json, pickle, hashlib, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import *

rng = np.random.default_rng(3203)
out = []
def log(s):
    print(s, flush=True); out.append(s)
M = pickle.load(open(os.path.join(CK, 'mats.pkl'), 'rb'))
NH = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

def row_mats(tag, signs):
    idx = {s: i for i, s in enumerate(signs)}
    mats = []
    for f in sorted(glob.glob(os.path.join(DATA, 'la21_ckpt', f'c*_{tag}.npz'))):
        z = np.load(f); rs = [str(s).lower() if tag == 'LB' else str(s) for s in z['signs']]
        P = (z['rows'][:, :, None] == z['rows'][:, None, :]).mean(0)
        S = np.full((len(signs),) * 2, np.nan)
        ii = [idx.get(s, -1) for s in rs]
        for a, ia in enumerate(ii):
            if ia < 0: continue
            for b, ib in enumerate(ii):
                if ib >= 0 and ia != ib: S[ia, ib] = P[a, b]
        mats.append(S)
    return mats

def pair_pool(words, group):
    E = one_sign_pairs(words, 2, group, 'all')
    for e in E:
        e['h'] = int(hashlib.md5(('-'.join(sorted(['-'.join(e['w1']), '-'.join(e['w2'])]))).encode()).hexdigest(), 16) % 2
    return E

def random_hyp():
    w = rng.dirichlet(np.ones(4) * 0.7)
    return dict(w=w, runs=None, thr=rng.choice([None, 0.3, 0.5]), minlen=int(rng.choice([2, 3, 3, 4])),
                pos=str(rng.choice(['all', 'final', 'nonfinal'])), loc=str(rng.choice(['any', 'local', 'rare'])))

def filt(E, h):
    out_ = []
    for e in E:
        if e['L'] < h['minlen']: continue
        if h['pos'] == 'final' and e['pos'] != e['L'] - 1: continue
        if h['pos'] == 'nonfinal' and e['pos'] == e['L'] - 1: continue
        if h['loc'] == 'local' and not (e['samedoc'] or e['samescribe']): continue
        if h['loc'] == 'rare' and not e['rare']: continue
        out_.append(e)
    return out_

def run_search(name, E, SM, rowm, idx, nh):
    R = {k: rank_norm(SM[k]) for k in ('blur', 'chamfer', 'hog', 'prof')}
    keys = list(R)
    hyps = []
    for k in range(nh):
        h = random_hyp()
        sub = rng.choice(len(rowm), size=rng.integers(1, len(rowm) + 1), replace=False)
        S = np.nanmean(np.stack([rowm[i] for i in sub]), 0)
        if h['thr'] is not None: S = np.where(np.isnan(S), np.nan, (S >= h['thr']).astype(float))
        V = sum(h['w'][i] * R[keys[i]] for i in range(4))
        Ef = filt(E, h)
        d0 = [e for e in Ef if e['h'] == 0]; d1 = [e for e in Ef if e['h'] == 1]
        if len(d0) < 12 or len(d1) < 12: continue
        zs0 = score_edges(d0, S, idx, 150, rng)['z']; zv0 = score_edges(d0, V, idx, 150, rng)['z']
        hyps.append(dict(h=h, S=S, V=V, d1=d1, n0=len(d0), zs0=zs0, zv0=zv0, gap0=zs0 - zv0))
    hyps = [x for x in hyps if np.isfinite(x['gap0'])]
    g = np.array([x['gap0'] for x in hyps])
    top = sorted(hyps, key=lambda x: -abs(x['gap0']))[:max(10, len(hyps) // 100)]
    held = []
    for x in top:
        zs1 = score_edges(x['d1'], x['S'], idx, 1000, rng)['z']; zv1 = score_edges(x['d1'], x['V'], idx, 1000, rng)['z']
        held.append((x['gap0'], zs1 - zv1, zs1, zv1, x['zs0'], x['zv0'], len(x['d1']), x['h']))
    rep = np.mean([np.sign(a) == np.sign(b) for a, b, *_ in held])
    snd_frac = np.mean(g > 0)
    log(f'  {name:22s} hyps {len(hyps)}; discovery: mean z sound {np.mean([x["zs0"] for x in hyps]):.2f}, mean z shape {np.mean([x["zv0"] for x in hyps]):.2f}, '
        f'share sound>shape {snd_frac:.2f}; top 1% ({len(top)}) held-out: sign of gap replicates {rep:.2f}, '
        f'mean held-out z sound {np.mean([h_[2] for h_ in held]):.2f}, shape {np.mean([h_[3] for h_ in held]):.2f}')
    return dict(n=len(hyps), snd_frac=float(snd_frac), rep=float(rep), held=[h_[:7] for h_ in held],
                zs=float(np.mean([x['zs0'] for x in hyps])), zv=float(np.mean([x['zv0'] for x in hyps])),
                heldzs=float(np.mean([h_[2] for h_ in held])), heldzv=float(np.mean([h_[3] for h_ in held])))

la_signs = M['la_signs']; iLA = {s: i for i, s in enumerate(la_signs)}
lb_signs = M['lb_signs']; iLB = {s: i for i, s in enumerate(lb_signs)}
rmLA = row_mats('LA', la_signs); rmLB = row_mats('LB', lb_signs)
log(f'# LA-32 cycle 3: random guessing, {NH} hypotheses per corpus (LA row runs {len(rmLA)}, LB {len(rmLB)})')
WA = la_words(); EA = pair_pool(WA, None)
res = {}
res['LA'] = run_search('LA real', EA, M['SM_LA'], rmLA, iLA, NH)
# control 1: LA with partners shuffled (degree kept)
Esh = [dict(e) for e in EA]; bs = [e['b'] for e in Esh]; rng.shuffle(bs)
for e, b in zip(Esh, bs): e['b'] = b
res['LAshuf'] = run_search('LA partner-shuffled', Esh, M['SM_LA'], rmLA, iLA, NH)
# control 2: LB at LA size (random subsample of pairs, ~ LA's pool size), 2 draws
EB = pair_pool(lb_words(), 'site')
for d in range(2):
    sub = [EB[i] for i in rng.choice(len(EB), len(EA), replace=False)]
    res[f'LBsize{d}'] = run_search(f'LB LA-size draw {d}', sub, M['SM_LB'], rmLB, iLB, NH // 2)

# pair-level guesses on LA (all pairs >= 2 signs)
SND = np.nanmean(np.stack(rmLA), 0); V = combine(M['SM_LA'], ['blur', 'chamfer', 'hog', 'prof'])
cnt = collections.Counter()
ex = collections.defaultdict(list)
for e in EA:
    if e['a'] in iLA and e['b'] in iLA:
        k = tuple(sorted((e['a'], e['b']))); cnt[k] += 1
        if len(ex[k]) < 3: ex[k].append('-'.join(e['w1']) + '/' + '-'.join(e['w2']))
log('\n## LA sign pairs confused >= 3 times (word pairs >= 2 signs), with blind-row co-assignment, shape rank, LB-value relation (outside check)')
rows = []
for k, n in cnt.most_common():
    if n < 3: break
    i, j = iLA[k[0]], iLA[k[1]]
    ca, cb = lb_cv(k[0].lower()), lb_cv(k[1].lower())
    rel = '-' if not (ca and cb) else ('C' if ca[0] == cb[0] else '') + ('V' if ca[1] == cb[1] else '')
    rows.append((k, n, SND[i, j], V[i, j], rel or 'none', ex[k]))
# null for counts: expected count of a pair under degree-preserving shuffle
stubs = np.array([iLA[x] for e in EA if e['a'] in iLA and e['b'] in iLA for x in (e['a'], e['b'])])
m = len(stubs) // 2; NC = collections.Counter(); reps = 500
for r in range(reps):
    rng.shuffle(stubs)
    for a, b in zip(stubs[:m], stubs[m:]):
        if a != b: NC[tuple(sorted((la_signs[a], la_signs[b])))] += 1
from scipy.stats import poisson
log(f'  (only pairs with Poisson P(count >= n | degree-kept null) < 0.05 shown; {len(rows)} pairs with n >= 3 tested)')
for k, n, s, v, rel, exs in rows:
    mu = NC[k] / reps
    if poisson.sf(n - 1, max(mu, 0.05)) >= 0.05: continue
    log(f'  P {poisson.sf(n - 1, max(mu, 0.05)):.4f} ' + f'  {k[0]:>5s}~{k[1]:<5s} n {n:2d} (null {mu:4.1f})  blind-row {s if np.isfinite(s) else float("nan"):.2f}  shape-rank {v:.2f}  LB-rel* {rel:4s}  e.g. {", ".join(exs)}')
res['pairs'] = [(list(k), n, NC[k] / reps, float(poisson.sf(n - 1, max(NC[k] / reps, 0.05))), float(s) if np.isfinite(s) else None, float(v), rel) for k, n, s, v, rel, _ in rows]
json.dump(res, open(os.path.join(CK, 'c3_res.json'), 'w'), indent=1, default=str)
open(os.path.join(CK, 'c3_report.txt'), 'w').write('\n'.join(out))
