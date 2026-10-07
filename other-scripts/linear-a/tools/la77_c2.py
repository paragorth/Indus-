"""LA-77 cycle 2: phylogeography without dates.

(a) Rooting rules (random a*log tokens + b*n sites + c*length + d*noise) selected on 20 planted
    diffusion worlds with known activation order, re-tested on 20 held-out planted worlds.
(b) Distance decay of tree transmissions (mean log km, and sea travel time, of cross-site
    parent->child links) vs 400 site-label permutations within support; same on 8 within-site sign
    shuffles, 16 planted distance-diffusion worlds (power) and 16 planted no-distance worlds (size).
(c) Site outflow (net parent-site share) with surviving rules; per-site z vs permutation null;
    split-half reproducibility; FROZEN with sha256 before any deposit phase is read (cycle 3).
"""
import sys, json, time
import numpy as np
from scipy.stats import spearmanr
from la77_geo import *

H = 400
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PW = dict(N0=1200, coin=40, zexp=0.5, q=0.35, mu=0.4)
t0 = time.time()
rng = np.random.default_rng(7702)
docs = load_docs()
codes = sorted({d['site'] for d in docs if d['words']})
docs = [d for d in docs if d['site'] in codes]
S = len(codes)
km = km_matrix(codes)
TR = json.load(open(os.path.join(DATA, 'la31', 'travel.json')))
ti = [TR['codes'].index(c) for c in codes]
sea = np.array(TR['windsea_annual'])[np.ix_(ti, ti)]; sea = (sea + sea.T) / 2
DL = {'km': np.log(km + 1), 'sea': np.log(sea + 1)}
mask = 1 - np.eye(S)
NR = int(os.environ.get("LA77_NR", 3000))
RULES = rng.normal(0, 1, (NR, 4))


def outflow(M):
    off = M * mask
    out, inn = off.sum(1), off.sum(0)
    return np.where(out + inn > 0, (out - inn) / np.maximum(out + inn, 1e-12), 0.0), out + inn


def with_rules(eng, X, rule_idx, famsel=None):
    """M averaged over hypotheses, hypothesis h uses tree h % H and rule RULES[r]."""
    saved = eng.root_coef.copy()
    M = np.zeros((S, S))
    for k, r in enumerate(rule_idx):
        h = k % eng.H
        eng.root_coef[h] = RULES[r]
        M += eng.transmissions(X, hyps=[h], famsel=famsel)
    eng.root_coef[:] = saved
    return M / len(rule_idx)


def dd_stats(M):
    off = (M + M.T) * mask
    return {k: float((off * D).sum() / max(off.sum(), 1e-12)) for k, D in DL.items()}, float(np.trace(M) / max(M.sum(), 1e-12))


def perm_test(eng, docs_, rule_idx, nperm, rr):
    X = eng.X(docs_)
    M = with_rules(eng, X, rule_idx)
    obs, loc = dd_stats(M)
    O, mass = outflow(M)
    nd = {k: [] for k in DL}; nl = []; NO = []
    for p in range(nperm):
        Xp = eng.X(permute_sites(docs_, rr))
        Mp = with_rules(eng, Xp, rule_idx)
        d, l = dd_stats(Mp)
        for k in DL:
            nd[k].append(d[k])
        nl.append(l); NO.append(outflow(Mp)[0])
    res = {}
    for k in DL:
        a = np.array(nd[k])
        res[k] = dict(obs=obs[k], null=float(a.mean()), sd=float(a.std()),
                      z=float((obs[k] - a.mean()) / max(a.std(), 1e-12)),
                      p_lower=float((np.sum(a <= obs[k]) + 1) / (len(a) + 1)))
    nl = np.array(nl)
    res['local'] = dict(obs=loc, null=float(nl.mean()), p_upper=float((np.sum(nl >= loc) + 1) / (len(nl) + 1)))
    NO = np.array(NO)
    res['O'] = O.tolist(); res['mass'] = mass.tolist()
    res['Oz'] = ((O - NO.mean(0)) / np.maximum(NO.std(0), 1e-9)).tolist()
    return res


log = []


def emit(x):
    log.append(x); print(json.dumps(x)[:900], round(time.time() - t0), flush=True)
    json.dump(log, open(os.path.join(CKPT, 'c2.json'), 'w'), indent=1)


# ---------------- (a) rooting rules on planted worlds
worlds = []
for w in range(int(os.environ.get("LA77_NW", 40))):
    pw, t = plant_world(docs, codes, km, rng, L=60, **PW)
    Tp, _ = type_table(pw)
    eng = Engine(Tp, 60, rng, codes, ntop_pool=3000)
    X = eng.X(pw)
    sc = np.zeros(NR)
    for r in range(NR):
        M = with_rules(eng, X, [r])
        O, mass = outflow(M)
        ok = mass > 0
        sc[r] = spearmanr(O[ok], -t[ok])[0] if ok.sum() > 3 else 0
    worlds.append(np.nan_to_num(sc))
    if w % 5 == 4:
        print('world', w, round(time.time() - t0), flush=True)
WS = np.array(worlds)
hw = len(WS) // 2; tr, te = WS[:hw].mean(0), WS[hw:].mean(0)
surv = np.argsort(-tr)[:NR // 20]
named = {'more sites': np.array([0, 1, 0, 0]), 'more tokens': np.array([1, 0, 0, 0]),
         'shorter': np.array([0, 0, -1, 0]), 'longer': np.array([0, 0, 1, 0])}
emit(dict(step='rooting rules', n_rules=NR, train_best=float(tr.max()), heldout_surv=float(te[surv].mean()),
          heldout_all=float(te.mean()), heldout_surv_pct=float(np.mean(te < te[surv].mean())),
          surv_coef_mean=RULES[surv].mean(0).tolist(),
          corr_coef_score=[float(np.corrcoef(RULES[:, k], te)[0, 1]) for k in range(4)]))
np.save(os.path.join(CKPT, 'c2_rules_surv.npy'), surv)
RULE_IDX = list(surv[:H])

# ---------------- (b, c) real data
T, _ = type_table(docs)
eng = Engine(T, H, rng, codes)
real = perm_test(eng, docs, RULE_IDX, NPERM, np.random.default_rng(11))
emit(dict(step='real', codes=codes, **real))
# no-HT version (drop pairs involving HT): recompute stats from same M
X = eng.X(docs)
M = with_rules(eng, X, RULE_IDX)
hti = codes.index('HT')
m2 = mask.copy(); m2[hti, :] = 0; m2[:, hti] = 0
off = (M + M.T) * m2
noht_obs = float((off * DL['km']).sum() / off.sum())
nn = []
rr = np.random.default_rng(12)
for p in range(min(NPERM, 200)):
    Mp = with_rules(eng, eng.X(permute_sites(docs, rr)), RULE_IDX)
    o = (Mp + Mp.T) * m2; nn.append(float((o * DL['km']).sum() / max(o.sum(), 1e-12)))
nn = np.array(nn)
emit(dict(step='real no-HT pairs km', obs=noht_obs, null=float(nn.mean()), p_lower=float((np.sum(nn <= noht_obs) + 1) / (len(nn) + 1))))
# split-half reproducibility of outflow
F = eng.F; rep = []; repn = []
for s in range(30):
    pf = rng.permutation(len(F)); a, b = pf[:len(F) // 2], pf[len(F) // 2:]
    Oa, ma = outflow(with_rules(eng, X, RULE_IDX[:100], famsel=a))
    Ob, mb = outflow(with_rules(eng, X, RULE_IDX[:100], famsel=b))
    ok = (ma > 0) & (mb > 0)
    rep.append(spearmanr(Oa[ok], Ob[ok])[0] if ok.sum() > 3 else np.nan)
    Xp = eng.X(permute_sites(docs, rng))
    Oa, ma = outflow(with_rules(eng, Xp, RULE_IDX[:100], famsel=a))
    Ob, mb = outflow(with_rules(eng, Xp, RULE_IDX[:100], famsel=b))
    ok = (ma > 0) & (mb > 0)
    repn.append(spearmanr(Oa[ok], Ob[ok])[0] if ok.sum() > 3 else np.nan)
emit(dict(step='outflow split-half', real=float(np.nanmean(rep)), null=float(np.nanmean(repn)),
          real_sd=float(np.nanstd(rep)), null_sd=float(np.nanstd(repn))))

# ---------------- FREEZE directions (no phases read anywhere in this script)
frozen = dict(codes=codes, outflow=[round(x, 6) for x in real['O']], outflow_z=[round(x, 6) for x in real['Oz']],
              mass=[round(x, 6) for x in real['mass']],
              rank_source_first=[codes[i] for i in np.argsort(-np.array(real['Oz']))],
              note='LA-77 inferred site outflow (parent-site share of tree transmissions), frozen before deposit phases are read')
h = sha(frozen)
json.dump(dict(frozen, sha256=h), open(os.path.join(OUT, 'frozen_direction.json'), 'w'), indent=1)
open(os.path.join(OUT, 'frozen_direction.sha256'), 'w').write(h + '\n')
emit(dict(step='frozen', sha256=h, rank=frozen['rank_source_first'][:10]))

# ---------------- controls
for s in range(8):
    sd = shuffle_within_site(docs, np.random.default_rng(900 + s))
    e2 = Engine(type_table(sd)[0], 150, rng, codes, ntop_pool=5000)
    r = perm_test(e2, sd, RULE_IDX[:150], 100, np.random.default_rng(13 + s))
    emit(dict(step='shuffled-within-site', i=s, km=r['km'], sea=r['sea'], local=r['local']))
for L, lab in ((60, 'planted distance L60'), (1e9, 'planted no-distance')):
    for s in range(16):
        pw, t = plant_world(docs, codes, km, rng, L=L, **PW)
        e2 = Engine(type_table(pw)[0], 150, rng, codes, ntop_pool=5000)
        r = perm_test(e2, pw, RULE_IDX[:150], 100, np.random.default_rng(50 + s))
        O = np.array(r['Oz']); ok = np.array(r['mass']) > 0
        emit(dict(step=lab, i=s, km=r['km'], local=r['local'],
                  dir_rho=float(spearmanr(O[ok], -t[ok])[0])))
print('done', time.time() - t0)
