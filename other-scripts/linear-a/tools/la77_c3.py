"""LA-77 cycle 3: open the dates.

(1) Check the sha256 of the frozen outflow vector, then read deposit phases (context field of the
    corpus, la17 coding) for the first time. Test: sources older? Spearman(outflow z, -phase),
    exact phase permutation over sites (20,000).
(2) Directed cross-phase transmissions: share of weight from the older to the younger site,
    fresh engine, vs 300 site-label permutations within support.
(3) Phase gap beyond distance (replicates la17 grade C on mutant links instead of shared words):
    symmetric transmission z per site pair regressed on log km and |phase gap|; QAP null.
(4) Linear B positive control (DAMOS; KN, PY, TH, MY, TI, KH): same distance decay and outflow;
    KN (LM II-IIIA1) should be the source if the pipeline reads direction.
(5) Within-site sign shuffles (8): their frozen-style outflow vs phase, as a control on (1).
"""
import sys, json, time, math
import numpy as np
from scipy.stats import spearmanr
from la77_geo import *
from la31_common import LB_SITES

t0 = time.time()
log = []


def emit(x):
    log.append(x); print(json.dumps(x)[:900], round(time.time() - t0), flush=True)
    json.dump(log, open(os.path.join(CKPT, 'c3.json'), 'w'), indent=1)


fz = json.load(open(os.path.join(OUT, 'frozen_direction.json')))
h_stored = open(os.path.join(OUT, 'frozen_direction.sha256')).read().strip()
body = {k: v for k, v in fz.items() if k != 'sha256'}
assert sha(body) == h_stored == fz['sha256'], 'frozen file altered'
emit(dict(step='sha check', ok=True, sha256=h_stored))
codes = fz['codes']; S = len(codes)
Oz = np.array(fz['outflow_z']); mass = np.array(fz['mass'])

# ---- phases (first read)
PHASE = {'MMIA': 0, 'MMII': 1, 'MMIII': 2, 'MMIIIA': 2, 'MMIIIB': 2.25, 'LMIA': 3, 'LMI': 3.5, 'LMIB': 4}
docs = load_docs()
ph = collections.defaultdict(list)
for d in docs:
    if d['context'] in PHASE:
        ph[d['site']].append(PHASE[d['context']])
phase = np.array([np.median(ph[c]) if c in ph else np.nan for c in codes])
emit(dict(step='phases', phase={c: (None if np.isnan(p) else float(p)) for c, p in zip(codes, phase)},
          n={c: len(ph[c]) for c in codes}))
ok = ~np.isnan(phase) & (mass > 0)
rho = spearmanr(Oz[ok], -phase[ok])[0]
rr = np.random.default_rng(31)
nul = np.array([spearmanr(Oz[ok], -rr.permutation(phase[ok]))[0] for _ in range(20000)])
emit(dict(step='outflow vs phase (frozen)', n_sites=int(ok.sum()), rho=float(rho),
          p_upper=float((np.sum(nul >= rho) + 1) / (len(nul) + 1)),
          sites=[(codes[i], float(Oz[i]), float(phase[i])) for i in np.where(ok)[0]]))
# leave-one-site-out stability
loo = []
for i in np.where(ok)[0]:
    k = ok.copy(); k[i] = False
    loo.append((codes[i], float(spearmanr(Oz[k], -phase[k])[0])))
emit(dict(step='leave-one-out rho', loo=loo))

# ---- (2) directed cross-phase share, fresh engine
docs = [d for d in docs if d['site'] in codes]
rng = np.random.default_rng(7703)
T, _ = type_table(docs)
eng = Engine(T, 300, rng, codes)
surv = np.load(os.path.join(CKPT, 'c2_rules_surv.npy'))
rules = np.random.default_rng(7702).normal(0, 1, (int(os.environ.get('LA77_NR', 3000)), 4))
# NB: c2 drew RULES as the first draw of default_rng(7702) -> identical bank
for h in range(eng.H):
    eng.root_coef[h] = rules[surv[h % len(surv)]]
dphase = phase[:, None] - phase[None, :]          # >0: row younger than column


def older_share(M):
    sel = ~np.isnan(dphase) & (dphase != 0)
    fw = (M * ((dphase < 0) & sel)).sum()          # parent older than child
    bw = (M * ((dphase > 0) & sel)).sum()
    return fw / max(fw + bw, 1e-12)


X = eng.X(docs)
M = eng.transmissions(X)
obs = older_share(M)
nn = np.array([older_share(eng.transmissions(eng.X(permute_sites(docs, rr)))) for _ in range(300)])
emit(dict(step='older->younger share', obs=float(obs), null=float(nn.mean()), sd=float(nn.std()),
          p_upper=float((np.sum(nn >= obs) + 1) / (len(nn) + 1))))

# ---- (3) phase gap beyond distance on symmetric mutant links
km = km_matrix(codes)
Msym = M + M.T
NP = [eng.transmissions(eng.X(permute_sites(docs, rr))) for _ in range(200)]
NS = np.array([m + m.T for m in NP])
Z = (Msym - NS.mean(0)) / np.maximum(NS.std(0), 1e-9)
iu = np.triu_indices(S, 1)
okp = ~np.isnan(dphase[iu]) & (NS.std(0)[iu] > 0)
y = Z[iu][okp]; A = np.column_stack([np.ones(okp.sum()), np.log(km[iu][okp] + 1), np.abs(dphase[iu][okp])])
beta = np.linalg.lstsq(A, y, rcond=None)[0]
qb = []
okc = ~np.isnan(phase)
idx = np.where(okc)[0]
for _ in range(5000):
    p = phase.copy(); p[idx] = rr.permutation(phase[idx])
    dp = np.abs(p[:, None] - p[None, :])[iu][okp]
    A2 = A.copy(); A2[:, 2] = dp
    qb.append(np.linalg.lstsq(A2, y, rcond=None)[0][2])
qb = np.array(qb)
emit(dict(step='phase gap beyond km (pair z)', n_pairs=int(okp.sum()), b_logkm=float(beta[1]), b_gap=float(beta[2]),
          p_lower_gap=float((np.sum(qb <= beta[2]) + 1) / (len(qb) + 1))))

# ---- (5) shuffled-sign controls through the same frozen-style pipeline
from la77_c2_lib import outflow_z_for
for s in range(8):
    sd = shuffle_within_site(docs, np.random.default_rng(900 + s))
    Ozs, ms = outflow_z_for(sd, codes, rules, surv, rng, nperm=100)
    k = ~np.isnan(phase) & (ms > 0)
    emit(dict(step='shuffled control outflow vs phase', i=s, rho=float(spearmanr(Ozs[k], -phase[k])[0])))

# ---- (4) Linear B positive control
from la15_common import load_lb
LBD = []
for d in load_lb():
    c = d['id'].split()[0]
    if c in LB_SITES:
        ws = [tuple(w.split('-')) for w in d['words']]
        ws = [w for w in ws if len(w) >= 2 and not any('*' in x for x in w)]
        LBD.append(dict(id=d['id'], site=c, support='tablet', context='', words=ws))
lbc = sorted(LB_SITES)
Tl, _ = type_table(LBD)
el = Engine(Tl, 150, rng, lbc, ntop_pool=4000)
for h in range(el.H):
    el.root_coef[h] = rules[surv[h % len(surv)]]


def havd(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    q = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(q))


kml = np.array([[havd(LB_SITES[a][1:], LB_SITES[b][1:]) for b in lbc] for a in lbc])
Dl = np.log(kml + 1); mk = 1 - np.eye(len(lbc))


def ddl(M):
    o = (M + M.T) * mk
    return float((o * Dl).sum() / max(o.sum(), 1e-12))


def outf(M):
    o = M * mk; a, b = o.sum(1), o.sum(0)
    return np.where(a + b > 0, (a - b) / np.maximum(a + b, 1e-12), 0)


Xl = el.X(LBD); Ml = el.transmissions(Xl)
obs = ddl(Ml); Ol = outf(Ml)
nd, no = [], []
for _ in range(150):
    Mp = el.transmissions(el.X(permute_sites(LBD, rr)))
    nd.append(ddl(Mp)); no.append(outf(Mp))
nd = np.array(nd); no = np.array(no)
Olz = (Ol - no.mean(0)) / np.maximum(no.std(0), 1e-9)
emit(dict(step='Linear B control', n_types=len(Tl), n_fam=len(el.F), km_obs=obs, km_null=float(nd.mean()),
          p_lower=float((np.sum(nd <= obs) + 1) / (len(nd) + 1)),
          outflow_z={c: float(v) for c, v in zip(lbc, Olz)},
          KN_rank=int(1 + np.sum(Olz > Olz[lbc.index('KN')]))))
print('done', time.time() - t0)
