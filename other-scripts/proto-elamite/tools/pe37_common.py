"""pe37 shared code: THE BONES ARE A RULER FOR THE ANIMAL SIGNS.

Outside data: animal bones from Tal-e Malyan (Banesh levels, c. 3400-2600 BCE),
M. A. Zeder's specimen records published on Open Context ("Tal-e Malyan
Zooarchaeology", https://opencontext.org/projects/d6b25ec9-2884-4e3c-00e8-0c5a6472fa63;
export tables 394c525a-151a-7107-e310-55b48f304fd3 and 6c6d9656-ad63-14c0-b24f-4153086c3620).
Only counts derived from those records are stored in the repo (data/pe37_bone_targets.json).

Pipeline
  bones  -> epiphyseal fusion counts (early / middle / late groups), pelvis sexing,
            Ovis vs Capra and Bos vs caprine NISP
  herd demography (ABC): random sex-specific survivorship models at steady state
            (offtake keeps the herd size constant), accepted if their kill-off
            reproduces the fusion and sexing counts; each accepted model gives the
            LIVING herd's young share and adult-female share
  signs  -> each herd sign is assigned a class (none, sheep F/M/Y, goat F/M/Y, cattle);
            the implied herd composition (sum of the counts per class) is scored by
            the bone-derived target densities (Beta fits to the ABC / NISP draws)
  search -> exhaustive enumeration when small, else random draws + annealing
"""
import json, math, os, sys, csv, glob
from collections import Counter
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe37_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
TARGETS = os.path.join(DATA, 'pe37_bone_targets.json')

CLASSES = ['none', 'sF', 'sM', 'sY', 'gF', 'gM', 'gY', 'cat']
NCL = len(CLASSES)

# ---------------------------------------------------------------- bones
FUSION_GROUPS = {
    'early': [('Scapula', 'D'), ('Humerus', 'D'), ('Radius', 'P'), ('Ilium', 'P'),
              ('Innominate', 'P'), ('Ilium-Ischium', 'P'), ('Ischium', 'P'), ('Pubis', 'P')],
    'middle': [('1st Phalanx', 'P'), ('2nd Phalanx', 'P'), ('Tibia', 'D'),
               ('Metacarpal III & IV', 'D'), ('Metatarsal III & IV', 'D'),
               ('Metapodial III & IV', 'D')],
    'late': [('Ulna', 'P'), ('Femur', 'P'), ('Femur', 'D'), ('Humerus', 'P'),
             ('Radius', 'D'), ('Tibia', 'P'), ('Calcaneus', 'P')],
}
# approximate fusion ages for caprines (years); sampled within these ranges
FUSION_AGE = {'early': (0.5, 0.85), 'middle': (1.1, 2.3), 'late': (2.5, 3.6)}
CAPRINE = ('Sheep/goat', 'Ovis', 'Capra')


def malyan_rows():
    rows = []
    for f in glob.glob(os.path.join(SCRATCH, 'pe37', 'mal_*.csv')):
        rows += list(csv.DictReader(open(f, encoding='utf-8')))
    return rows


def _fz(x):
    if x in ('Fused', 'Fusing'):
        return 1
    if x == 'Unfused':
        return 0
    return None


def bone_counts(rows, early_date='-3400.0', unit=None):
    b = [r for r in rows if r['Early Date (BCE/CE)'] == early_date
         and (unit is None or r['Context (4)'][:7] == unit)]
    tax = Counter(r['Has Biological Taxonomy [Label]'] for r in b)
    cap = [r for r in b if r['Has Biological Taxonomy [Label]'] in CAPRINE]
    fus = {}
    for g, els in FUSION_GROUPS.items():
        f = u = 0
        for r in cap:
            for el, end in els:
                if r['Element'] == el:
                    s = _fz(r['Proximal Fusion'] if end == 'P' else r['Distal Fusion'])
                    if el == 'Scapula' and s is None:
                        s = _fz(r['Proximal Fusion'])
                    if s == 1:
                        f += 1
                    elif s == 0:
                        u += 1
        fus[g] = [f, u]
    sx = Counter(r['Sex'] for r in cap)
    return {'n_specimens': len(b), 'taxa': dict(tax), 'fusion_caprine': fus,
            'pelvis_sex': {'F': sx['Female'] + sx['Possibly female'],
                           'M': sx['Male'] + sx['Possibly male']},
            'ovis': tax['Ovis'], 'capra': tax['Capra'], 'bos': tax['Bos'],
            'caprine': sum(tax[c] for c in CAPRINE),
            'equid': tax['Equus; Equidae'] + tax['Equus'],
            'pig': tax['Sus scrofa; Sus scrofa domesticus']}


# ---------------------------------------------------------------- herd demography
AGES = 12          # annual age classes 0..11


def herd_model(rng):
    """Random steady-state herd. Returns dict with female/male survivorship l(t) at
    yearly nodes, or None. Hazards are constant within a year."""
    sf = np.empty(AGES)
    sm = np.empty(AGES)
    sf[0] = rng.uniform(0.45, 0.97)        # first-year survival (deaths incl. culls)
    sm[0] = rng.uniform(0.15, 0.97)
    sf[1] = rng.uniform(0.6, 0.98)
    sm[1] = rng.uniform(0.05, 0.97)
    sf[2] = rng.uniform(0.6, 0.98)
    sm[2] = rng.uniform(0.05, 0.97)
    fa, ma = rng.uniform(0.55, 0.95), rng.uniform(0.3, 0.95)
    sf[3:] = fa
    sm[3:] = ma
    sf[-1] = sm[-1] = 0.0                 # nobody survives past AGES
    lf = np.concatenate([[1.0], np.cumprod(sf)])   # l at t = 0..AGES
    lm = np.concatenate([[1.0], np.cumprod(sm)])
    return lf, lm


def l_at(l, t):
    """Survivorship at fractional age t (constant hazard within each year)."""
    k = int(math.floor(t))
    if k >= len(l) - 1:
        return 0.0
    a, b = l[k], l[k + 1]
    if a <= 0:
        return 0.0
    if b <= 0:
        return a * (1 - (t - k))   # linear for the last (closing) year
    return a * (b / a) ** (t - k)


def area(l, t0, t1, n=None):
    """Exact person-years between integer ages t0 and t1 under l_at."""
    s = 0.0
    for k in range(int(t0), int(t1)):
        if k >= len(l) - 1:
            break
        a, b = l[k], l[k + 1]
        if a <= 0:
            break
        if b <= 0:
            s += a / 2
        elif abs(b - a) < 1e-12:
            s += a
        else:
            s += (b - a) / math.log(b / a)
    return s


def demography_abc(bc, n=200000, rng=None, eff=0.1, seed=1):
    """ABC by importance weights: each random herd model is weighted by the
    (over-dispersed) binomial likelihood of the fusion and pelvis-sex counts.
    eff: effective sample fraction (bones are not independent animals).
    Births per adult female (from age 1) are set by the steady-state condition and
    must fall in 0.5-1.8 per year. Returns arrays of living-herd statistics and weights."""
    rng = rng or np.random.default_rng(seed)
    out = []
    fus = bc['fusion_caprine']
    ps = bc['pelvis_sex']
    for _ in range(n):
        lf, lm = herd_model(rng)
        # steady state: female births x female adult-years = 1 (half the young are female)
        fyears = area(lf, 1.0, AGES, 30)
        b = 2.0 / fyears if fyears > 0 else 99
        if not (0.5 <= b <= 1.8):
            continue
        ll = 0.0
        tau = {g: rng.uniform(*FUSION_AGE[g]) for g in FUSION_AGE}
        for g, (f, u) in fus.items():
            p = 0.5 * (l_at(lf, tau[g]) + l_at(lm, tau[g]))   # bones of animals dying after tau
            p = min(max(p, 1e-6), 1 - 1e-6)
            ll += eff * (f * math.log(p) + u * math.log(1 - p))
        # adult (>= 1 yr) deaths by sex (pelvis sexing works on adults)
        dF, dM = l_at(lf, 1.0), l_at(lm, 1.0)
        q = dF / (dF + dM) if dF + dM > 0 else 0.5
        q = min(max(q, 1e-6), 1 - 1e-6)
        ll += eff * (ps['F'] * math.log(q) + ps['M'] * math.log(1 - q))
        yF, yM = area(lf, 0, 1, 20), area(lm, 0, 1, 20)
        aF, aM = area(lf, 1, AGES, 30), area(lm, 1, AGES, 30)
        tot = yF + yM + aF + aM
        out.append((ll, (yF + yM) / tot, aF / (aF + aM), b))
    A = np.array(out)
    w = np.exp(A[:, 0] - A[:, 0].max())
    w /= w.sum()
    return {'young': A[:, 1], 'adultF': A[:, 2], 'births': A[:, 3], 'w': w,
            'ess': float(1.0 / np.sum(w ** 2)), 'n_valid': len(A)}


def wstats(x, w):
    m = float(np.sum(w * x))
    v = float(np.sum(w * (x - m) ** 2))
    return m, math.sqrt(v)


def beta_from(m, sd):
    m = min(max(m, 1e-3), 1 - 1e-3)
    v = min(sd ** 2, m * (1 - m) * 0.95)
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


# ---------------------------------------------------------------- composition scoring
def class_sums(A, t):
    """A: (N, K) int classes, t: (K,) totals -> (N, NCL) class sums."""
    S = np.zeros((A.shape[0], NCL))
    for c in range(NCL):
        S[:, c] = (A == c) @ t
    return S


def proportions(S):
    sh = S[:, 1] + S[:, 2] + S[:, 3]
    go = S[:, 4] + S[:, 5] + S[:, 6]
    cap = sh + go
    adF = S[:, 1] + S[:, 4]
    ad = adF + S[:, 2] + S[:, 5]
    with np.errstate(invalid='ignore', divide='ignore'):
        p_sheep = sh / cap
        p_young = (S[:, 3] + S[:, 6]) / cap
        p_adF = adF / ad
        p_cat = S[:, 7] / (cap + S[:, 7])
    return p_sheep, p_young, p_adF, p_cat


def score_props(props, target, use=('sheep', 'young', 'adF')):
    """Sum of log Beta densities of the implied proportions. Invalid (nan, 0 or 1
    where the Beta has no mass) -> -inf. target: {name: (a, b)}."""
    p_sheep, p_young, p_adF, p_cat = props
    P = {'sheep': p_sheep, 'young': p_young, 'adF': p_adF, 'cat': p_cat}
    s = np.zeros(len(p_sheep))
    for k in use:
        a, b = target[k]
        x = np.clip(P[k], 1e-9, 1 - 1e-9)
        v = stats.beta.logpdf(x, a, b)
        v[np.isnan(P[k])] = -np.inf
        s += v
    return s


def all_assignments(K, classes):
    """Every assignment of K signs to the given class ids (int array (C^K, K))."""
    cl = np.asarray(classes)
    C = len(cl)
    idx = np.arange(C ** K)
    A = np.empty((C ** K, K), dtype=np.int8)
    for j in range(K):
        A[:, j] = cl[(idx // C ** (K - 1 - j)) % C]
    return A


def enumerate_scores(t, target, classes=(0, 1, 2, 3, 4, 5, 6), use=('sheep', 'young', 'adF'),
                     A=None):
    K = len(t)
    if A is None:
        A = all_assignments(K, classes)
    sc = np.empty(len(A))
    step = 1 << 20
    for i in range(0, len(A), step):
        S = class_sums(A[i:i + step], np.asarray(t, float))
        sc[i:i + step] = score_props(proportions(S), target, use)
    return A, sc


def posterior_marginals(A, sc, K):
    ok = np.isfinite(sc)
    w = np.exp(sc[ok] - sc[ok].max())
    w /= w.sum()
    M = np.zeros((K, NCL))
    for j in range(K):
        M[j] = np.bincount(A[ok, j], weights=w, minlength=NCL)
    return M, w


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(path, 'w'), indent=1, default=conv)
