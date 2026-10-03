#!/usr/bin/env python3
"""S-DARK-69: THE REGISTER MAP. How many distinct writing registers does the corpus contain, and what defines each?

Every complete text is described by TEXT-SIDE features only (no object information):
  opener (seal opener 817/861/820/692/920 | tablet opener 501/413/503 | none), connective W2/W60, markers W1 / W31 / W32,
  head class (S289 paradigm + W700/W595/mjar/none), suffix (W400 | W90/91 | none), numeral series (short | tall | both | none),
  fish word class (F220 | F240 | Fmod 235/233/231 | none), person sign (90/91/93/71 in the body), 12-grid (W55/56),
  quantity phrase (numeral + good), second unit (head sign medial), length bin.
A latent class model (multinomial mixture, EM, Dirichlet 0.5 smoothing) is fitted on these features alone; K is chosen by
5-fold held-out log-likelihood (K = 1..12, 8 restarts each), stability = mean adjusted Rand between restarts.
The arrow: the text-side classes predict the OBJECT-SIDE format (fine CISI type, shape, material, boss, sides) far above a
permutation null (AMI, labels permuted 1,000x), i.e. the code had format-specific sub-grammars.
Cycle 1  fit, K choice, stability, AMI / accuracy against object format; with and without the length feature.
Cycle 2  per-register card: defining features, size, site, chronology, designation-slot loyalty (pan-Indus pool vs register-
         specific inventory; null = register labels permuted within site x length-bin).
Cycle 3  transitions: two-faced objects, seals in tablet registers and tablets in seal registers, heads shared across
         registers, register structure by city (MD vs Harappa vs held-out sites; refit per city, ARI with the global map).
Cycle 4  IM77 through the bridge (M numbers): Wells-trained model applied to IM77 feature vectors; register -> recorded
         object type accuracy on all IM77 and on the 324 IM77-only texts (loop27_sets.json 'new') vs the modal baseline.
Usage: python3 tools/dark_loop69.py <1|2|3|4> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json, sys, csv, collections, math, re, os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, mutual_info_score as MI

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(69)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ---------------- sign classes (Wells) ----------------
HEAD_GROUP = {740: 'jar', 741: 'mjar', 742: 'mjar', 745: 'mjar', 520: 'arrow', 151: 'W151', 156: 'W156', 154: 'W154/158',
              158: 'W154/158', 527: 'box527', 526: 'box527', 226: 'W226', 617: 'W615/617', 615: 'W615/617', 236: 'W236',
              700: 'W700', 595: 'W595'}
SUF = {400: 'W400', 90: 'W90/91', 91: 'W90/91'}
SEAL_OP = {817, 861, 820, 920, 692}
TAB_OP = {503, 413, 501}
CONN = {2, 60}
PERSON = {90, 91, 93, 71}
GRID = {55, 56}
SHORT = set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30))
TALL = set(range(32, 40))
NUMS = (SHORT | TALL) - {1, 2, 31}
GOODS = {390, 405, 407, 900, 384, 388, 645, 904, 220}
FISH = {220: 'F220', 240: 'F240', 235: 'Fmod', 233: 'Fmod', 231: 'Fmod'}
FOREIGN_REGIONS = {'Persian Gulf', 'Mesopotamia', 'Iranian Plateau', 'Central Asia'}
# frame signs (not designation-slot elements)
FRAME_W = set(HEAD_GROUP) | set(SUF) | SEAL_OP | TAB_OP | CONN | {1, 31} | SHORT | TALL | set(FISH) | GRID | PERSON

# IM77 (M numbers)
M_HEAD = {342: 'jar', 343: 'mjar', 344: 'mjar', 345: 'mjar', 211: 'arrow', 12: 'W151', 15: 'W156', 254: 'box527',
          60: 'W226', 245: 'W615/617', 66: 'W236', 328: 'W700', 252: 'W595'}
M_SUF = {176: 'W400', 1: 'W90/91'}
M_SEAL_OP = {267, 391, 293, 150}
M_TAB_OP = {210, 177, 204}  # W501/W413/W503 through bridge_extended.json
M_CONN = {99, 100, 123}
M_PERSON = {1, 3, 2}
M_GRID = {121, 122}
M_TALL = set(range(87, 97)); M_SHORT = set(range(101, 121)); M_NUMS = (M_TALL | M_SHORT) - {86, 97, 98, 99, 100}
M_GOODS = {161, 162, 167, 168, 169, 287, 59}
M_FISH = {59: 'F220', 67: 'F240', 65: 'Fmod', 72: 'Fmod', 70: 'Fmod'}
M_MARK1 = {97, 98}; M_MARK31 = {86}; M_MARK32 = {87}
FRAME_M = set(M_HEAD) | set(M_SUF) | M_SEAL_OP | M_CONN | M_MARK1 | M_MARK31 | M_TALL | M_SHORT | set(M_FISH) | M_GRID | M_PERSON

def type_fine(t):
    t0 = t.split(':')[0]; sub = t.split(':')[1] if ':' in t else ''
    if t0 == 'SEAL':
        return {'S': 'seal_sq', 'R': 'seal_bar', 'C': 'seal_round', 'CY': 'seal_round', 'L': 'seal_round'}.get(sub, 'seal_oth')
    if t0 == 'TAB':
        return {'B': 'tab_mould', 'I': 'tab_inc', 'C': 'tab_cu'}.get(sub, 'tab_oth')
    if t0 == 'TAG': return 'sealing'
    if t0 in ('POT', 'POsT'): return 'pot'
    if t0 == 'BNGL': return 'bangle'
    if t0 == 'ROD': return 'rod'
    return 'misc'
def type_coarse(tf):
    if tf.startswith('seal'): return 'seal'
    if tf.startswith('tab'): return 'tablet'
    return tf
def lenbin(n): return '1' if n <= 1 else '2' if n == 2 else '3' if n == 3 else '4' if n == 4 else '5-6' if n <= 6 else '7+'
def material_group(m):
    m = (m or '').lower()
    if m in ('', '-'): return 'unknown'
    if 'steatite' in m: return 'steatite'
    if 'faience' in m or 'paste' in m or 'stoneware' in m: return 'faience'
    if 'clay' in m or 'terracotta' in m or 'ceramic' in m: return 'clay'
    if 'copper' in m or 'bronze' in m: return 'copper'
    if 'ivory' in m or 'bone' in m: return 'ivory/bone'
    return 'other'
def boss_group(b):
    if not b or b == '-': return 'unknown'
    return 'boss' if b.startswith('PB') or b == '?B' else 'noboss'
def phase_of(site, period, phase):
    if site == 'Mohenjo-daro':
        p = period.strip(); return 'MD-' + p if p in ('Early', 'Intermediate', 'Late') else None
    if site == 'Harappa':
        if period == '3' and phase in ('B', 'B/C', 'C'): return 'H-3' + phase
        if phase.startswith('Stratum'):
            s = phase.split()[1]; return 'H-Str-' + ('I-II' if s in ('I', 'II') else 'III-IV' if s in ('III', 'IV') else 'V-VI')
        return None
    if site == 'Dholavira': return 'Dlv-' + period if period in ('4', '5', '6') else None
    if site == 'Kalibangan': return 'K-' + phase if phase in ('Early', 'Middle', 'Late') else None
    return None

FEATS = ['opener', 'conn', 'W1', 'W31', 'W32', 'head', 'suffix', 'numser', 'fish', 'person', 'grid', 'quantity', 'second', 'len']
FEATS_NOLEN = [f for f in FEATS if f != 'len']

def features(seq, head_group, suf_map, seal_op, tab_op, conn, mark1, mark31, mark32, short, tall, nums, goods, fish, person, grid, frame):
    s = list(seq); suffix = []
    while len(s) > 1 and s[-1] in suf_map: suffix.append(s.pop())
    head = head_group.get(s[-1])
    if head is None:
        head = 'none'; body = s; head_sign = None
    else:
        head_sign = s[-1]; body = s[:-1]
    f = {}
    f['opener'] = 'seal_op' if (s and s[0] in seal_op) else 'tab_op' if (s and s[0] in tab_op) else 'none'
    f['conn'] = 'yes' if any(x in conn for x in s) else 'no'
    f['W1'] = 'yes' if any(x in mark1 for x in s) else 'no'
    f['W31'] = 'yes' if any(x in mark31 for x in s) else 'no'
    f['W32'] = 'yes' if any(x in mark32 for x in body) else 'no'
    f['head'] = head
    f['suffix'] = suf_map[suffix[-1]] if suffix else 'no'
    sh = any(x in short for x in body); ta = any(x in tall for x in body)
    f['numser'] = 'both' if sh and ta else 'short' if sh else 'tall' if ta else 'none'
    fl = [fish[x] for x in body if x in fish]
    f['fish'] = 'none' if not fl else ('F240' if 'F240' in fl else 'Fmod' if 'Fmod' in fl else 'F220')
    f['person'] = 'yes' if any(x in person for x in body) else 'no'
    f['grid'] = 'yes' if any(x in grid for x in s) else 'no'
    f['quantity'] = 'yes' if any(body[i] in nums and body[i + 1] in goods for i in range(len(body) - 1)) else 'no'
    f['second'] = 'yes' if any(x in head_group for x in body) else 'no'
    f['len'] = lenbin(len(seq))
    # designation-slot elements: body signs outside the frame classes
    f['mid_els'] = [x for x in body if x not in frame]
    f['head_sign'] = head_sign
    return f

def wells_feats(seq):
    return features(seq, HEAD_GROUP, SUF, SEAL_OP, TAB_OP, CONN, {1}, {31}, {32}, SHORT, TALL, NUMS, GOODS, FISH, PERSON, GRID, FRAME_W)
def im_feats(seq):
    return features(seq, M_HEAD, M_SUF, M_SEAL_OP, M_TAB_OP, M_CONN, M_MARK1, M_MARK31, M_MARK32, M_SHORT, M_TALL, M_NUMS, M_GOODS, M_FISH,
                    M_PERSON, M_GRID, FRAME_M)

def load_wells(level):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    g = lambda r, k: (r.get(k) or '')
    by_cisi = collections.defaultdict(list)
    for r in rows: by_cisi[r['cisi']].append(r)
    def parsed(r): return [x for x in (int(y) for y in re.findall(r'\d{3}', r['text'])) if x != 0]
    j = 0; T = []
    for c in C:
        while rows[j]['cisi'] != c['cisi']: j += 1
        r = rows[j]; j += 1
        s = c[level]
        if not s or c['complete'] != 'Y': continue
        raw = c['seq_raw']
        if list(reversed(parsed(r))) != raw:
            alt = [x for x in by_cisi[c['cisi']] if list(reversed(parsed(x))) == raw]
            r = alt[0] if alt else r
        tf = type_fine(c['type'])
        t = dict(id=g(r, 'id'), cisi=c['cisi'], site=c['site'], type=c['type'], tf=tf, tc=type_coarse(tf),
                 foreign=g(r, 'region') in FOREIGN_REGIONS, shape=c['shape'] if c['shape'] not in ('', '-') else 'unknown',
                 material=material_group(c['material']), boss=boss_group(g(r, 'boss')), sides=g(r, 'sides') or 'unknown',
                 phase=phase_of(c['site'], c['period'], c['phase']), seq=list(s), n=len(s),
                 big=c['site'] in ('Mohenjo-daro', 'Harappa'))
        t.update(wells_feats(t['seq']))
        T.append(t)
    return T

IM_TYPE = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tab_cu',
           'pottery graffito': 'pot', 'ivory/bone rod': 'rod', 'miscellaneous': 'misc', 'bronze implement': 'misc'}
def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    new = {tuple(x) for x in json.load(open(OUT + 'loop27_sets.json'))['new']}
    by = collections.defaultdict(list)
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        by[(r['text_no'], r['side'])].append(r)
    T = []
    for (tn, side), ls in by.items():
        ls.sort(key=lambda r: int(r['line']))
        seq = []
        for r in ls: seq += [int(x) for x in r['signs_clean'].split()]
        if 0 in seq or not seq: continue
        r = ls[0]
        t = dict(id=f'{tn}:{side}', text_no=tn, site=r['site'], tf=IM_TYPE.get(r['object_type'], 'misc'), seq=seq, n=len(seq),
                 foreign=r['site'] == 'West Asian finds', new=(tn, side) in new, big=r['site'] in ('Mohenjodaro', 'Harappa'))
        t['tc'] = t['tf']
        t.update(im_feats(seq))
        T.append(t)
    return T

# ---------------- latent class model ----------------
class LCA:
    def __init__(self, feats, K, alpha=0.5):
        self.feats = feats; self.K = K; self.alpha = alpha
    def encode(self, T):
        X = np.zeros((len(T), len(self.feats)), int)
        for j, f in enumerate(self.feats):
            for i, t in enumerate(T): X[i, j] = self.cats[j].get(t[f], -1)
        return X
    def set_cats(self, T):
        self.cats = []
        for f in self.feats:
            vals = sorted({t[f] for t in T}); self.cats.append({v: k for k, v in enumerate(vals)})
    def fit(self, T, n_iter=200, seed=0, X=None):
        r = np.random.default_rng(seed)
        if X is None: X = self.encode(T)
        N, F = X.shape; K = self.K
        resp = r.dirichlet(np.ones(K), N)
        prev = -np.inf
        for it in range(n_iter):
            self.pi = (resp.sum(0) + self.alpha) / (N + K * self.alpha)
            self.theta = []
            for j in range(F):
                C = len(self.cats[j]); th = np.full((K, C), self.alpha)
                for c in range(C):
                    m = X[:, j] == c
                    if m.any(): th[:, c] += resp[m].sum(0)
                th /= th.sum(1, keepdims=True); self.theta.append(np.log(th))
            ll = self.loglik_matrix(X)
            mx = ll.max(1, keepdims=True); lse = mx + np.log(np.exp(ll - mx).sum(1, keepdims=True))
            resp = np.exp(ll - lse); tot = lse.sum()
            if tot - prev < 1e-4: break
            prev = tot
        self.ll_train = tot; return self
    def loglik_matrix(self, X):
        ll = np.tile(np.log(self.pi), (X.shape[0], 1))
        for j in range(X.shape[1]):
            v = X[:, j]; ok = v >= 0
            ll[ok] += self.theta[j][:, v[ok]].T
        return ll
    def loglik(self, X):
        ll = self.loglik_matrix(X); mx = ll.max(1, keepdims=True)
        return float((mx + np.log(np.exp(ll - mx).sum(1, keepdims=True))).sum())
    def predict(self, X): return self.loglik_matrix(X).argmax(1)
    def posterior(self, X):
        ll = self.loglik_matrix(X); mx = ll.max(1, keepdims=True); return np.exp(ll - mx) / np.exp(ll - mx).sum(1, keepdims=True)

def best_fit(T, feats, K, restarts=8, X=None, cats_from=None):
    base = LCA(feats, K); base.set_cats(cats_from or T)
    if X is None: X = base.encode(T)
    fits = []
    for s in range(restarts):
        m = LCA(feats, K); m.cats = base.cats; m.fit(T, seed=1000 * K + s, X=X); fits.append(m)
    fits.sort(key=lambda m: -m.ll_train)
    labs = [m.predict(X) for m in fits]
    stab = np.mean([ARI(labs[0], l) for l in labs[1:]]) if len(labs) > 1 else 1.0
    return fits[0], stab, X

def cv_select(T, feats, Ks, folds=5, restarts=4):
    """5-fold held-out log-likelihood per text; returns {K: (mean, se)} with se over folds (per-text means)"""
    base = LCA(feats, 1); base.set_cats(T); X = base.encode(T)
    N = len(T); perm = rng.permutation(N); fold = np.zeros(N, int); fold[perm] = np.arange(N) % folds
    out = {}
    for K in Ks:
        per = []
        for f in range(folds):
            tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
            Ttr = [T[i] for i in tr]
            best = None
            for s in range(restarts):
                m = LCA(feats, K); m.cats = base.cats; m.fit(Ttr, seed=7 * K + s, X=X[tr])
                if best is None or m.ll_train > best.ll_train: best = m
            per.append(best.loglik(X[te]) / len(te))
        per = np.array(per)
        # paired SE vs nothing yet; store raw folds
        out[K] = per
    return out

def choose_K(cv, frac=0.90):
    """held-out LL rises with K without a plateau (cycle 1), so the 1-SE rule returns Kmax; the working K is the smallest K that
    reaches `frac` of the K1 -> Kmax held-out gain (stated as a resolution choice, not a discovered number)"""
    mean = {K: float(np.mean(v)) for K, v in cv.items()}
    bestK = max(mean, key=mean.get)
    se1 = bestK
    for K in sorted(cv):
        d = np.asarray(cv[bestK]) - np.asarray(cv[K]); se = d.std(ddof=1) / math.sqrt(len(d))
        if d.mean() <= se: se1 = K; break
    lo = mean[min(mean)]; hi = mean[bestK]
    Kf = min(K for K in mean if mean[K] >= lo + frac * (hi - lo))
    return bestK, se1, Kf, mean

def cv_cached(T, feats, Ks, key, restarts=4):
    fn = OUT + f'loop69_cv_{key}.json'
    if os.path.exists(fn):
        d = json.load(open(fn)); return {int(k): np.array(v) for k, v in d.items()}
    cv = cv_select(T, feats, Ks, restarts=restarts)
    json.dump({str(k): v.tolist() for k, v in cv.items()}, open(fn, 'w')); return cv

def synthetic_control(T, feats, K, level):
    """recoverability control: sample N texts from the fitted K-class model; does the same CV rule find K?"""
    m, _, X = best_fit(T, feats, K, restarts=4)
    N = len(T); z = rng.choice(K, N, p=m.pi)
    S = []
    for i in range(N):
        t = {}
        for j, f in enumerate(feats):
            p_ = np.exp(m.theta[j][z[i]]); inv = {v: k for k, v in m.cats[j].items()}
            t[f] = inv[int(rng.choice(len(p_), p=p_ / p_.sum()))]
        S.append(t)
    cv = cv_select(S, feats, range(max(1, K - 4), K + 7), restarts=3)
    bestK, se1, Kf, mean = choose_K(cv)
    return bestK, se1, mean

def perm_null_ami(lab, obj, nperm):
    obs = AMI(obj, lab); mi = MI(obj, lab) / math.log(2)
    null = np.empty(nperm); nullmi = np.empty(nperm)
    for p in range(nperm):
        l2 = rng.permutation(lab); null[p] = AMI(obj, l2); nullmi[p] = MI(obj, l2) / math.log(2)
    return obs, (null.mean(), np.quantile(null, 0.975), (np.sum(null >= obs) + 1) / (nperm + 1)), mi, (nullmi.mean(), np.quantile(nullmi, 0.975))

def map_accuracy(lab, obj, folds=5):
    """predict the object label from the register by the majority rule, 5-fold; baseline = modal object label"""
    lab = np.asarray(lab); obj = np.asarray(obj); N = len(lab)
    perm = rng.permutation(N); fold = np.zeros(N, int); fold[perm] = np.arange(N) % folds
    correct = 0; base = 0
    for f in range(folds):
        tr = fold != f; te = fold == f
        maj = {}
        for k in np.unique(lab[tr]):
            c = collections.Counter(obj[tr][lab[tr] == k]); maj[k] = c.most_common(1)[0][0]
        modal = collections.Counter(obj[tr]).most_common(1)[0][0]
        pred = np.array([maj.get(k, modal) for k in lab[te]])
        correct += (pred == obj[te]).sum(); base += (obj[te] == modal).sum()
    return correct / N, base / N

def describe_register(T, lab, k, feats, overall):
    idx = [i for i in range(len(T)) if lab[i] == k]
    parts = []
    for f in feats:
        c = collections.Counter(T[i][f] for i in idx); n = len(idx)
        top = [(v, c[v] / n, overall[f][v]) for v in c]
        top.sort(key=lambda x: -(x[1] - x[2]))
        parts.append(f + ': ' + ', '.join(f'{v} {s:.2f} (base {b:.2f})' for v, s, b in top[:2] if s - b > 0.08 or s > 0.6))
    return [p for p in parts if ': ' in p and len(p.split(': ')[1]) > 0]

def name_register(T, lab, k):
    idx = [i for i in range(len(T)) if lab[i] == k]
    tf = collections.Counter(T[i]['tf'] for i in idx)
    return tf.most_common(3)

# =====================================================================================
def dedup(T):
    """one unit per distinct (text, fine type, site): moulded-tablet and copper copies collapse"""
    seen = {}; U = []
    for t in T:
        k = (tuple(t['seq']), t['tf'], t['site'])
        if k not in seen: seen[k] = 1; U.append(t)
    return U

def cond_perm_ami(lab, obj, strat, nperm):
    """AMI(register, object) vs register labels permuted within strata (e.g. length bin): does the register carry format
    information beyond the stratifying variable?"""
    lab = np.asarray(lab); obj = np.asarray(obj); strat = np.asarray(strat)
    groups = [np.where(strat == g)[0] for g in np.unique(strat)]
    obs = AMI(obj, lab); null = []
    L = lab.copy()
    for p in range(nperm):
        for g in groups: L[g] = lab[g][rng.permutation(len(g))]
        null.append(AMI(obj, L))
    null = np.array(null)
    return obs, null.mean(), np.quantile(null, 0.975), (np.sum(null >= obs) + 1) / (nperm + 1)

def nb_ceiling(T, feats, dim, folds=5):
    """supervised ceiling: naive Bayes on the same text features predicting the object label (5-fold)"""
    obj = np.array([t[dim] for t in T]); N = len(T)
    perm = rng.permutation(N); fold = np.zeros(N, int); fold[perm] = np.arange(N) % folds
    correct = 0
    for f in range(folds):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        cls = sorted(set(obj[tr])); prior = collections.Counter(obj[tr])
        cnt = {c: [collections.Counter() for _ in feats] for c in cls}
        for i in tr:
            for j, fe in enumerate(feats): cnt[obj[i]][j][T[i][fe]] += 1
        nv = [len({t[fe] for t in T}) for fe in feats]
        for i in te:
            best = None
            for c in cls:
                lp = math.log(prior[c])
                for j, fe in enumerate(feats): lp += math.log((cnt[c][j][T[i][fe]] + 0.5) / (prior[c] + 0.5 * nv[j]))
                if best is None or lp > best[0]: best = (lp, c)
            correct += best[1] == obj[i]
    return correct / N

def cycle1(level):
    T0 = load_wells(level)
    P(f'# S-DARK-69 cycle 1 ({level}): latent registers from text-side features; nperm={NP}')
    for unit, T in (('objects', T0), ('dedup', dedup(T0))):
        P(f'\n######## unit = {unit}: {len(T)} complete texts; fine types: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(t["tf"] for t in T).most_common()))
        for tag, feats in (('with_len', FEATS), ('no_len', FEATS_NOLEN)):
            if unit == 'dedup' and tag == 'no_len': continue
            P(f'\n## Feature set {tag} ({len(feats)} features)')
            if unit == 'objects' and tag == 'with_len':
                cv = cv_cached(T, feats, range(1, 21), f'{level}_{unit}_{tag}', restarts=3)
            else:
                cv = cv_cached(T, feats, range(1, 17), f'{level}_{unit}_{tag}', restarts=2)
            bestK, se1, Ksel, mean = choose_K(cv)
            P('held-out loglik per text by K: ' + ', '.join(f'K{K} {v:.3f}' for K, v in mean.items()))
            P(f'best K {bestK}; 1-SE K {se1}; working K (90% of K1->Kmax held-out gain) {Ksel}; 80%: {choose_K(cv, 0.8)[2]}, 95%: {choose_K(cv, 0.95)[2]}')
            if unit == 'objects' and tag == 'with_len':
                sb, ss, sm = synthetic_control(T, feats, Ksel, level)
                P(f'CONTROL synthetic data drawn from the fitted K={Ksel} model: CV best K {sb}, 1-SE K {ss} (' + ', '.join(f'K{k} {v:.3f}' for k, v in sm.items()) + ')')
            for K in [Ksel]:
                m, stab, X = best_fit(T, feats, K)
                lab = m.predict(X)
                P(f'\n### K = {K} ({unit}, {tag}): stability (mean ARI over 8 restarts) {stab:.3f}; class sizes ' +
                  ', '.join(f'R{k} {int((lab == k).sum())}' for k in range(K)))
                overall = {f: collections.Counter(t[f] for t in T) for f in feats}
                for f in overall:
                    n = len(T); overall[f] = {v: c / n for v, c in overall[f].items()}
                for k in range(K):
                    P(f'  R{k} n={int((lab == k).sum())}: top types ' + ', '.join(f'{a} {b}' for a, b in name_register(T, lab, k)))
                    for line in describe_register(T, lab, k, feats, overall):
                        if not line.split(': ')[1].startswith(('no 1.00', 'none 1.00')): P('      ' + line)
                for dim in ('tf', 'tc', 'shape', 'material', 'boss', 'sides'):
                    obj = np.array([t[dim] for t in T])
                    keep = obj != 'unknown'
                    ami, (nm, n97, pv), mi, (nmm, nm97) = perm_null_ami(lab[keep], obj[keep], NP)
                    acc, base = map_accuracy(lab[keep], obj[keep])
                    P(f'  {dim:9s} n={keep.sum()}: AMI {ami:.3f} (null mean {nm:.4f}, 97.5% {n97:.4f}, P {pv:.3f}); MI {mi:.3f} bits (null {nmm:.3f}); '
                      f'register->{dim} accuracy {acc:.3f} vs modal {base:.3f}')
                obj = np.array([t['tf'] for t in T])
                o, nm, n97, pv = cond_perm_ami(lab, obj, [t['len'] for t in T], min(NP, 300))
                P(f'  CONTROL within-length null (register permuted inside length bin): AMI(tf) {o:.3f} vs {nm:.3f} (97.5% {n97:.3f}, P {pv:.3f})')
                o, nm, n97, pv = cond_perm_ami(lab, obj, [t['len'] + '|' + t['head'] for t in T], min(NP, 300))
                P(f'  CONTROL within length x head null: AMI(tf) {o:.3f} vs {nm:.3f} (97.5% {n97:.3f}, P {pv:.3f})')
                o, nm, n97, pv = cond_perm_ami(lab, obj, [t['site'] if t['big'] else 'other' for t in T], min(NP, 300))
                P(f'  CONTROL within-site null: AMI(tf) {o:.3f} vs {nm:.3f} (97.5% {n97:.3f}, P {pv:.3f})')
                for f in ('len', 'head', 'opener', 'numser'):
                    if f in feats:
                        lf = np.array([t[f] for t in T])
                        acc, base = map_accuracy(lf, obj)
                        P(f'  comparator {f} alone -> fine type: AMI {AMI(obj, lf):.3f}, accuracy {acc:.3f}')
                lf = np.array([t['len'] + '|' + t['head'] for t in T]); acc, _ = map_accuracy(lf, obj)
                P(f'  comparator len x head cell ({len(set(lf))} cells) -> fine type: AMI {AMI(obj, lf):.3f}, accuracy {acc:.3f}')
                P(f'  supervised ceiling (naive Bayes, same features, 5-fold) -> fine type accuracy {nb_ceiling(T, feats, "tf"):.3f}; coarse {nb_ceiling(T, feats, "tc"):.3f}')
                if unit == 'objects' and tag == 'with_len' and K == Ksel:
                    json.dump(dict(level=level, K=K, feats=feats, cats=[{str(kk): vv for kk, vv in c.items()} for c in m.cats],
                                   pi=m.pi.tolist(), theta=[th.tolist() for th in m.theta],
                                   labels={t['id']: int(l) for t, l in zip(T, lab)}),
                              open(OUT + f'loop69_model_{level}.json', 'w'))
    open(OUT + f'loop69_c1_{level}.txt', 'w').write('\n'.join(LOG))

def load_model(level):
    d = json.load(open(OUT + f'loop69_model_{level}.json'))
    m = LCA(d['feats'], d['K']); m.cats = [{k: v for k, v in c.items()} for c in d['cats']]
    m.pi = np.array(d['pi']); m.theta = [np.array(th) for th in d['theta']]
    return m, d

def register_names(T, lab, K):
    names = {}
    for k in range(K):
        idx = [i for i in range(len(T)) if lab[i] == k]
        tf = collections.Counter(T[i]['tf'] for i in idx).most_common(1)[0][0]
        hd = collections.Counter(T[i]['head'] for i in idx).most_common(1)[0][0]
        op = collections.Counter(T[i]['opener'] for i in idx).most_common(1)[0][0]
        ln = np.mean([T[i]['n'] for i in idx])
        names[k] = f'R{k}[{tf};{hd};{op};L{ln:.1f}]'
    return names

def cycle2(level):
    T = load_wells(level); m, d = load_model(level); X = m.encode(T); lab = m.predict(X); K = m.K
    names = register_names(T, lab, K)
    P(f'# S-DARK-69 cycle 2 ({level}): register cards, K={K}, {len(T)} texts; nperm={NP}')
    rows = []
    sites = ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro']
    for k in range(K):
        idx = [i for i in range(len(T)) if lab[i] == k]; n = len(idx)
        tf = collections.Counter(T[i]['tf'] for i in idx)
        st = collections.Counter(T[i]['site'] if T[i]['site'] in sites else ('foreign' if T[i]['foreign'] else 'other') for i in idx)
        ph = collections.Counter(T[i]['phase'] for i in idx if T[i]['phase'])
        hd = collections.Counter(T[i]['head'] for i in idx)
        P(f'\n## {names[k]} n={n}')
        P('  types: ' + ', '.join(f'{a} {b} ({b / n:.2f})' for a, b in tf.most_common(6)))
        P('  sites: ' + ', '.join(f'{a} {b}' for a, b in st.most_common()))
        P('  phases: ' + ', '.join(f'{a} {b}' for a, b in sorted(ph.items())))
        P('  heads: ' + ', '.join(f'{a} {b}' for a, b in hd.most_common(6)))
        P('  mean length %.2f; opener %.2f; conn %.2f; W1 %.2f; W31 %.2f; W32 %.2f; suffix W400 %.2f; numser %s; fish %s; person %.2f; grid %.2f; quantity %.2f; second %.2f' % (
            np.mean([T[i]['n'] for i in idx]), np.mean([T[i]['opener'] != 'none' for i in idx]), np.mean([T[i]['conn'] == 'yes' for i in idx]),
            np.mean([T[i]['W1'] == 'yes' for i in idx]), np.mean([T[i]['W31'] == 'yes' for i in idx]), np.mean([T[i]['W32'] == 'yes' for i in idx]),
            np.mean([T[i]['suffix'] == 'W400' for i in idx]),
            dict(collections.Counter(T[i]['numser'] for i in idx)), dict(collections.Counter(T[i]['fish'] for i in idx)),
            np.mean([T[i]['person'] == 'yes' for i in idx]), np.mean([T[i]['grid'] == 'yes' for i in idx]),
            np.mean([T[i]['quantity'] == 'yes' for i in idx]), np.mean([T[i]['second'] == 'yes' for i in idx])))
        ex = [T[i] for i in idx][:400]; ex = rng.choice(len(ex), min(4, len(ex)), replace=False)
        P('  examples: ' + ' | '.join(f"{[T[i] for i in idx][e]['cisi']} {'-'.join(map(str, [T[i] for i in idx][e]['seq']))}" for e in ex))
        rows.append(dict(level=level, register=names[k], n=n, types=';'.join(f'{a}:{b}' for a, b in tf.most_common(4)),
                         sites=';'.join(f'{a}:{b}' for a, b in st.most_common()), heads=';'.join(f'{a}:{b}' for a, b in hd.most_common(4)),
                         mean_len=round(float(np.mean([T[i]['n'] for i in idx])), 2)))
    # site x register independence given type (null: register labels permuted within fine type x length-bin)
    P('\n## Register x site, conditional on type (null: labels permuted within fine type x length-bin)')
    def mi_site(l):
        return MI([t['site'] if t['site'] in sites else 'other' for t in T], l) / math.log(2)
    obs = mi_site(lab)
    strata = collections.defaultdict(list)
    for i, t in enumerate(T): strata[(t['tf'], t['len'])].append(i)
    groups = [np.array(v) for v in strata.values() if len(v) > 1]
    null = []
    L = lab.copy()
    for p in range(min(NP, 300)):
        for g in groups: L[g] = lab[g][rng.permutation(len(g))]
        null.append(mi_site(L))
    P(f'  MI(register; site) {obs:.4f} bits vs null {np.mean(null):.4f} (97.5% {np.quantile(null, 0.975):.4f}; P {(np.sum(np.array(null) >= obs) + 1) / (len(null) + 1):.3f})')
    # chronology: register x phase within site, null = labels permuted within site x fine type
    P('\n## Register x phase (objects with a phase; null: labels permuted within site x fine type)')
    ph_idx = [i for i, t in enumerate(T) if t['phase']]
    for site_pref in ('MD-', 'H-3', 'H-Str', 'Dlv-', 'K-'):
        idx = np.array([i for i in ph_idx if T[i]['phase'].startswith(site_pref)])
        if len(idx) < 30: continue
        phs = [T[i]['phase'] for i in idx]; l0 = lab[idx]
        obs = MI(phs, l0) / math.log(2)
        st = collections.defaultdict(list)
        for j, i in enumerate(idx): st[T[i]['tf']].append(j)
        gr = [np.array(v) for v in st.values() if len(v) > 1]
        nul = []
        for p_ in range(300):
            L2 = l0.copy()
            for g in gr: L2[g] = l0[g][rng.permutation(len(g))]
            nul.append(MI(phs, L2) / math.log(2))
        P(f'  {site_pref:6s} n={len(idx)} phases {dict(collections.Counter(phs))}: MI {obs:.4f} vs null {np.mean(nul):.4f} (97.5% {np.quantile(nul, 0.975):.4f}, P {(np.sum(np.array(nul) >= obs) + 1) / 301:.3f})')
        tab = collections.defaultdict(collections.Counter)
        for i in idx: tab[T[i]['phase']][names[lab[i]].split('[')[0]] += 1
        for ph in sorted(tab): P(f'      {ph}: ' + ', '.join(f'{k} {v}' for k, v in sorted(tab[ph].items())))
    # designation-slot loyalty
    P('\n## Designation slot: loyalty of middle elements to a register (elements with >= 5 tokens; null: register labels permuted within site x length-bin)')
    def loyalty(l, min_tok=5):
        cnt = collections.defaultdict(collections.Counter)
        for i, t in enumerate(T):
            for e in set(t['mid_els']): cnt[e][l[i]] += 1
        els = [e for e in cnt if sum(cnt[e].values()) >= min_tok]
        loy = np.mean([cnt[e].most_common(1)[0][1] / sum(cnt[e].values()) for e in els])
        multi = np.mean([len(cnt[e]) >= 2 for e in els])
        return loy, multi, len(els), cnt
    obs_l, obs_m, nel, cnt = loyalty(lab)
    strata = collections.defaultdict(list)
    for i, t in enumerate(T): strata[(t['site'], t['len'])].append(i)
    groups = [np.array(v) for v in strata.values() if len(v) > 1]
    nl = []; nm = []
    for p in range(min(NP, 300)):
        for g in groups: L[g] = lab[g][rng.permutation(len(g))]
        a, b, _, _ = loyalty(L); nl.append(a); nm.append(b)
    P(f'  {nel} elements: mean loyalty {obs_l:.3f} vs null {np.mean(nl):.3f} [{np.quantile(nl, 0.025):.3f}, {np.quantile(nl, 0.975):.3f}]; '
      f'share in >= 2 registers {obs_m:.3f} vs null {np.mean(nm):.3f}')
    # per register: share of its middle elements (types) that also occur in another register; and register-exclusive elements
    for k in range(K):
        els_k = [e for e in cnt if cnt[e][k] > 0]
        shared = sum(1 for e in els_k if sum(cnt[e].values()) - cnt[e][k] > 0)
        excl = [e for e in els_k if sum(cnt[e].values()) - cnt[e][k] == 0 and cnt[e][k] >= 3]
        P(f'  {names[k]}: {len(els_k)} element types, {shared / max(len(els_k), 1):.2f} also in another register; exclusive (>= 3 tokens): {excl[:12]}')
    # loyalty within the seal registers only (is there a register-specific inventory among seals?)
    seal_regs = [k for k in range(K) if collections.Counter(T[i]['tc'] for i in range(len(T)) if lab[i] == k).most_common(1)[0][0] == 'seal']
    P(f'  seal-dominated registers: {[names[k] for k in seal_regs]}')
    with open(OUT + 'loop69_registers.csv', 'a' if level != 'seq_raw' else 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if level == 'seq_raw': w.writeheader()
        for r in rows: w.writerow(r)
    open(OUT + f'loop69_c2_{level}.txt', 'w').write('\n'.join(LOG))

def cycle3(level):
    T = load_wells(level); m, d = load_model(level); X = m.encode(T); lab = m.predict(X); K = m.K
    post = m.posterior(X)
    names = register_names(T, lab, K)
    dom = {}
    for k in range(K): dom[k] = collections.Counter(T[i]['tc'] for i in range(len(T)) if lab[i] == k).most_common(1)[0][0]
    P(f'# S-DARK-69 cycle 3 ({level}): transitions and city structure, K={K}')
    P('registers: ' + ', '.join(f'{names[k]} -> {dom[k]}' for k in range(K)))
    # (a) cross-format texts
    P('\n## (a) Objects whose text falls in a register dominated by another medium')
    cross = collections.Counter(); ex = collections.defaultdict(list)
    for i, t in enumerate(T):
        if dom[lab[i]] != t['tc'] and post[i].max() > 0.8:
            cross[(t['tc'], names[lab[i]])] += 1
            if len(ex[(t['tc'], names[lab[i]])]) < 3: ex[(t['tc'], names[lab[i]])].append(f"{t['cisi']} {'-'.join(map(str, t['seq']))}")
    tot = collections.Counter(t['tc'] for t in T)
    for (tc, r), c in cross.most_common(20):
        P(f'  {tc} in {r}: {c} of {tot[tc]} ({c / tot[tc]:.3f}) e.g. ' + ' | '.join(ex[(tc, r)]))
    # seals in tablet registers broken down by fine seal type
    P('  by fine type (posterior > 0.8): ' + ', '.join(f'{tf} {c}/{n}' for tf, c, n in
      [(tf, sum(1 for i, t in enumerate(T) if t['tf'] == tf and dom[lab[i]] != t['tc'] and post[i].max() > 0.8), sum(1 for t in T if t['tf'] == tf))
       for tf in ('seal_sq', 'seal_bar', 'seal_round', 'tab_mould', 'tab_inc', 'tab_cu', 'sealing', 'pot')]))
    # (b) two-faced objects: do the faces share a register?
    P('\n## (b) Two-faced objects (same CISI number, >= 2 complete texts): same register on both faces?')
    by = collections.defaultdict(list)
    for i, t in enumerate(T): by[t['cisi']].append(i)
    pairs = [(v[0], v[1]) for v in by.values() if len(v) >= 2]
    same = np.mean([lab[a] == lab[b] for a, b in pairs]) if pairs else float('nan')
    # null: random pairs within the same fine type
    byt = collections.defaultdict(list)
    for i, t in enumerate(T): byt[t['tf']].append(i)
    nulls = []
    for p in range(200):
        s = []
        for a, b in pairs:
            pool = byt[T[a]['tf']]; c = pool[rng.integers(len(pool))]; s.append(lab[a] == lab[c])
        nulls.append(np.mean(s))
    P(f'  {len(pairs)} two-faced objects: faces in the same register {same:.3f} vs random same-type pair {np.mean(nulls):.3f} [{np.quantile(nulls, 0.025):.3f}, {np.quantile(nulls, 0.975):.3f}]')
    pc = collections.Counter((names[lab[a]], names[lab[b]]) if lab[a] <= lab[b] else (names[lab[b]], names[lab[a]]) for a, b in pairs)
    P('  face-register pairs: ' + ', '.join(f'{a} + {b}: {c}' for (a, b), c in pc.most_common(8)))
    # (c) heads across registers
    P('\n## (c) Heads across registers (share of each head\'s texts per register)')
    heads = ['jar', 'arrow', 'W151', 'W156', 'W154/158', 'box527', 'W226', 'W615/617', 'W700', 'none']
    for h in heads:
        idx = [i for i, t in enumerate(T) if t['head'] == h]
        if len(idx) < 8: continue
        c = collections.Counter(lab[i] for i in idx)
        P(f'  {h:9s} n={len(idx)}: ' + ', '.join(f'R{k} {c[k] / len(idx):.2f}' for k in range(K) if c[k] / len(idx) >= 0.05))
    # register entropy of heads vs the opener
    # (d) city structure
    P('\n## (d) Register structure by city: refit K-class model per city, ARI with the global map; register shares among seals')
    groups = {'MD': [i for i, t in enumerate(T) if t['site'] == 'Mohenjo-daro'], 'Harappa': [i for i, t in enumerate(T) if t['site'] == 'Harappa'],
              'held-out': [i for i, t in enumerate(T) if not t['big']]}
    for g, idx in groups.items():
        Tg = [T[i] for i in idx]
        mg, stab, Xg = best_fit(Tg, d['feats'], K, restarts=6, cats_from=T)
        lg = mg.predict(Xg)
        ari = ARI(lab[idx], lg)
        obj = np.array([t['tf'] for t in Tg]); keep = obj != 'unknown'
        ami_glob = AMI(obj[keep], lab[idx][keep]); ami_loc = AMI(obj[keep], lg[keep])
        acc_g, base = map_accuracy(lab[idx][keep], obj[keep]); acc_l, _ = map_accuracy(lg[keep], obj[keep])
        P(f'  {g}: n={len(idx)}; refit stability {stab:.2f}; ARI(global, local refit) {ari:.3f}; AMI with fine type: global map {ami_glob:.3f}, local refit {ami_loc:.3f}; '
          f'accuracy global {acc_g:.3f} / local {acc_l:.3f} vs modal {base:.3f}')
        # cv: K chosen locally
        cv = cv_select(Tg, d['feats'], range(1, 16), restarts=2)
        bestK, se1, Ksel, mean = choose_K(cv)
        P(f'      local held-out loglik: best K {bestK}, 1-SE K {se1}, 90%-gain K {Ksel}; ' + ', '.join(f'K{k} {v:.3f}' for k, v in mean.items()))
    P('  register shares among SEALS by site (global map):')
    for s in ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro', 'HELD-OUT-ALL', 'FOREIGN']:
        if s == 'HELD-OUT-ALL': idx = [i for i, t in enumerate(T) if not t['big'] and t['tc'] == 'seal' and not t['foreign']]
        elif s == 'FOREIGN': idx = [i for i, t in enumerate(T) if t['foreign']]
        else: idx = [i for i, t in enumerate(T) if t['site'] == s and t['tc'] == 'seal']
        if len(idx) < 10: continue
        c = collections.Counter(lab[i] for i in idx)
        P(f'    {s:13s} n={len(idx)}: ' + ', '.join(f'R{k} {c[k] / len(idx):.2f}' for k in range(K) if c[k] > 0))
    # JSD of seal register shares MD vs Harappa vs within-city halves
    def shares(idx):
        c = collections.Counter(lab[i] for i in idx); v = np.array([c[k] for k in range(K)], float) + 0.5; return v / v.sum()
    def jsd(p, q):
        mm = (p + q) / 2; return 0.5 * np.sum(p * np.log2(p / mm)) + 0.5 * np.sum(q * np.log2(q / mm))
    md = [i for i, t in enumerate(T) if t['site'] == 'Mohenjo-daro' and t['tc'] == 'seal']
    hp = [i for i, t in enumerate(T) if t['site'] == 'Harappa' and t['tc'] == 'seal']
    ho = [i for i, t in enumerate(T) if not t['big'] and t['tc'] == 'seal' and not t['foreign']]
    n = min(len(md), len(hp), len(ho))
    def draw(idx): return list(rng.choice(idx, n, replace=False))
    between_mh = np.mean([jsd(shares(draw(md)), shares(draw(hp))) for _ in range(100)])
    between_mo = np.mean([jsd(shares(draw(md)), shares(draw(ho))) for _ in range(100)])
    within = []
    for _ in range(100):
        p = rng.permutation(md); within.append(jsd(shares(list(p[:n])), shares(list(p[n:2 * n])) if len(md) >= 2 * n else shares(list(p[n:]))))
    P(f'  JSD of seal register shares (n={n} draws): MD vs Harappa {between_mh:.4f}, MD vs held-out {between_mo:.4f}, within-MD halves {np.mean(within):.4f} '
      f'(ratios x{between_mh / max(np.mean(within), 1e-9):.2f}, x{between_mo / max(np.mean(within), 1e-9):.2f})')
    open(OUT + f'loop69_c3_{level}.txt', 'w').write('\n'.join(LOG))

def cycle4(level):
    T = load_wells(level); m, d = load_model(level); K = m.K
    Xw = m.encode(T); labw = m.predict(Xw); names = register_names(T, labw, K)
    I = load_im77()
    P(f'# S-DARK-69 cycle 4 ({level}): IM77 through the bridge; Wells-trained K={K} model; nperm={NP}')
    P(f'IM77 texts {len(I)} (IM77-only {sum(1 for t in I if t["new"])}); types: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(t['tf'] for t in I).most_common()))
    # encode IM77 with the Wells category maps (feature values are shared labels)
    Xi = m.encode(I); labi = m.predict(Xi); posti = m.posterior(Xi)
    unk = (Xi < 0).sum(); P(f'IM77 feature values unseen in Wells categories: {unk} (tablet opener class absent in M space, stated)')
    P('IM77 register shares: ' + ', '.join(f'{names[k]} {np.mean(labi == k):.3f} (Wells {np.mean(labw == k):.3f})' for k in range(K)))
    # register x IM77 type
    P('\n## Register x recorded IM77 object type (row shares)')
    for k in range(K):
        idx = labi == k
        c = collections.Counter(np.array([t['tf'] for t in I])[idx])
        P(f'  {names[k]} n={idx.sum()}: ' + ', '.join(f'{a} {b} ({b / idx.sum():.2f})' for a, b in c.most_common(5)))
    # prediction accuracy: map register -> IM77 type learned on IM77 (5-fold) and learned on Wells (coarse map seal/tablet/sealing/pot) applied to IM77
    for label, sub in (('all IM77', [i for i in range(len(I))]), ('IM77-only (324)', [i for i, t in enumerate(I) if t['new']]),
                       ('IM77 big cities', [i for i, t in enumerate(I) if t['big']]), ('IM77 small sites + WA', [i for i, t in enumerate(I) if not t['big']])):
        obj = np.array([I[i]['tf'] for i in sub]); l = labi[sub]
        ami, (nm, n97, pv), mi, (nmm, nm97) = perm_null_ami(l, obj, NP)
        acc, base = map_accuracy(l, obj)
        # Wells-trained majority map, translated into IM77 categories: IM77 'sealing' = Wells TAB:B moulded tablets + TAG sealings,
        # 'miniature tablet' = Wells TAB:I, 'copper tablet' = TAB:C
        W2I = {'seal_sq': 'seal', 'seal_bar': 'seal', 'seal_round': 'seal', 'seal_oth': 'seal', 'tab_mould': 'sealing', 'sealing': 'sealing',
               'tab_inc': 'tablet', 'tab_cu': 'tab_cu', 'pot': 'pot', 'rod': 'rod'}
        wtc = np.array([W2I.get(t['tf'], 'misc') for t in T])
        wmap = {k: collections.Counter(wtc[labw == k]).most_common(1)[0][0] for k in range(K)}
        acc_w = np.mean([wmap[k] == o for k, o in zip(l, obj)])
        base_w = collections.Counter(obj).most_common(1)[0][1] / len(obj)
        # null for the Wells-learned map: IM77 labels permuted among texts
        nullw = [np.mean([wmap[k] == o for k, o in zip(l, rng.permutation(obj))]) for _ in range(min(NP, 500))]
        P(f'  {label}: n={len(sub)}; AMI(register, type) {ami:.3f} (null {nm:.4f}, 97.5% {n97:.4f}, P {pv:.3f}); MI {mi:.3f} bits; '
          f'IM77-learned map accuracy {acc:.3f} vs modal {base:.3f}; Wells-learned map accuracy {acc_w:.3f} vs modal {base_w:.3f} (permuted-label null {np.mean(nullw):.3f}, 97.5% {np.quantile(nullw, 0.975):.3f})')
    # the Wells 70% overlap check: IM77-only vs overlap
    # IM77 type mix per register in new texts
    sub = [i for i, t in enumerate(I) if t['new']]
    P('\n## IM77-only texts: register x type')
    for k in range(K):
        idx = [i for i in sub if labi[i] == k]
        if not idx: continue
        c = collections.Counter(I[i]['tf'] for i in idx)
        P(f'  {names[k]} n={len(idx)}: ' + ', '.join(f'{a} {b}' for a, b in c.most_common(5)) + ' | e.g. ' +
          ' | '.join(f"{I[i]['id']} {'-'.join(map(str, I[i]['seq']))}" for i in idx[:3]))
    # refit on IM77 alone: does K hold and do classes match?
    cv = cv_select(I, d['feats'], range(1, 17), restarts=2)
    bestK, se1, Ksel, mean = choose_K(cv)
    P(f'\nIM77 own fit: held-out loglik best K {bestK}, 1-SE K {se1}, 90%-gain K {Ksel}; ' + ', '.join(f'K{k} {v:.3f}' for k, v in mean.items()))
    mi_, stab, Xi2 = best_fit(I, d['feats'], K, restarts=6, cats_from=T)
    li = mi_.predict(Xi2)
    P(f'IM77 refit at K={K}: stability {stab:.2f}; ARI(Wells-model labels, IM77 refit labels) {ARI(labi, li):.3f}; '
      f'AMI with type: Wells model {AMI([t["tf"] for t in I], labi):.3f}, IM77 refit {AMI([t["tf"] for t in I], li):.3f}')
    open(OUT + f'loop69_c4_{level}.txt', 'w').write('\n'.join(LOG))

if __name__ == '__main__':
    {1: cycle1, 2: cycle2, 3: cycle3, 4: cycle4}[CY](LV)
