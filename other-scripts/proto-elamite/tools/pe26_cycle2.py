"""pe26 cycle 2: rank Susa tablets by how plateau-like their CLAY looks; freeze + hash BEFORE scoring against
the hXRF labels (data/pe17_xrf_labels.json) and the pe17 text ranking.
Only NMI has colour photos of plateau tablets: Yahya P009532, P009535, P009536, P009538 (all one photo batch with 93 Susa).
P009536 = YT-01 is Susa clay by hXRF, so it is held out of training (it becomes a test item). Training plateau set: 3 tablets.
Features: clay colour+texture, centred within the photo campaign (software|year-month; sensitivity: per day).
Score: massive random guessing -- 2,000 random feature subsets (3-8 features), each a diagonal discriminant
(standardised mean difference Yahya-minus-Susa), averaged. Training Susa tablets are scored leave-one-out.
Planted control: 5 random Susa tablets get the Yahya-minus-Susa colour difference; their percentile is recorded."""
import sys, json, os, hashlib, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
from scipy.stats import spearmanr
rng = np.random.default_rng(2626)
MODE = os.environ.get('BATCH', 'month')
rows = [r for r in load() if r['group'] == 'PE' and r.get('is_grey', 0) < 0.5]
def bkey(r):
    if MODE == 'day': return r['batch']
    parts = r['batch'].split('|'); return parts[2][:22] + '|' + parts[3][:7]
bat = [bkey(r) for r in rows]
X = within_batch_centre(X_of(rows, CLAY), bat)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
ids = [r['id'] for r in rows]; ix = {p: i for i, p in enumerate(ids)}
HOLD = {'P009536', 'P009157'}  # YT-01 (Susa clay at Yahya) and ST-11 (Yahya clay at Susa): test items only
Y = [i for i, r in enumerate(rows) if r['museum'] == 'NMI' and r['site'] == 'Yahya' and r['id'] not in HOLD]
S = [i for i, r in enumerate(rows) if r['museum'] == 'NMI' and r['site'] == 'Susa' and r['id'] not in HOLD]
K = X.shape[1]; NM = 2000
subs = [rng.choice(K, size=rng.integers(3, 9), replace=False) for _ in range(NM)]
def make_w(Yi, Si):
    d = X[Yi].mean(0) - X[Si].mean(0); sd = X[Si].std(0) + 1e-6; return d / sd ** 2
CNT = np.zeros(K)
for s_ in subs: CNT[s_] += 1
CNT /= NM  # a mean of linear subset scores is one linear score with feature weights = inclusion rate
def score_rows(w, R):
    return X[R] @ (w * CNT)
w_full = make_w(Y, S)
allsusa = [i for i, r in enumerate(rows) if r['site'] == 'Susa' and r['museum'] in ('NMI', 'Louvre')]
sc = {}
for i in allsusa:
    if i in S:
        w = make_w(Y, [j for j in S if j != i])  # leave-one-out
        sc[i] = float(score_rows(w, [i])[0])
    else:
        sc[i] = float(score_rows(w_full, [i])[0])
# Yahya tablets scored leave-one-out (each Yahya training tablet removed when scored)
ysc = {}
for i in [j for j, r in enumerate(rows) if r['site'] == 'Yahya']:
    w = make_w([j for j in Y if j != i], S); ysc[ids[i]] = float(score_rows(w, [i])[0])
susa = []
for m in ('NMI', 'Louvre'):
    L = sorted([i for i in allsusa if rows[i]['museum'] == m], key=lambda i: -sc[i])
    for k, i in enumerate(L):
        susa.append(dict(id=ids[i], museum=m, batch=bat[i], clay_plateau_score=round(sc[i], 4), rank_in_museum=k + 1,
                         pct_in_museum=round(1 - k / max(len(L) - 1, 1), 4)))
frozen = dict(note=f'pe26 blind clay-appearance ranking (CDLI thumbnails; colour+texture centred within photo campaign [{MODE}]); '
                   'trained on 3 NMI Yahya tablets (YT-01 and ST-11 held out) vs NMI Susa; frozen before scoring against hXRF and pe17',
              batch_mode=MODE, train_yahya=[ids[i] for i in Y], susa=susa, yahya_loo=ysc)
h = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
frozen['sha256_of_content_without_this_field'] = h
json.dump(frozen, open(os.path.join(D, f'pe26_frozen_clay_ranking_{MODE}.json'), 'w'), indent=0)
print('FROZEN', MODE, 'sha256', h)

# ---------- planted control ----------
pl = []
for rep in range(200):
    pick = list(rng.choice(S, 5, replace=False)); Xs = X.copy()
    X[pick] += (X[Y].mean(0) - X[S].mean(0))
    w = make_w(Y, [j for j in S if j not in pick]); s_all = score_rows(w, S)
    ranks = [np.mean(s_all <= s_all[S.index(p)]) for p in pick]; pl.append(np.mean(ranks))
    X[:] = Xs
print('planted 5 imports (full Yahya-Susa shift): mean percentile', round(float(np.mean(pl)), 3))
pl2 = []
for rep in range(200):
    pick = list(rng.choice(S, 5, replace=False)); Xs = X.copy()
    X[pick] += 0.5 * (X[Y].mean(0) - X[S].mean(0))
    w = make_w(Y, [j for j in S if j not in pick]); s_all = score_rows(w, S)
    pl2.append(np.mean([np.mean(s_all <= s_all[S.index(p)]) for p in pick])); X[:] = Xs
print('planted half shift: mean percentile', round(float(np.mean(pl2)), 3))

# ---------- scoring (after freeze) ----------
res = dict(mode=MODE, sha256=h, planted_full=float(np.mean(pl)), planted_half=float(np.mean(pl2)))
nmi = [s for s in susa if s['museum'] == 'NMI']
st = [s for s in nmi if s['id'] == 'P009157']
res['ST11'] = st[0] if st else None
print('ST-11 (Yahya clay at Susa):', st[0] if st else 'no colour photo', 'of', len(nmi), 'NMI Susa')
res['yahya'] = ysc
r01 = sorted(ysc, key=lambda p: ysc[p])
print('Yahya LOO scores (low = Susa-like):', {p: round(v, 3) for p, v in sorted(ysc.items(), key=lambda kv: kv[1])}, ' YT-01=P009536 rank from Susa-like end:', r01.index('P009536') + 1, 'of', len(r01))
t = json.load(open(os.path.join(D, 'pe17_frozen_ranking.json')))
tx = {s['id']: s['plateau_score'] for s in t['susa']}
for m in ('NMI', 'Louvre', 'all'):
    L = [s for s in susa if (m == 'all' or s['museum'] == m) and s['id'] in tx]
    a = np.array([s['clay_plateau_score'] for s in L]); b = np.array([tx[s['id']] for s in L])
    rho = spearmanr(a, b).correlation; bl = [s['batch'] for s in L]
    nr = np.array([spearmanr(a[perm_within_batch(np.arange(len(a)), bl, rng)], b).correlation for _ in range(2000)])
    p = (np.sum(np.abs(nr) >= abs(rho)) + 1) / 2001
    top = np.argsort(-a)[:max(5, len(a) // 10)]; ptop = float(np.mean([np.mean(b <= b[i]) for i in top]))
    print(f'pe17 text vs clay [{m}] n={len(L)} rho={rho:.3f} p(within-batch perm, two-sided)={p:.4f}; top-10% clay tablets mean text percentile {ptop:.3f}')
    res['pe17_' + m] = dict(n=len(L), rho=float(rho), p=float(p), top_text_pct=ptop)
if st:
    print('ST-11 pe17 text percentile:', [s for s in t['susa'] if s['id'] == 'P009157'][:1])
json.dump(res, open(os.path.join(CK, f'cycle2_{MODE}.json'), 'w'), indent=1, default=float)
