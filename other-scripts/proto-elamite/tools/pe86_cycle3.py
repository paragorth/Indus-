"""pe86 cycle 3: predict header presence on tablets whose first line is lost, from the surviving clay end; freeze; and
re-test cycle-1 leads on the line-art record."""
import sys, os, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe86_common as Q
import pe85_common as C
import pe83_common as P
rng = np.random.default_rng(8603)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DA = Q.build(False); R = DA['R']
lab = [r['t']['lines'][0]['label'] for r in R]
primed = np.array(["'" in x for x in lab]); l1ok = C.col(R, 'l1ok') == 1
lac = np.array([r['t']['lines'][0]['lacuna'] for r in R])
hd = DA['hd']
COV = ['la', 'asp', 'tw', 'nl', 'dx', 'rev']
def X_of(idx, clay=True):
    cols = [C.col(R, k)[idx] for k in COV]
    if clay:
        for k in ('cf_top', 'cf_bot', 'revend', 'cf_hull', 'cf_hull_bot'):
            v = C.col(R, k)[idx]; v = np.where(np.isfinite(v), v, np.nanmedian(v)); cols.append(v)
        cols.append(cols[-5] - cols[-4])  # top minus bottom
    return np.column_stack(cols)
train = np.where(l1ok)[0]
known_damaged = np.where(~l1ok & ~primed & ~lac)[0]  # line 1 damaged but present: header status readable (pseudo-test)
unknown = np.where(primed)[0]  # start of text lost: header status unknown
out = dict(n_train=len(train), n_known_damaged=len(known_damaged), n_unknown=len(unknown))

def fit(Xtr, ytr):
    m = LogisticRegression(C=1.0, max_iter=2000); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m.fit((Xtr - mu) / sd, ytr); return lambda X: m.predict_proba((X - mu) / sd)[:, 1]

# 3a batch-grouped CV on intact tablets: clay+size vs size-only (the clay's own contribution), and shuffled-label null
b = DA['batch'][train]; ub = np.unique(b); rng.shuffle(ub); folds = np.array_split(ub, 5)
def cv(clay, y):
    p = np.zeros(len(train))
    for f in folds:
        te = np.isin(b, f); tr = ~te
        p[te] = fit(X_of(train[tr], clay), y[tr])(X_of(train[te], clay))
    return p
yT = hd[train]
pc = cv(True, yT); ps = cv(False, yT)
out['3a_cv_auc_clay_plus_size'] = round(roc_auc_score(yT, pc), 3); out['3a_cv_auc_size_only'] = round(roc_auc_score(yT, ps), 3)
nul = []
for k in range(100):
    yp = Q.perm_within(yT, DA['strata'][train], rng); nul.append(roc_auc_score(yp, cv(True, yp)) - roc_auc_score(yp, cv(False, yp)))
out['3a_gain'] = round(out['3a_cv_auc_clay_plus_size'] - out['3a_cv_auc_size_only'], 3)
out['3a_gain_null_q95'] = round(float(np.quantile(nul, .95)), 3); out['3a_gain_p'] = round((sum(n >= out['3a_gain'] for n in nul) + 1) / 101, 4)
print('3a', {k: v for k, v in out.items() if k.startswith('3a')}, flush=True)

# 3b pseudo-test on damaged-but-present first lines (header status readable, clay at the header end may be damaged)
fc = fit(X_of(train, True), yT); fs = fit(X_of(train, False), yT)
yk = hd[known_damaged]
pk_c = fc(X_of(known_damaged, True)); pk_s = fs(X_of(known_damaged, False))
out['3b_known_damaged'] = dict(n=len(known_damaged), headed=int(yk.sum()), auc_clay_plus_size=round(roc_auc_score(yk, pk_c), 3), auc_size_only=round(roc_auc_score(yk, pk_s), 3))
# paired bootstrap of the gain
g = []
for _ in range(1000):
    i = rng.integers(0, len(yk), len(yk))
    if yk[i].min() == yk[i].max(): continue
    g.append(roc_auc_score(yk[i], pk_c[i]) - roc_auc_score(yk[i], pk_s[i]))
out['3b_known_damaged']['gain_ci95'] = [round(float(np.quantile(g, .025)), 3), round(float(np.quantile(g, .975)), 3)]
print('3b', out['3b_known_damaged'], flush=True)

# 3c what kind of header: among intact headed, does the clay predict M157 vs other header, beyond size? (grouped CV)
hdi = train[yT == 1]
kind = np.array([1.0 if (common.header(R[i]['t']) or [''])[0] == 'M157' else 0.0 for i in hdi])
bb = DA['batch'][hdi]
def cvk(clay, y):
    p = np.zeros(len(hdi))
    for f in folds:
        te = np.isin(bb, f); tr = ~te
        if te.sum() == 0 or y[tr].min() == y[tr].max(): continue
        p[te] = fit(X_of(hdi[tr], clay), y[tr])(X_of(hdi[te], clay))
    return p
out['3c_kind_M157'] = dict(n=len(hdi), m157=int(kind.sum()), auc_clay_plus_size=round(roc_auc_score(kind, cvk(True, kind)), 3), auc_size_only=round(roc_auc_score(kind, cvk(False, kind)), 3))
print('3c', out['3c_kind_M157'], flush=True)

# 3d freeze predictions for tablets whose first line is lost
pu_c = fc(X_of(unknown, True)); pu_s = fs(X_of(unknown, False))
fz = dict(created='2026-10-07', loop='pe86', model='logistic on 589 intact tablets: size, aspect, thickness/width, log lines, x, reverse + top/bottom corner fill, reverse header-end fill, hull fills, top-bottom',
          rule='Would support: when the first line of these tablets is recovered (join, collation, new photo of an edge), the top third by clay_minus_size has a header rate >= 0.15 above the bottom third. Would kill: difference < 0.05 (n >= 20 resolved).',
          tablets={R[i]['id']: dict(p_header_clay=round(float(a), 3), p_header_size_only=round(float(s), 3), clay_minus_size=round(float(a - s), 3),
                                    first_surviving=R[i]['t']['lines'][0]['raw'][:60]) for i, a, s in zip(unknown, pu_c, pu_s)})
fp = os.path.join(common.DATA, 'pe86_frozen_lost_headers.json')
json.dump(fz, open(fp, 'w'), indent=1, sort_keys=True)
h = Q.hashlib.sha256(open(fp, 'rb').read()).hexdigest(); open(fp + '.sha256', 'w').write(h + '  pe86_frozen_lost_headers.json\n')
out['3d_frozen'] = dict(file='data/pe86_frozen_lost_headers.json', sha256=h, n=len(unknown), mean_p_clay=round(float(pu_c.mean()), 3),
                        n_clay_says_headed_p_gt_0_7=int((pu_c > .7).sum()), n_p_lt_0_3=int((pu_c < .3).sum()),
                        sd_clay_minus_size=round(float(np.std(pu_c - pu_s)), 3))
# internal check on the unknown set: surviving first line looks like an opener (no numerals) -> plausibly the header or the next line
looks = np.array([float(not R[i]['t']['lines'][0]['numerals']) for i in unknown])
out['3d_frozen']['corr_pclay_minus_size_vs_first_surviving_numeral_free'] = round(Q.corr(pu_c - pu_s, looks), 3)
print('3d', out['3d_frozen'], flush=True)
json.dump(out, open(os.path.join(Q.CK, 'c3.json'), 'w'), indent=1)

# 3e re-test of the cycle-1 leads on the hand-drawn line-art record (all tablets with drawings and intact line 1;
# and the subset with no photo outline, never used in cycle 1)
LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json'))); cat = P.catalogue()
photo_ids = set(r['id'] for r in R)
RL = []
for t in common.load():
    la = LA.get(t['id']); c = cat.get(t['id'])
    if not la or not t['lines'] or not c:
        continue
    l1 = t['lines'][0]
    if l1['lacuna'] or l1['damaged']:
        continue
    h_, w_, th = P.num(c['height']), P.num(c['width']), P.num(c['thickness'])
    RL.append(dict(id=t['id'], t=t, top=la['la_cf_top'], bot=la['la_cf_bot'], asp=np.log(la['la_aspect']) if la.get('la_aspect') else 0.0,
                   la=np.log(h_ * w_) if h_ and w_ else np.nan, tw=(th / w_) if th and w_ else np.nan, nl=np.log(len(t['lines'])),
                   dx=float(any(g == 'x' for l in t['lines'] for g in l['signs'])), rev=float(any(l['surface'] == 'reverse' for l in t['lines'])),
                   ho=t['id'] not in photo_ids))
for r in RL:
    for k in ('la', 'tw'):
        if not np.isfinite(r[k]):
            r[k] = np.nanmedian([x[k] for x in RL])
ZL = np.column_stack([np.ones(len(RL))] + [np.array([r[k] for r in RL], float) for k in ('la', 'asp', 'tw', 'nl', 'dx', 'rev')])
FL = Q.line_feats(RL, min_n=1)
hL = np.array([float(common.header(r['t']) is not None) for r in RL])
yt = C.resid(np.array([r['top'] for r in RL]), ZL); yb = C.resid(np.array([r['bot'] for r in RL]), ZL)
ho = np.array([r['ho'] for r in RL]); sL = P.strata(ZL)
def lead(y, sub, feat, sign):
    f = FL.get(feat, np.zeros(len(RL)))
    res = {}
    for nm, m in (('all_lineart', sub), ('heldout_no_photo', sub & ho)):
        if f[m].sum() < 3 or (1 - f[m]).sum() < 3:
            res[nm] = dict(n=int(f[m].sum()), r=None); continue
        r0 = Q.corr(f[m], y[m]); c = 0
        for _ in range(2000):
            yp = Q.perm_within(y, sL, rng); c += sign * Q.corr(f[m], yp[m]) >= sign * r0
        res[nm] = dict(n=int(f[m].sum()), of=int(m.sum()), r=round(r0, 3), p_one_sided=round((c + 1) / 2001, 4))
    return res
out['3e'] = {'one_sign_plus_number_unheaded_top(+)': lead(yt, hL == 0, 'L1:1sign+number', 1),
             'M327+X_header_top(-)': lead(yt, hL == 1, 'L1first:|M327+X|', -1),
             'L2_no_numerals_headed_bottom(-)': lead(yb, hL == 1, 'L2:no_numerals', -1),
             'illegible_header_shaped_unheaded_top(-)': lead(yt, hL == 0, 'L1:header_shaped_illegible', -1),
             'illegible_header_shaped_unheaded_bottom(+)': lead(yb, hL == 0, 'L1:header_shaped_illegible', 1)}
out['3e_n'] = dict(all=len(RL), heldout=int(ho.sum()))
print('3e', json.dumps(out['3e'], indent=0), flush=True)
json.dump(out, open(os.path.join(Q.CK, 'c3.json'), 'w'), indent=1)
