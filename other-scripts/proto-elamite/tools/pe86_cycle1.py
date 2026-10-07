"""pe86 cycle 1: disagreement scan between clay (square top) and text header definition."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe86_common as Q
rng = np.random.default_rng(8601)
D = Q.build(True); R = D['R']; hd = D['hd']; s = D['strata']
F = Q.line_feats(R)

def scan(y, sub, F, B=500, tag=''):
    keys = [k for k, v in F.items() if v[sub].sum() >= 5 and (1 - v[sub]).sum() >= 5]
    X = np.column_stack([F[k][sub] for k in keys]); X = (X - X.mean(0)) / X.std(0)
    def rs(yy):
        a = yy[sub]; a = (a - a.mean()) / a.std(); return X.T @ a / len(a)
    r0 = rs(y); mx = []
    cnt = np.zeros(len(keys))
    for _ in range(B):
        rp = rs(Q.perm_within(y, s, rng)); mx.append(np.abs(rp).max()); cnt += np.abs(rp) >= np.abs(r0)
    line = float(np.quantile(mx, .95))
    order = np.argsort(-np.abs(r0))
    res = [dict(f=keys[i], n=int(F[keys[i]][sub].sum()), r=round(float(r0[i]), 3), p=round((cnt[i] + 1) / (B + 1), 4)) for i in order[:15]]
    return dict(n=int(sub.sum()), nfeat=len(keys), fw95=round(line, 3), top=res, passing=[x for x in res if abs(x['r']) > line])

out = {}
U = hd == 0; H = hd == 1
out['1a_unheaded_top'] = scan(D['ytop'], U, F)
out['1a_headed_top'] = scan(D['ytop'], H, F)
out['1d_unheaded_bot'] = scan(D['ybot'], U, F)
out['1d_headed_bot'] = scan(D['ybot'], H, F)
print(json.dumps({k: (v['fw95'], v['top'][:5]) for k, v in out.items()}, indent=0), flush=True)
# 1b
y = D['ytop']; yb = D['ybot']; g = F['L1:header_shaped_illegible'] == 1
grp = {'headed': H, 'illegible_header_shaped': U & g, 'other_unheaded': U & ~g}
out['1b'] = {k: dict(n=int(m.sum()), top=round(float(np.mean(y[m])), 1), bot=round(float(np.mean(yb[m])), 1)) for k, m in grp.items()}
# permutation p: illegible group vs other unheaded, stratified shuffle of y
d0 = y[U & g].mean() - y[U & ~g].mean(); c = 0
for _ in range(2000):
    yp = Q.perm_within(y, s, rng); c += (yp[U & g].mean() - yp[U & ~g].mean()) >= d0
out['1b']['p_one_sided_vs_other_unheaded'] = round((c + 1) / 2001, 4)
# does reclassifying them as headed raise the overall header r?
import pe83_common as P
hd2 = hd.copy(); hd2[U & g] = 1
out['1b']['r_top_text_def'] = P.partial(D['top'], hd, D['Z'])['r']; out['1b']['r_top_with_illegible_as_headed'] = P.partial(D['top'], hd2, D['Z'])['r']
print('1b', out['1b'], flush=True)
# 1c planted
pl = []
Ui = np.where(U)[0]
for k in range(5):
    w = (np.argsort(np.argsort(y[Ui])) + 1.0) ** 2; w /= w.sum()
    pick = rng.choice(Ui, 15, replace=False, p=w)
    F2 = dict(F); v = np.zeros(len(R)); v[pick] = 1; F2['L1tok:PLANT'] = v
    sc = scan(y, U, F2, B=200)
    pl.append(dict(rank=[x['f'] for x in sc['top']].index('L1tok:PLANT') if 'L1tok:PLANT' in [x['f'] for x in sc['top']] else 99,
                   passed=any(x['f'] == 'L1tok:PLANT' for x in sc['passing']), r=[x['r'] for x in sc['top'] if x['f'] == 'L1tok:PLANT']))
out['1c_planted'] = pl
print('1c', pl, flush=True)
# disagreement lists
def lst(m, asc):
    idx = np.where(m)[0]; idx = idx[np.argsort(y[idx])][: 15] if asc else idx[np.argsort(-y[idx])][:15]
    return [dict(id=R[i]['id'], l1=R[i]['t']['lines'][0]['raw'], l2=R[i]['t']['lines'][1]['raw'] if len(R[i]['t']['lines']) > 1 else '', ytop=round(float(y[i]), 1)) for i in idx]
out['square_unheaded'] = lst(U, False); out['round_headed'] = lst(H, True)
json.dump(out, open(os.path.join(Q.CK, 'c1.json'), 'w'), indent=1)
