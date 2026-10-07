"""pe85 cycle 3: CHOOSE vs MAKE simulation, gravity (stand-to-dry) model, neighbour concordance re-test on the line-art
record and held-out volumes, held-out line-art-only tablets for cycle-1 candidates."""
import sys, os, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe85_common as C
import pe83_common as P
import common
from pe85_cycle1 import setup, corr

rng = np.random.default_rng(853)


def neighbour(y, R, vol, B=2000):
    """correlation of y between publication-order neighbours inside a volume; null: order shuffled within volume."""
    order = {}
    for v in set(vol):
        ii = [i for i in range(len(R)) if vol[i] == v and np.isfinite(R[i]['pub'])]
        order[v] = sorted(ii, key=lambda i: R[i]['pub'])
    def stat(perm):
        a, b = [], []
        for v, ii in perm.items():
            for i, j in zip(ii[:-1], ii[1:]):
                a.append(y[i]); b.append(y[j])
        return corr(np.array(a), np.array(b))
    s0 = stat(order); null = []
    for _ in range(B):
        null.append(stat({v: list(rng.permutation(ii)) for v, ii in order.items()}))
    return dict(r=round(s0, 3), null_q95=round(float(np.quantile(null, .95)), 3), p=round((np.sum(np.array(null) >= s0) + 1) / (B + 1), 4))


out = {}
R, hd, Z, A, vol, strata = setup(True)
Z0 = Z[:, :-1]  # covariates without header
top = C.col(R, 'cf_top'); bot = C.col(R, 'cf_bot')

# ---- 3a CHOOSE vs MAKE ---------------------------------------------------------------------------------------
# CHOOSE: the writer picks the squarer of the two ends as the header end (prob q), else random; ends themselves unchanged.
# MAKE: the header end is squared (shifted) when a header is written; the other end untouched.
# Simulate on headless tablets' unordered end pairs, matched to headed tablets by covariate stratum.
obs = dict(top=P.partial(top, hd, Z0)['r'], bot=P.partial(bot, hd, Z0)['r'])
s = P.strata(Z0)
H = np.where(hd == 1)[0]; N = np.where(hd == 0)[0]
bystr = {k: N[s[N] == k] for k in np.unique(s)}


def sim(mode, q, B=200):
    rt, rb = [], []
    for _ in range(B):
        t2, b2 = top.copy(), bot.copy()
        for i in H:
            pool = bystr.get(s[i]);
            j = rng.choice(pool) if pool is not None and len(pool) else rng.choice(N)
            a, b = top[j], bot[j]
            if mode == 'choose':
                hi, lo = max(a, b), min(a, b)
                if rng.random() < q:
                    t2[i], b2[i] = hi, lo
                else:
                    t2[i], b2[i] = (a, b) if rng.random() < .5 else (b, a)
            else:
                t2[i], b2[i] = min(1.0, a + q), b
        # headless keep their own ends but random orientation under choose? no: headless orientation fixed by text
        rt.append(P.partial(t2, hd, Z0)['r']); rb.append(P.partial(b2, hd, Z0)['r'])
    return np.array(rt), np.array(rb)


res = {'observed': obs, 'n_headed': int(len(H)), 'n_headless': int(len(N))}
for mode, grid in (('choose', [0.3, 0.5, 0.7, 0.9, 1.0]), ('make', [0.04, 0.06, 0.08, 0.10, 0.12])):
    g = {}
    for q in grid:
        rt, rb = sim(mode, q, B=60)
        g[str(q)] = dict(top=[round(float(np.quantile(rt, .025)), 3), round(float(rt.mean()), 3), round(float(np.quantile(rt, .975)), 3)],
                         bot=[round(float(np.quantile(rb, .025)), 3), round(float(rb.mean()), 3), round(float(np.quantile(rb, .975)), 3)])
    res[mode] = g
# headless end asymmetry (orientation fixed by reading) vs proto-cuneiform pipeline baseline
pc = json.load(open(os.path.join(P.CK, 'protocun.json')))['rows']
pcd = np.array([r['cf_top'] - r['cf_bot'] for r in pc if r['cf_top'] is not None and r['cf_bot'] is not None and not r['hd']])
res['asym_top_minus_bot'] = dict(PE_headed=round(float(np.nanmean(top[H] - bot[H])), 3), PE_headless=round(float(np.nanmean(top[N] - bot[N])), 3),
                                 PC_headless=round(float(np.mean(pcd)), 3), PC_n=int(len(pcd)))
# planted check of the discriminator: make a CHOOSE world from the real headless pairs and see if the test calls it CHOOSE
out['3a_choose_vs_make'] = res
print(json.dumps(res, indent=0), flush=True)

# ---- 3b gravity (stand to dry on the header end): header effect grows with tablet mass --------------------------
mass = 1.5 * C.col(R, 'la') + np.log(C.col(R, 'tw'))  # log(h*w*t) with t = tw*w, w ~ sqrt(h*w)
ytop = C.resid(top, Z)
g = {}
for nm, mm in (('all', np.ones(len(R), bool)), ('MDP26', vol == 'MDP 26'), ('other', vol != 'MDP 26')):
    x = C.zs(mass) * (hd - hd.mean())
    r0 = corr(ytop[mm], x[mm]); nl = [corr(C.perm_within(ytop, strata, rng)[mm], x[mm]) for _ in range(1000)]
    g[nm] = dict(r=round(r0, 3), p_one=round((np.sum(np.array(nl) >= r0) + 1) / 1001, 4))
out['3b_gravity_photo'] = g
print('gravity', g, flush=True)

# ---- 3c neighbour re-test: held-out volumes and the line-art record ---------------------------------------------
nb = {}
for nm, mm in (('MDP26', vol == 'MDP 26'), ('other', vol != 'MDP 26')):
    idx = np.where(mm)[0]; RR = [R[i] for i in idx]
    nb[nm + '_top'] = neighbour(ytop[idx], RR, vol[idx]); nb[nm + '_bot'] = neighbour(C.resid(bot, Z)[idx], RR, vol[idx])
out['3c_neighbour_photo_split'] = nb
print('neighbour photo', nb, flush=True)

# line-art record over all tablets with drawings and intact line 1 (independent outline source)
LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json')))
cat = P.catalogue()
import pe71_lib as L
S = {r['id']: r for r in L.load()}
RL = []
for t in common.load():
    la = LA.get(t['id']); c = cat.get(t['id'])
    if not la or not t['lines'] or not c:
        continue
    l1 = t['lines'][0]
    if l1['lacuna'] or l1['damaged']:
        continue
    h, w, th = P.num(c['height']), P.num(c['width']), P.num(c['thickness'])
    s_ = S.get(t['id'], {})
    RL.append(dict(id=t['id'], hd=float(common.header(t) is not None), top=la['la_cf_top'], bot=la['la_cf_bot'], asp=np.log(la['la_aspect']) if la.get('la_aspect') else 0.0,
                   la=np.log(h * w) if h and w else np.nan, tw=(th / w) if th and w else np.nan, nl=np.log(len(t['lines'])),
                   dx=float(any(g == 'x' for l in t['lines'] for g in l['signs'])), rev=float(any(l['surface'] == 'reverse' for l in t['lines'])),
                   vol=s_.get('vol') or t.get('volume') or 'other', pub=s_.get('pub', np.nan), photo=t['id'] in set(r['id'] for r in R),
                   cap=None, tag=float(any(l['surface'] == 'top' for l in t['lines']))))
for r in RL:
    for k in ('la', 'tw'):
        if not np.isfinite(r[k]):
            r[k] = np.nanmedian([x[k] for x in RL])
ZL = np.column_stack([np.ones(len(RL))] + [np.array([r[k] for r in RL], float) for k in ('la', 'asp', 'tw', 'nl', 'dx', 'rev')])
hl = np.array([r['hd'] for r in RL]); tl = np.array([r['top'] for r in RL]); bl = np.array([r['bot'] for r in RL])
vl = np.array([r['vol'] for r in RL])
ZLh = np.column_stack([ZL, hl])
yl = C.resid(tl, ZLh); ylb = C.resid(bl, ZLh)
la_res = {'n': len(RL), 'header_top': P.partial(tl, hl, ZL, 1000, rng), 'header_bot': P.partial(bl, hl, ZL, 1000, rng),
          'neighbour_top': neighbour(yl, RL, vl), 'neighbour_bot': neighbour(ylb, RL, vl)}
# held-out tablets: line art only, no photo features (never in cycles 1-2)
ho = np.array([not r['photo'] for r in RL])
la_res['heldout_n'] = int(ho.sum()); la_res['heldout_headed'] = int(hl[ho].sum())
la_res['heldout_header_top'] = P.partial(tl[ho], hl[ho], ZL[ho], 2000, rng)
la_res['heldout_header_bot'] = P.partial(bl[ho], hl[ho], ZL[ho], 2000, rng)
x = np.array([r['rev'] for r in RL]) * (hl - hl.mean())
la_res['heldout_header_x_reverse_top'] = round(corr(yl[ho], x[ho]), 3)
la_res['photo_set_header_x_reverse_top_lineart'] = round(corr(yl[~ho], x[~ho]), 3)
massL = 1.5 * np.array([r['la'] for r in RL]) + np.log(np.array([r['tw'] for r in RL]))
xg = C.zs(massL) * (hl - hl.mean())
la_res['gravity_lineart_all'] = round(corr(yl, xg), 3); la_res['gravity_lineart_heldout'] = round(corr(yl[ho], xg[ho]), 3)
tg = np.array([r['tag'] for r in RL])
mh = hl == 1
la_res['tag_top_headed'] = dict(n_tag=int(tg[mh].sum()), diff=round(float(yl[mh][tg[mh] == 1].mean() - yl[mh][tg[mh] == 0].mean()), 3),
                                p_two=round(float((np.sum([abs(np.mean(yl[mh][p == 1]) - np.mean(yl[mh][p == 0])) >= abs(yl[mh][tg[mh] == 1].mean() - yl[mh][tg[mh] == 0].mean()) for p in (rng.permutation(tg[mh]) for _ in range(3000))]) + 1) / 3001), 4))
out['3c_lineart'] = la_res
print('lineart', json.dumps(la_res, indent=0), flush=True)
json.dump(out, open(os.path.join(C.CK, 'c3.json'), 'w'), indent=1, default=float)
