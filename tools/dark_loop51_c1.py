#!/usr/bin/env python3
"""Loop 51 cycle 1: origin tracing of the foreign-found texts.

For every Indus-script text found outside the Indus culture area: score against each home population
(Mohenjo-daro, Harappa, Lothal, Kalibangan, Dholavira, Chanhu-daro, other-Indus pooled) with
 (a) a per-site boundary bigram model with pooled backoff -> log likelihood ratio vs the pooled corpus
     (captures opener / closer menus, frozen pairs and site-local stock texts at once);
 (b) nearest-neighbour normalised edit distance to that site's texts;
 (c) exact copies at that site (S366 site-local reuse).
Assignment = argmax (a); (b) reported as a check. Calibration: leave-one-out assignment of every home text
(object removed; and all copies of the text removed). Null for the foreign set: 1,000 draws of equally many
home texts, length-matched, (i) from the corpus mix and (ii) from a uniform mix of the six named sites;
the chance distribution of per-site assignment counts and of the top-site share.
usage: python3 tools/dark_loop51_c1.py LEVEL [MODE] [--nperm N] [--nn-per-site N]
"""
import sys, os, json, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dark_loop51_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
MODE = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'W'
NPERM = int(sys.argv[sys.argv.index('--nperm') + 1]) if '--nperm' in sys.argv else 1000
NNPS = int(sys.argv[sys.argv.index('--nn-per-site') + 1]) if '--nn-per-site' in sys.argv else 80
rnd = random.Random(51)
OUT = f'data/derived/dark/loop51_c1_{LV}_{MODE}.txt'
LOG = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

M = mapper(MODE)
objs = load(LV)
for o in objs: o['m'] = M(o['seq'])
home = [o for o in objs if not o['foreign'] and not o['border'] and o['site'] != 'Unknown' and len(o['seq']) >= 1]
foreign_all = [o for o in objs if o['foreign']]
F = dedup([o for o in foreign_all if len(o['seq']) >= 2])
LAB = HOME_SITES + ['other-Indus']
P(f'# S-DARK-51 cycle 1, level {LV}, mode {MODE}: origin tracing of foreign-found texts')
P(f'foreign objects with >= 1 sign: {len(foreign_all)} (sites: {sorted(collections.Counter(o["site"] for o in foreign_all).items())})')
P(f'analysis set: {len(F)} distinct foreign texts of >= 2 signs; home population {len(home)} texts; '
  f'by site {dict(collections.Counter(home_label(o["site"]) for o in home))}')

by_site = collections.defaultdict(list)
for o in home: by_site[home_label(o['site'])].append(o['m'])
model = SiteModel({s: by_site[s] for s in LAB}, k=10.0)
pool_by_site = {s: [(o['idx'], o['m']) for o in home if home_label(o['site']) == s] for s in LAB}
copies = collections.Counter((home_label(o['site']), o['m']) for o in home)

def scores(t, exclude_site=None, exclude_text=None, exclude_idx=None, do_nn=True):
    lr = {}
    for s in LAB:
        ex = exclude_text if (exclude_site == s and exclude_text is not None) else None
        lr[s] = model.loglr(s, t, exclude=ex)
    best = max(LAB, key=lambda s: lr[s])
    srt = sorted(lr.values(), reverse=True)
    margin = srt[0] - srt[1]
    res = dict(lr=lr, best=best, margin=margin)
    if do_nn:
        d = {}
        for s in LAB:
            dn, arg, nt = nn(t, pool_by_site[s], exclude_idx=exclude_idx, cap=len(t), rnd=rnd)
            d[s] = (dn, arg, nt)
        res['nn'] = d; dmin = min(v[0] for v in d.values())
        tied = [s for s in LAB if abs(d[s][0] - dmin) <= 1e-12]
        res['nn_best'] = rnd.choice(tied); res['nn_tied'] = tied
    return res

# ---------------------------------------------------------------- foreign texts
P('\n== per-text scores (log LR vs pooled corpus; NN = normalised edit distance to nearest text at the site; copies = exact copies at home sites)')
assign = collections.Counter(); assign_nn = collections.Counter(); rows = []
for o in sorted(F, key=lambda o: (o['group'], o['site'])):
    t = o['m']
    r = scores(t)
    cp = {s: copies[(s, t)] for s in LAB if copies[(s, t)]}
    nnb = r['nn_best']; nnv = r['nn'][nnb]
    lrs = ' '.join('%s:%+.1f' % (s[:4], r['lr'][s]) for s in LAB)
    nns = '-'.join(str(x if not isinstance(x, tuple) else x[1]) for x in nnv[1][1]) if nnv[1] else ''
    P(f"  {o['group']:5s} {o['site']:20s} {o['shape']:8s} {'-'.join(map(str, o['seq'])):32s} -> {r['best']:12s} "
      f"(margin {r['margin']:.2f}; LR {lrs}) | NN {nnb} {nnv[0]:.2f} tied-sites {len(r['nn_tied'])} [{nns}] | copies {cp if cp else '-'}")
    assign[r['best']] += 1; assign_nn[nnb] += 1
    rows.append(dict(site=o['site'], group=o['group'], shape=o['shape'], seq=o['seq'], best=r['best'], margin=r['margin'],
                     lr=r['lr'], nn_best=nnb, nn_d=nnv[0], copies=cp))
P(f'\nassignment counts (bigram LR): {dict(assign)}')
P(f'assignment counts (NN):        {dict(assign_nn)}')
agree = sum(1 for r in rows if r['best'] == r['nn_best'])
P(f'LR and NN agree on {agree}/{len(rows)} texts')
for g in ['gulf', 'meso', 'iran', 'casia']:
    sub = [r for r in rows if r['group'] == g]
    if sub: P('  group %-5s n=%2d: LR %s; NN %s' % (g, len(sub), dict(collections.Counter(r['best'] for r in sub)), dict(collections.Counter(r['nn_best'] for r in sub))))
for sh in ['round', 'square', 'cylinder', 'pot', 'sealing']:
    sub = [r for r in rows if r['shape'] == sh]
    if sub: P('  shape %-8s n=%2d: LR %s; NN %s' % (sh, len(sub), dict(collections.Counter(r['best'] for r in sub)), dict(collections.Counter(r['nn_best'] for r in sub))))

# ---------------------------------------------------------------- leave-one-out calibration on home texts
P('\n== leave-one-out calibration (bigram LR), every home text of >= 2 signs')
home2 = [o for o in home if len(o['seq']) >= 2]
loo = []   # (true, assigned_obj_removed, assigned_text_removed, length)
for o in home2:
    s = home_label(o['site']); t = o['m']
    r1 = scores(t, exclude_site=s, exclude_text=t, do_nn=False)          # the object removed (one copy)
    # all copies at that site removed
    n = copies[(s, t)]
    for _ in range(n): model.add(s, t, -1)
    lr = {u: model.loglr(u, t) for u in LAB}
    for _ in range(n): model.add(s, t, +1)
    r2 = max(LAB, key=lambda u: lr[u])
    loo.append((s, r1['best'], r2, len(t), r1['margin']))
def confusion(loo, col):
    cm = collections.defaultdict(collections.Counter)
    for row in loo: cm[row[0]][row[col]] += 1
    return cm
for col, name in [(1, 'object removed'), (2, 'all copies of the text removed')]:
    cm = confusion(loo, col)
    acc = sum(cm[s][s] for s in LAB) / len(loo)
    assigned = collections.Counter(row[col] for row in loo)
    chance = sum((sum(cm[s].values()) / len(loo)) * (assigned[s] / len(loo)) for s in LAB)
    P(f'  [{name}] accuracy {acc:.3f} vs chance (marginal product) {chance:.3f}; assigned marginal {dict(assigned)}')
    for s in LAB:
        tot = sum(cm[s].values())
        if tot: P(f'     true {s:12s} n={tot:5d}: ' + ' '.join(f'{u[:4]} {cm[s][u]/tot:.2f}' for u in LAB) + f'  (recall {cm[s][s]/tot:.2f})')
    # precision: P(true = s | assigned = s)
    P('     precision: ' + ' '.join(f'{s[:4]} {sum(1 for r in loo if r[col]==s and r[0]==s)/max(1,assigned[s]):.2f}' for s in LAB))

# ---------------------------------------------------------------- null distributions for the foreign set
P(f'\n== null: {NPERM} draws of {len(F)} home texts, length-matched to the foreign set, assigned leave-one-out (object removed)')
loo_by_len = collections.defaultdict(list)          # length -> list of (true, assigned)
for row in loo: loo_by_len[row[3]].append(row)
loo_by_len_site = collections.defaultdict(list)
for row in loo: loo_by_len_site[(row[0], row[3])].append(row)
flens = [len(o['m']) for o in F]
def nearest_len_rows(L, store, key=lambda L: L):
    for dl in range(0, 10):
        for LL in (L - dl, L + dl):
            if store.get(key(LL)): return store[key(LL)]
    return []
null_counts = {s: [] for s in LAB}; null_top = []; null_top2 = []
null_counts_u = {s: [] for s in LAB}; null_top_u = []
for it in range(NPERM):
    c = collections.Counter(); cu = collections.Counter()
    for L in flens:
        row = rnd.choice(nearest_len_rows(L, loo_by_len)); c[row[1]] += 1
        s = rnd.choice(HOME_SITES)
        rows_s = nearest_len_rows(L, loo_by_len_site, key=lambda LL: (s, LL))
        if not rows_s: rows_s = nearest_len_rows(L, loo_by_len)
        cu[rnd.choice(rows_s)[1]] += 1
    for s in LAB: null_counts[s].append(c[s]); null_counts_u[s].append(cu[s])
    v = sorted(c.values(), reverse=True) + [0, 0]; null_top.append(v[0] / len(F)); null_top2.append((v[0] + v[1]) / len(F))
    vu = sorted(cu.values(), reverse=True) + [0]; null_top_u.append(vu[0] / len(F))
obs_sorted = sorted(assign.values(), reverse=True) + [0, 0]
P(f'observed: top-site share {obs_sorted[0]/len(F):.2f} ({assign.most_common(1)[0][0]}), top-2 share {(obs_sorted[0]+obs_sorted[1])/len(F):.2f}')
P(f'  corpus-mix null: top share {sum(null_top)/NPERM:.2f} [{sorted(null_top)[int(0.025*NPERM)]:.2f}-{sorted(null_top)[int(0.975*NPERM)]:.2f}] '
  f'P(obs >=) = {pval(obs_sorted[0]/len(F), null_top):.3f}; top-2 {sum(null_top2)/NPERM:.2f} P = {pval((obs_sorted[0]+obs_sorted[1])/len(F), null_top2):.3f}')
P(f'  uniform-site null: top share {sum(null_top_u)/NPERM:.2f} P = {pval(obs_sorted[0]/len(F), null_top_u):.3f}')
P('  per-site assignment counts, observed vs corpus-mix null mean [2.5-97.5%] P_hi P_lo; uniform null mean P_hi P_lo:')
for s in LAB:
    nc = sorted(null_counts[s]); nu = sorted(null_counts_u[s])
    P(f'    {s:12s} obs {assign[s]:2d} | mix {sum(nc)/NPERM:5.2f} [{nc[int(0.025*NPERM)]}-{nc[int(0.975*NPERM)]}] '
      f'P_hi {pval(assign[s], nc):.3f} P_lo {pval(assign[s], nc, "lo"):.3f} | uniform {sum(nu)/NPERM:5.2f} P_hi {pval(assign[s], nu):.3f} P_lo {pval(assign[s], nu, "lo"):.3f}')

# margin: are foreign texts assigned with less confidence than home texts of the same length?
fm = [r['margin'] for r in rows]; hm = [row[4] for row in loo]
null_m = []
for it in range(NPERM):
    null_m.append(sum(rnd.choice(nearest_len_rows(L, loo_by_len))[4] for L in flens) / len(F))
P(f'\nmean LR margin (best - second): foreign {sum(fm)/len(fm):.3f} vs length-matched home {sum(null_m)/NPERM:.3f} '
  f'[{sorted(null_m)[int(0.025*NPERM)]:.3f}-{sorted(null_m)[int(0.975*NPERM)]:.3f}] P_lo {pval(sum(fm)/len(fm), null_m, "lo"):.3f}')

# ---------------------------------------------------------------- NN calibration on a stratified sample
P(f'\n== NN calibration: {NNPS} home texts per site (>= 2 signs), nearest neighbour with the object itself excluded')
cmn = collections.defaultdict(collections.Counter); nn_rows = []
for s in LAB:
    cand = [o for o in home2 if home_label(o['site']) == s]
    for o in rnd.sample(cand, min(NNPS, len(cand))):
        d = {u: nn(o['m'], pool_by_site[u], exclude_idx=o['idx'], cap=len(o['m']), rnd=rnd)[0] for u in LAB}
        dmin = min(d.values()); b = rnd.choice([u for u in LAB if abs(d[u] - dmin) <= 1e-12]); cmn[s][b] += 1; nn_rows.append((s, b, len(o['m'])))
acc = sum(cmn[s][s] for s in LAB) / len(nn_rows)
P(f'  NN accuracy {acc:.3f} (uniform chance {1/len(LAB):.3f}); assigned marginal {dict(collections.Counter(b for _, b, _ in nn_rows))}')
for s in LAB:
    tot = sum(cmn[s].values())
    if tot: P(f'     true {s:12s} n={tot:3d}: ' + ' '.join(f'{u[:4]} {cmn[s][u]/tot:.2f}' for u in LAB))
nn_null = []
nn_by_len = collections.defaultdict(list)
for r in nn_rows: nn_by_len[r[2]].append(r)
for it in range(NPERM):
    c = collections.Counter(rnd.choice(nearest_len_rows(L, nn_by_len))[1] for L in flens)
    nn_null.append(sorted(c.values(), reverse=True)[0] / len(F))
nn_obs = assign_nn.most_common(1)[0]
P(f'  NN foreign top-site share {nn_obs[1]/len(F):.2f} ({nn_obs[0]}) vs uniform-site null {sum(nn_null)/NPERM:.2f} P = {pval(nn_obs[1]/len(F), nn_null):.3f}')

json.dump(dict(level=LV, mode=MODE, rows=[dict(r, seq=list(r['seq'])) for r in rows], assign=dict(assign), assign_nn=dict(assign_nn),
               loo_acc_obj=sum(1 for r in loo if r[0] == r[1]) / len(loo), loo_acc_text=sum(1 for r in loo if r[0] == r[2]) / len(loo)),
          open(OUT.replace('.txt', '.json'), 'w'), indent=0, default=str)
P(f'\nwritten {OUT}')
