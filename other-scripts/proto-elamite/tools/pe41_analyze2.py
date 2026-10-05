"""pe41 cycle-2 analysis: target-half replication against shuffled halves, PC calibration, held-out recovery, economy fit.
usage: python3 pe41_analyze2.py OUT.json META.json runs.jsonl.gz [...]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import LABELS, STATN, load_real, corpus_stats, sample_params, Population, shuffle_corpus
from pe41_analyze import load, votes, PC_KNOWN, heldout_acc, pc_control

NL = len(LABELS)


def enrich(V):
    base = V.sum(0) / max(V.sum(), 1)
    n = V.sum(1, keepdims=True)
    sh = V / np.maximum(n, 1)
    z = (sh - base) / np.sqrt(base * (1 - base) / np.maximum(n, 1) + 1e-12)
    return z, base


def replicate(runs, meta, a, b, zmin=3.0, minv=30, seed=0):
    sa, sb = meta[a]['signs'], meta[b]['signs']
    VA, VB = votes(runs, a, len(sa)), votes(runs, b, len(sb))
    zA, _ = enrich(VA); zB, _ = enrich(VB)
    common = [s for s in sa if s in sb and VA[sa.index(s)].sum() >= minv and VB[sb.index(s)].sum() >= minv]
    ia = [sa.index(s) for s in common]; ib = [sb.index(s) for s in common]
    ta, tb = zA[ia].argmax(1), zB[ib].argmax(1)
    agree = float(np.mean(ta == tb)) if common else 0.0
    rng = np.random.default_rng(seed)
    perm = [np.mean(ta == rng.permutation(tb)) for _ in range(1000)]
    # correlation of enrichment profiles, sign by sign
    cors = [np.corrcoef(zA[i], zB[j])[0, 1] for i, j in zip(ia, ib)]
    zz = np.minimum(zA[ia], zB[ib])
    rep = []
    for k, s in enumerate(common):
        l = int(zz[k].argmax())
        if zz[k, l] >= zmin:
            rep.append(dict(sign=s, label=LABELS[l], zmin=float(zz[k, l])))
    return dict(n=len(common), agree=agree, perm=float(np.mean(perm)), p=float(np.mean(np.array(perm) >= agree)),
                mean_cor=float(np.nanmean(cors)) if cors else 0.0, n_rep=len(rep), rep=sorted(rep, key=lambda r: -r['zmin']))


def econ(runs, target_stats, scale, q=0.05):
    S = np.array([r['stats'] for r in runs])
    d = np.sqrt((((S - target_stats) / scale) ** 2).mean(1))
    best = d <= np.quantile(d, q)
    out = {}
    def put(k, x):
        out[k] = dict(best=float(np.mean(x[best])), all=float(np.mean(x)), z=float((np.mean(x[best]) - np.mean(x)) / (np.std(x) / np.sqrt(best.sum()) + 1e-12)))
    for k, v in runs[0]['p'].items():
        if isinstance(v, (int, float)):
            put(k, np.array([r['p'][k] for r in runs], float))
    for j, n in enumerate(['ration', 'delivery', 'labor', 'inventory', 'debt']):
        put('mix_' + n, np.array([r['p']['mix'][j] for r in runs]))
    for j, n in enumerate(['ration', 'delivery', 'labor', 'inventory', 'debt']):
        put('named_' + n, np.array([r['p']['p_named'][j] for r in runs]))
    put('L', np.array([r['L'] for r in runs], float))
    for k in runs[0]['pol']:
        put('pol_' + k, np.array([r['pol'][k] for r in runs], float))
    return out, d, best


if __name__ == '__main__':
    outf, metaf = sys.argv[1], sys.argv[2]
    runs = load(sys.argv[3:]); meta = json.load(open(metaf))
    R = dict(n_pop=len(runs))
    for a, b in [('PEA', 'PEB'), ('SHUF0A', 'SHUF0B'), ('PCA', 'PCB'), ('PCSHA', 'PCSHB')]:
        R['rep_' + a[:-1]] = replicate(runs, meta, a, b)
        r = R['rep_' + a[:-1]]
        print('replication', a[:-1], 'n', r['n'], 'agree %.3f perm %.3f p %.3f meancor %.3f replicated votes %d' % (r['agree'], r['perm'], r['p'], r['mean_cor'], r['n_rep']))
    R['heldout'] = heldout_acc(runs, meta, 1.0)
    ha = R['heldout']
    print('heldout acc %.3f perm %.3f maj %.3f | bal %.3f perm %.3f, p<0.05 %d/%d' % (np.mean([h['acc'] for h in ha]), np.mean([h['perm_mean'] for h in ha]), np.mean([h['majority'] for h in ha]), np.mean([h['bal'] for h in ha]), np.mean([h['bal_perm'] for h in ha]), sum(h['bal_p'] < 0.05 for h in ha), len(ha)))
    R['pc'] = pc_control(runs, meta, 1.0)
    for k, v in R['pc'].items():
        print('PC', k, {a: round(b, 3) for a, b in v.items()})
    # PC replicated votes on the known lists
    kn = {s: c for c, l in PC_KNOWN.items() for s in l}
    pcr = [(r['sign'], r['label'], kn.get(r['sign'])) for r in R['rep_PC']['rep']]
    hit = sum(1 for s, l, c in pcr if c and c == l); tot = sum(1 for s, l, c in pcr if c)
    print('PC replicated votes on known signs: %d/%d correct' % (hit, tot), [x for x in pcr if x[2]])
    R['pc_rep_known'] = dict(hit=hit, n=tot, rows=pcr)
    # economy fit by corpus shape
    S = np.array([r['stats'] for r in runs]); scale = S.std(0) + 1e-9
    pe, _ = load_real('PE'); st = corpus_stats(pe)
    e, d, best = econ(runs, st, scale)
    R['econ_PE'] = e; R['econ_PE_dist'] = dict(best=float(d[best].mean()), median=float(np.median(d)))
    print('PE shape distance: best5%% %.3f median %.3f' % (d[best].mean(), np.median(d)))
    for k in sorted(e, key=lambda k: -abs(e[k]['z']))[:16]:
        print('econ', k, round(e[k]['best'], 3), 'vs', round(e[k]['all'], 3), 'z %.1f' % e[k]['z'])
    # control: does the same shape-matching recover a held-out economy's own parameters?
    hold = []
    for j in range(8):
        s = 900000 + j; p = sample_params(np.random.default_rng(s))
        P = Population(p, s); tabs, _ = P.write_corpus()
        e2, d2, b2 = econ(runs, corpus_stats(tabs), scale)
        for k in e2:
            if k in p and isinstance(p[k], (int, float)):
                x = np.array([r['p'][k] for r in runs], float)
                # truth's percentile within the matched set vs within the prior
                hold.append(dict(h=j, k=k, err_best=abs(e2[k]['best'] - p[k]) / (x.std() + 1e-9), err_prior=abs(x.mean() - p[k]) / (x.std() + 1e-9)))
    import collections
    agg = collections.defaultdict(list)
    for h in hold:
        agg[h['k']].append(h['err_prior'] - h['err_best'])
    R['econ_recovery'] = {k: float(np.mean(v)) for k, v in agg.items()}
    good = sorted(agg, key=lambda k: -np.mean(agg[k]))
    print('econ recovery (gain in sd units, >0 = matched set closer to truth than prior):', [(k, round(float(np.mean(agg[k])), 2)) for k in good[:12]])
    print('worst:', [(k, round(float(np.mean(agg[k])), 2)) for k in good[-6:]])
    for r in R['rep_PE']['rep'][:40]:
        print('PE rep', r)
    for r in R['rep_SHUF0']['rep'][:10]:
        print('SHUF rep', r)
    json.dump(R, open(outf, 'w'), indent=1, default=str)
