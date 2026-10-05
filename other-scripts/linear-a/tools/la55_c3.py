#!/usr/bin/env python3
"""LA-55 cycle 3: Heaps-type growth per word class across archives (classes defined independently of scaling:
la45 classes for Linear A, known classes for LB / Ur III), real vs N1 / N3 re-dealt archives; predicted contents
of a new archive of size N; site-size covariate (settlement hectares from open sources).
usage: la55_c3.py CORPUS NNULL"""
import sys, ast, re, collections, json
import numpy as np
from la55_common import *

corpus = sys.argv[1]; NN = int(sys.argv[2])
rng = np.random.default_rng(seed('la55-c3-' + corpus))


def la45_classes():
    la45 = {}
    for line in open(os.path.join(D, 'la45_ckpt', 'c1_report_LA.txt')):
        m = re.match(r'\s+(HEADER|COMMODITY|ENTRY|RARE) (\[.*\])\s*$', line)
        if m:
            for t in ast.literal_eval(m.group(2)):
                w = t[0]
                k = 'L:' + w[2:].split('+')[0] if w.startswith('L:') else ('w:' + w if '-' in w else 's:' + w)
                la45.setdefault(k, m.group(1))
    return la45


docs, mk = build_corpus(corpus)
data = Data(docs, min_unit=3, min_k=1)         # all types: Heaps counts every type
if corpus.startswith('LA'):
    truth = la45_classes()
    # words never typed by la45: split by form
    for t in data.types:
        if t not in truth:
            truth[t] = 'OTHER_LOGO' if t.startswith('L:') else ('OTHER_SIGN' if t.startswith('s:') else 'OTHER_WORD')
elif corpus.startswith('LB'):
    truth = lb_truth_classes()
else:
    truth = ur_truth_classes(data.docs)
cls_names = sorted(set(truth.get(t) for t in data.types if truth.get(t)))
mask = {c: np.array([truth.get(t) == c for t in data.types]) for c in cls_names}


def heaps(u):
    K, N = data.counts(u)
    keep = N >= 3
    x = np.log(N[keep]); out = {}
    for c in cls_names:
        V = (K[mask[c]][:, keep] > 0).sum(0).astype(float)
        ok = V > 0
        if ok.sum() < 4:
            out[c] = (np.nan, np.nan); continue
        # slope of log(V+1) on log N (robust to zeros); and mean V at the median archive
        h = np.polyfit(x, np.log(V + 1), 1)[0]
        out[c] = (h, V.sum() / keep.sum())
    return out


real = heaps(data.u)
n1 = [heaps(data.shuffle(rng)) for _ in range(NN)]
n3 = [heaps(data.shuffle(rng, True)) for _ in range(NN)]
res = {'corpus': corpus, 'classes': {}}
print('==', corpus, 'docs', len(data.docs), 'units', data.A)
for c in cls_names:
    h = real[c][0]
    a1 = np.array([r[c][0] for r in n1]); a3 = np.array([r[c][0] for r in n3])
    row = dict(n=int(mask[c].sum()), h=h, N1=(np.nanmean(a1), np.nanstd(a1)), N3=(np.nanmean(a3), np.nanstd(a3)),
               p_lo1=float((np.sum(a1 <= h) + 1) / (NN + 1)), p_lo3=float((np.sum(a3 <= h) + 1) / (NN + 1)),
               p_hi1=float((np.sum(a1 >= h) + 1) / (NN + 1)), p_hi3=float((np.sum(a3 >= h) + 1) / (NN + 1)))
    res['classes'][c] = row
    print('  %-12s n %5d  h %.3f  N1 %.3f+-%.3f (lo %.3f hi %.3f)  N3 %.3f+-%.3f (lo %.3f hi %.3f)' % (
        c, row['n'], h, row['N1'][0], row['N1'][1], row['p_lo1'], row['p_hi1'], row['N3'][0], row['N3'][1],
        row['p_lo3'], row['p_hi3']))
# relative growth: h_c / h_N1(c) ("deficit"), ranks of classes
res['rel'] = {c: res['classes'][c]['h'] / res['classes'][c]['N3'][0] for c in cls_names}
print('  h / h_N3:', {c: round(v, 3) for c, v in res['rel'].items()})

if corpus == 'LA':
    # ---- predictions: expected types present in a new archive of N documents (shrunk scaling fit, words K>=3)
    d3 = Data(docs, min_unit=3, min_k=3)
    K, N = d3.counts(); x = np.log(N)
    b = fit_beta(K, N)
    mu = np.exp(np.log(K.sum(1) / np.exp(b[:, None] * x[None, :]).sum(1))[:, None] + b[:, None] * x[None, :])
    info = (mu * (x[None, :] - (mu * x).sum(1, keepdims=True) / mu.sum(1, keepdims=True)) ** 2).sum(1)
    w = info / (info + 4); bs = 1 + w * (b - 1)
    al = np.log(K.sum(1) / np.exp(bs[:, None] * x[None, :]).sum(1))
    c1 = json.load(open(os.path.join(CK, 'c1_LA.json')))
    assert c1['types'] == d3.types
    cls3 = np.array(c1['N3']['cls'])
    pred = {}
    for n in (5, 10, 20, 50, 100, 200, 500, 1000):
        m = np.exp(al + bs * math.log(n)); p = 1 - np.exp(-m)
        p0 = 1 - np.exp(-K.sum(1) / N.sum() * n)
        row = {'all': float(p.sum()), 'all_prop': float(p0.sum())}
        for c in ('SATURATING', 'LINEAR', 'THRESHOLD', 'SUPERLINEAR'):
            row[c] = float(p[cls3 == c].sum())
        row['likely'] = [(d3.types[i], round(float(p[i]), 2), round(float(p0[i]), 2), cls3[i])
                         for i in np.argsort(-p)[:25]]
        pred[n] = row
        print('  new archive N=%4d: expected types present %.1f (proportional %.1f) by class %s' % (
            n, row['all'], row['all_prop'], {c: round(row[c], 1) for c in ('SATURATING', 'LINEAR', 'THRESHOLD', 'SUPERLINEAR')}))
    for n in (20, 100):
        print('   likely at N=%d:' % n, pred[n]['likely'][:15])
    res['pred'] = pred
    res['beta_shrunk'] = dict(zip(d3.types, bs.tolist())); res['beta_raw'] = dict(zip(d3.types, b.tolist()))

if corpus == 'LA':
    # ---- site size covariate (Neopalatial settlement extent, ha; open sources, see la55_final.txt)
    HA = {'Knossos': 100, 'Malia': 55, 'Phaistos': 55, 'Palaikastro': 17.5, 'Thera': 20, 'Gournia': 1.7}
    sd = la_units(level='site')
    by = collections.defaultdict(list)
    for d in sd: by[d['site']].append(d)
    rows = []
    for s, h in HA.items():
        ds = by.get(s, [])
        tok = collections.Counter(t for d in ds for t in d['terms'])
        rows.append((s, h, len(ds), len(tok), len([t for t in tok if t.startswith('L:')])))
    from scipy.stats import spearmanr
    import itertools
    ha = np.array([r[1] for r in rows]); out = {}
    for j, name in ((2, 'documents'), (3, 'types'), (4, 'logogram types')):
        v = np.array([r[j] for r in rows], float)
        rho = spearmanr(ha, v).correlation
        perms = [spearmanr(ha, np.array(p)).correlation for p in itertools.permutations(v)]
        p = float(np.mean(np.array(perms) >= rho - 1e-12))
        sl = np.polyfit(np.log(ha), np.log(v + 1), 1)[0]
        out[name] = dict(rho=rho, p=p, slope=sl)
        print('  site size vs %s: rho %.2f (exact perm p %.3f), log-log slope %.2f' % (name, rho, p, sl))
    # type richness corrected for documents: residual of log types on log docs vs ha
    lt = np.log([r[3] + 1 for r in rows]); ld = np.log([r[2] + 1 for r in rows])
    resid = lt - np.polyval(np.polyfit(ld, lt, 1), ld)
    rho = spearmanr(ha, resid).correlation
    perms = [spearmanr(ha, np.array(p)).correlation for p in itertools.permutations(resid)]
    out['types|docs'] = dict(rho=rho, p=float(np.mean(np.array(perms) >= rho - 1e-12)))
    print('  site size vs types given documents: rho %.2f p %.3f' % (rho, out['types|docs']['p']))
    print('  rows', rows)
    res['sites'] = dict(rows=rows, tests=out)
jdump(res, 'c3_%s.json' % corpus)
