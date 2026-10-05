#!/usr/bin/env python3
"""la51: summarise cycle-1/2 JSONs. usage: la51_report.py CORPUS"""
import json, os, sys, numpy as np, collections
import la51_common as L
corpus = sys.argv[1]
J = lambda p: json.load(open(os.path.join(L.CK, f'c1_{corpus}_{p}.json')))
out = {}
try:
    real = J('real')['runs']
    out['real_surv'] = [r['survivors'] for r in real]
    out['real_terms'] = [len(r['surv_terms']) for r in real]
    out['n'] = (real[0]['n_docs'], real[0]['n_deps'], real[0]['n_terms'], real[0]['n_links'])
    cnt = collections.Counter(t for r in real for t in r['surv_terms'])
    best = collections.defaultdict(list)
    for r in real:
        for b in r['best']: best[b[0]].append((b[1], b[2], round(b[3], 2)))
except FileNotFoundError:
    real = None
for nk in ('n1', 'n2'):
    try:
        runs = J(nk)['runs']
    except FileNotFoundError:
        continue
    s = np.array([r['survivors'] for r in runs]); t = np.array([len(r['surv_terms']) for r in runs])
    tc = collections.Counter(x for r in runs for x in r['surv_terms'])
    out[nk] = dict(surv_mean=float(s.mean()), surv_max=int(s.max()), terms_mean=float(t.mean()),
                   P_surv=float(((s >= np.mean(out.get('real_surv', [0]))).sum() + 1) / (len(s) + 1)),
                   P_terms=float(((t >= np.mean(out.get('real_terms', [0]))).sum() + 1) / (len(t) + 1)),
                   reps=len(runs), term_rate={k: v / len(runs) for k, v in tc.items()})
if real:
    rows = []
    for term, c in cnt.most_common():
        r1 = out.get('n1', {}).get('term_rate', {}).get(term, 0.0)
        r2 = out.get('n2', {}).get('term_rate', {}).get(term, 0.0)
        rows.append((term, c, len(real), r1, r2, best[term][0]))
    out['terms'] = rows
try:
    pl = J('plant')['runs']
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in pl:
        a = agg[(r['cls'], r['p_in'])]; a[0] += 1; a[1] += r['any']; a[2] += r['exact']
    out['plant'] = {f'{k[0]}@{k[1]}': v for k, v in agg.items()}
except FileNotFoundError:
    pass
print(json.dumps({k: v for k, v in out.items() if k not in ('n1', 'n2')}, indent=0, default=str, ensure_ascii=False))
for nk in ('n1', 'n2'):
    if nk in out: print(nk, {k: v for k, v in out[nk].items() if k != 'term_rate'})
json.dump(out, open(os.path.join(L.CK, f'report_{corpus}.json'), 'w'), indent=1, default=str, ensure_ascii=False)
