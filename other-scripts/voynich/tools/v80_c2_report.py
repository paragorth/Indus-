"""v80 cycle 2 report: row-HMM held-out gain over the edge law, by K; excess over the interior-shuffle null."""
import json, os, glob, sys
import v80_lib as L
d = os.path.join(L.CK, sys.argv[1] if len(sys.argv) > 1 else 'c2')
out = {}
for fn in sorted(glob.glob(os.path.join(d, '*.json'))):
    nm = os.path.basename(fn)[:-5]
    if nm == 'report': continue
    R = json.load(open(fn))
    rec = {}
    for rep in ('pre2', 'full'):
        prof = {}
        for K in sorted(set(r['K'] for r in R)):
            rs = [r for r in R if r['rep'] == rep and r['K'] == K]
            b = max(rs, key=lambda r: r['ins'])          # restart chosen in-sample
            prof[K] = b['gain']
        g1 = prof[1]; best = max(prof, key=lambda k: prof[k])
        thr = g1 + 0.9 * (prof[best] - g1)
        khat = min(k for k in prof if prof[k] >= thr)
        rec[rep] = dict(prof=prof, best=prof[best], over1=prof[best] - g1, K=best, khat=khat)
    out[nm] = rec
for nm, rec in out.items():
    base = nm.replace('_sh', '')
    ex = ''
    if nm.endswith('_sh') is False and (nm + '_sh') in out:
        ex = ' | excess pre2 %+.4f full %+.4f' % (rec['pre2']['over1'] - out[nm + '_sh']['pre2']['over1'],
                                                   rec['full']['over1'] - out[nm + '_sh']['full']['over1'])
    print('%-14s ' % nm + '  '.join('%s: K=1 %.4f best %.4f (K%d, khat %d, over K1 %+.4f)' % (
        rep, r['prof'][1], r['best'], r['K'], r['khat'], r['over1']) for rep, r in rec.items()) + ex)
json.dump(out, open(os.path.join(d, 'report.json'), 'w'))
