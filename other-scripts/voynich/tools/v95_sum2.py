"""summarise cycle-2 checkpoints: top-20 fit hypotheses by TRAIN under-dispersion (lowest R), held-out log R vs twins 4-7;
guess-mode legality (register machine) real vs twins."""
import json, sys, os, numpy as np
import v95_lib as L, v95_c2 as C


def table(name, top=20):
    d = json.load(open(os.path.join(L.CK, 'c2_%s.json' % name)))
    H = C.hypotheses(d['nh']); rs = []; lg = []
    for r, h in zip(d['rows'], H):
        if 'te' not in r: continue
        te = np.array(r['te'])
        if not np.all(np.isfinite(te)) or te[5:].mean() <= 0: continue
        null = te[5:9]
        lr = np.log(te[0] / null.mean())
        lt = [np.log(te[k] / np.mean([te[j] for j in range(5, 9) if j != k])) for k in range(5, 9)]
        z = lr / (np.std(lt, ddof=1) + 0.05)
        x = dict(i=r['i'], unit=h['unit'], mode=h['mode'], m=len(h['groups']), Rtr=r['Rtr'], logR=lr, z=z, c=r['c'])
        rs.append(x)
        if 'legal' in r: lg.append((r['legal'][0] - np.mean(r['legal'][5:9]), r['legal'][0], np.mean(r['legal'][5:9]), r['i']))
    fit = sorted([x for x in rs if x['mode'] == 'fit'], key=lambda x: x['Rtr'])[:top]
    return fit, rs, sorted(lg, reverse=True)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        fit, rs, lg = table(n)
        zs = [x['z'] for x in fit]
        print('%-8s top20 fit: train R median %.2f | held-out logR median %+.2f, z<=-3 & logR<=-0.2: %d/20 | min logR(all) %+.2f '
              '| units %s | legality best excess %+.3f (real %.3f twins %.3f)' % (
                  n, np.median([x['Rtr'] for x in fit]), np.median([x['logR'] for x in fit]),
                  sum((x['z'] <= -3) and (x['logR'] <= -0.2) for x in fit), min(x['logR'] for x in rs),
                  ''.join(x['unit'][1] for x in fit), lg[0][0] if lg else np.nan, lg[0][1] if lg else np.nan,
                  lg[0][2] if lg else np.nan))
