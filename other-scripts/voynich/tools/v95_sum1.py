"""summarise cycle-1 checkpoints: per corpus, select top hypotheses on TRAIN excess (real minus twin mean), report TEST."""
import json, sys, os, numpy as np
import v95_lib as L, v95_c1 as C

def table(name, top=20):
    d = json.load(open(os.path.join(L.CK, 'c1_%s.json' % name)))
    H = C.hypotheses(d['nh'])
    rs = []
    for r, h in zip(d['rows'], H):
        if r['real'] is None or any(r['tw%d' % s] is None for s in range(C.NTWIN)): continue
        tr = r['real'][0]; te = r['real'][1]
        if not np.all(np.isfinite([tr, te])): continue
        cl = lambda v: float(np.clip(v, -3, 1)) if np.isfinite(v) else -3.0
        ttr = [cl(r['tw%d' % s][0]) for s in range(C.NTWIN)]; tte = [cl(r['tw%d' % s][1]) for s in range(C.NTWIN)]
        if not np.all(np.isfinite([tr, te] + ttr + tte)): continue
        ex_tr = tr - np.mean(ttr); ex_te = te - np.mean(tte); sd = np.std(tte, ddof=1) + 0.02
        rs.append(dict(i=r['i'], key=h['key'], mode=h['mode'], m=len(h['groups']), Gtr=tr, Gte=te, ex_tr=ex_tr, ex_te=ex_te,
                       z_te=ex_te / sd, twte=np.mean(tte), c=r.get('c')))
    rs.sort(key=lambda x: -x['Gtr'])            # selection: the real corpus's own train-half momentum
    return rs[:top], rs

if __name__ == '__main__':
    for n in sys.argv[1:]:
        t, allr = table(n)
        zt = [x['z_te'] for x in t]
        ok = sum((x['Gte'] >= 0.2) and (x['ex_te'] >= 0.5) and (x['z_te'] >= 3) for x in t)
        print('%-8s n=%d  top20(by train G): test G median %+.2f  max %+.2f | pass (G>=0.2, ex>=0.5, z>=3) %d/20 | max held-out G of all %+.2f | best: %s %s G %.2f/%.2f' % (
            n, len(allr), np.median([x['Gte'] for x in t]), max(x['Gte'] for x in t), ok,
            max(x['Gte'] for x in allr), t[0]['key'], t[0]['mode'], t[0]['Gtr'], t[0]['Gte']))
