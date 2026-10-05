"""pe52 cycle 3: are the one-off strings predictable from sign weights? (nested, no circularity)

Tablets are split in two halves H1/H2 (2 random halvings, both directions). On H1 alone the random-model
search (S=4 splits x M=80 models; the machine was shared) chooses DESCRIPTOR signs with rule B20 (pe52_ruleb.py, eff 0.2). A descriptor-only
ridge model is then fitted on H1 and scored on H2, separately for
  hapax strings  (multi-sign strings that occur once in the whole corpus: the 'name universe' of pe50)
  recurring strings (multi-sign strings seen >= 2 times)
  single-sign entries.
Score = relative error reduction vs the mode's baseline (0 = tablet / system mean), R = 1 - MSE_model / MSE_0.
Also: share of H2 distinct multi-sign strings whose every sign is an H1 descriptor (fully described) and the share
of their signs that are descriptors.
usage: python3 pe52_cycle3.py <corpus> <mode> <null>"""
import sys, os, json, random
import numpy as np
from collections import Counter
import pe52_lib as L
from pe52_run import corpus, NULLS
from pe52_ruleb import ruleb


def sub(C, tabs):
    return [r for r in C if r['tab'] in tabs]


def main(cname, mode, null):
    out = os.path.join(L.CK, f'c3_{cname}_{mode}_{null}.json')
    if os.path.exists(out):
        print(open(out).read()); return
    C, eff = corpus(cname)
    C = NULLS[null](C, 2024)
    freq = Counter(tuple(sorted(set(r['w']))) for r in C)
    tabs = sorted({r['tab'] for r in C})
    res = []
    for h in range(2):
        rng = random.Random(500 + h); t2 = tabs[:]; rng.shuffle(t2)
        halves = [set(t2[:len(t2) // 2]), set(t2[len(t2) // 2:])]
        for d in range(2):
            H1, H2 = sub(C, halves[d]), sub(C, halves[1 - d])
            D1, R1 = L.run_corpus(H1, mode, 4, 80, seed0=900 + 10 * h + d)
            T1 = L.summarise(D1, R1)
            cl = ruleb(T1, 4, 0.2)          # rule B20 (cycle 2): stable weight, |effect| >= 0.2
            Ds = sorted(s for s, c in cl.items() if c in 'DG')
            # descriptor-only ridge on all of H1, scored on H2 (same target construction on the union)
            Dall = L.Data(H1 + H2, mode)
            ia = np.array([i for i, r in enumerate(Dall.rows) if r['tab'] in halves[d]])
            ib = np.array([i for i, r in enumerate(Dall.rows) if r['tab'] in halves[1 - d]])
            ta, tb = L.center(Dall, ia, ib)
            feats = [('S', s) for s in Ds]
            if mode == 'ctx':
                feats += [('X', None), ('N', None)]
            Xa = L.design(Dall, ia, feats); Xb = L.design(Dall, ib, feats)
            a, b = L.ridge(Xa, ta, 3.0)
            pb = a + Xb @ b if len(feats) else np.full(len(ib), a)
            if mode == 'ctx':   # baseline = context-only model, so only the signs' contribution is scored
                fx = [('X', None), ('N', None)]
                a0, b0 = L.ridge(L.design(Dall, ia, fx), ta, 3.0)
                p0 = a0 + L.design(Dall, ib, fx) @ b0
            else:
                p0 = np.zeros(len(ib))
            cls = []
            for i in ib:
                k = tuple(sorted(Dall.bag[i]))
                cls.append('single' if len(k) < 2 else 'hapax' if freq.get(k, 0) == 1 else 'recur')
            cls = np.array(cls)
            row = dict(h=h, d=d, nD=len(Ds), D=Ds)
            for c in ('hapax', 'recur', 'single'):
                m = cls == c
                if m.sum() < 5:
                    row[c] = None; continue
                e0 = np.mean((tb[m] - p0[m]) ** 2); e1 = np.mean((tb[m] - pb[m]) ** 2)
                row[c] = dict(n=int(m.sum()), R=float(1 - e1 / e0), mse0=float(e0))
            Dset = set(Ds)
            keys = {tuple(sorted(Dall.bag[i])) for i in ib if len(Dall.bag[i]) >= 2}
            row['full'] = float(np.mean([set(k) <= Dset for k in keys])) if keys else 0.0
            row['share'] = float(np.mean([len(set(k) & Dset) / len(k) for k in keys])) if keys else 0.0
            row['nstr'] = len(keys)
            if eff:
                row['plant_hit'] = len(Dset & set(eff))
            res.append(row)
            print(cname, mode, null, h, d, 'nD', len(Ds), {c: (row[c]['R'] if row[c] else None) for c in ('hapax', 'recur', 'single')},
                  'full', round(row['full'], 3), 'share', round(row['share'], 3), flush=True)
    summ = {}
    for c in ('hapax', 'recur', 'single'):
        v = [r[c]['R'] for r in res if r[c]]
        summ[c] = dict(mean=float(np.mean(v)) if v else None, pos=float(np.mean([x > 0 for x in v])) if v else None,
                       n=int(np.mean([r[c]['n'] for r in res if r[c]])) if v else 0)
    summ['nD'] = float(np.mean([r['nD'] for r in res]))
    summ['full'] = float(np.mean([r['full'] for r in res]))
    summ['share'] = float(np.mean([r['share'] for r in res]))
    summ['Dfreq'] = Counter(s for r in res for s in r['D']).most_common(40)
    json.dump(dict(rows=res, summary=summ), open(out, 'w'))
    print(json.dumps(summ))


if __name__ == '__main__':
    main(*sys.argv[1:4])
