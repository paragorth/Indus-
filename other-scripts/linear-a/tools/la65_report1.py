#!/usr/bin/env python3
"""la65 cycle 1 report: family-wise thresholds from entry-shuffled nulls, significant groups, dossiers,
Linear B series purity, planted recovery. Writes data/la65_ckpt/c1_dossiers.json and prints rows."""
import os, json, glob, random
from collections import Counter
import numpy as np
import la65_common as K
from la65_c1 import corpus


def load(name, s):
    fn = os.path.join(K.CK, 'c1_%s_%d.json' % (name, s))
    return json.load(open(fn)) if os.path.exists(fn) else None


def maxes(groups):
    m = [-9.0] * 4
    for G, h, tm in groups:
        c = K.size_class(len(G))
        m[c] = max(m[c], h)
    return m


def thresholds(nulls):
    M = [maxes(g['groups']) for g in nulls]
    return [max(x[c] for x in M) for c in range(4)], M


def fdr_thresholds(real, nulls, q=0.2):
    """per size class: smallest t with mean null count above t / real count above t <= q"""
    thr = []
    for c in range(4):
        R = [h for G, h, t in real['groups'] if K.size_class(len(G)) == c]
        N = [[h for G, h, t in n['groups'] if K.size_class(len(G)) == c] for n in nulls]
        best = 99.0
        for t in np.arange(1.5, 25, 0.25):
            nr = sum(h > t for h in R)
            nn = np.mean([sum(h > t for h in x) for x in N])
            if nr >= 1 and nn / nr <= q:
                best = float(t)
                break
        thr.append(best)
    return thr


def count_sig(groups, thr):
    return [x for x in groups if x[1] > thr[K.size_class(len(x[0]))]]


def dossiers(name, real, thr):
    T, meta = corpus(name)
    ix = K.Index(T, meta)
    pos = {t['id']: i for i, t in enumerate(T)}
    sig = count_sig(real['groups'], thr)
    D = K.merge_dossiers(ix, [([pos[i] for i in G], h, tuple(tm)) for G, h, tm in sig])
    out = []
    for d in D:
        ids = sorted(T[i]['id'] for i in d['m'])
        out.append({'ids': ids, 'score': round(d['score'], 2), 'tmpl': [list(t) for t in d['tmpl'][:5]],
                    'n_tmpl': len(d['tmpl'])})
    out.sort(key=lambda d: -d['score'])
    return out, sig, T


def null_fp(nulls):
    fp = []
    for j, g in enumerate(nulls):
        others = [x for k, x in enumerate(nulls) if k != j]
        thr, _ = thresholds(others)
        fp.append(len(count_sig(g['groups'], thr)))
    return fp


def main():
    res = {}
    rows = []
    # ---------------- Linear A, with and without metadata
    for name in ['LA', 'LAN']:
        real = load(name, 0)
        nulls = [load(name, s) for s in range(1, 6)]
        nulls = [n for n in nulls if n]
        thr, M = thresholds(nulls)
        D, sig, T = dossiers(name, real, thr)
        fp = null_fp(nulls)
        res[name] = {'thr': thr, 'sig': len(sig), 'fp': fp, 'dossiers': D, 'n_tabs': len(T),
                     'in_doss': len({i for d in D for i in d['ids']})}
        rows.append('%s: %d tablets; %d distinct groups; thresholds (2/3/4-5/6+) %s; significant %d; '
                    'null runs vs the other nulls %s; dossiers %d covering %d tablets' % (
                        name, len(T), len(real['groups']), ' / '.join('%.1f' % x for x in thr), len(sig), fp,
                        len(D), res[name]['in_doss']))
        if name == 'LAN':
            thrF = fdr_thresholds(real, nulls)
            DF, sigF, _ = dossiers(name, real, thrF)
            res['LANF'] = {'thr': thrF, 'sig': len(sigF), 'dossiers': DF,
                           'in_doss': len({i for d in DF for i in d['ids']})}
            rows.append('LANF (FDR 0.2): thresholds %s; groups %d; dossiers %d covering %d tablets' % (
                thrF, len(sigF), len(DF), res['LANF']['in_doss']))
            for d in DF:
                rows.append('   F %s score %.1f via %s (%d templates)' % (' '.join(d['ids']), d['score'], d['tmpl'][0], d['n_tmpl']))
        for d in D:
            rows.append('   %s score %.1f  via %s (%d templates)' % (' '.join(d['ids']), d['score'],
                                                                     d['tmpl'][0], d['n_tmpl']))
    # ---------------- Linear B draws
    lbsum = []
    for dd in range(3):
        name = 'LB%d' % dd
        real = load(name, 0)
        nulls = [load(name, s) for s in range(1, 4)]
        nulls = [n for n in nulls if n]
        if not real or not nulls:
            continue
        thr, M = thresholds(nulls)
        D, sig, T = dossiers(name, real, thr)
        fp = null_fp(nulls)
        ser = {t['id']: t['series'] for t in T}
        pur = [Counter(ser[i] for i in d['ids']).most_common(1)[0][1] / len(d['ids']) for d in D]
        rng = random.Random(5)
        ids = [t['id'] for t in T]
        rp = []
        for d in D:
            for _ in range(200):
                g = rng.sample(ids, len(d['ids']))
                rp.append(Counter(ser[i] for i in g).most_common(1)[0][1] / len(g))
        sers = Counter(ser[i] for d in D for i in d['ids'])
        res[name] = {'thr': thr, 'sig': len(sig), 'fp': fp, 'dossiers': D, 'purity': pur,
                     'rand_purity': float(np.mean(rp)) if rp else None, 'n_tabs': len(T),
                     'in_doss': len({i for d in D for i in d['ids']}), 'series': sers.most_common(12)}
        rows.append('%s: thresholds %s; significant %d; nulls %s; dossiers %d covering %d tablets; '
                    'series purity %.2f (random groups %.2f); pure dossiers %d/%d; series %s' % (
                        name, ' / '.join('%.1f' % x for x in thr), len(sig), fp, len(D), res[name]['in_doss'],
                        np.mean(pur) if pur else 0, res[name]['rand_purity'] or 0,
                        sum(p == 1 for p in pur), len(pur), sers.most_common(10)))
        lbsum.append(res[name])
        thrF = fdr_thresholds(real, nulls)
        DF, sigF, _ = dossiers(name, real, thrF)
        purF = [Counter(ser[i] for i in d['ids']).most_common(1)[0][1] / len(d['ids']) for d in DF]
        rpF = []
        for d in DF:
            for _ in range(200):
                g = rng.sample(ids, len(d['ids']))
                rpF.append(Counter(ser[i] for i in g).most_common(1)[0][1] / len(g))
        res[name + 'F'] = {'thr': thrF, 'sig': len(sigF), 'dossiers': DF, 'purity': purF,
                           'rand_purity': float(np.mean(rpF)) if rpF else None}
        rows.append('%sF (FDR 0.2): thresholds %s; groups %d; dossiers %d covering %d tablets; purity %.2f (random %.2f)' % (
            name, thrF, len(sigF), len(DF), len({i for d in DF for i in d['ids']}), np.mean(purF) if purF else 0,
            np.mean(rpF) if rpF else 0))
    # ---------------- planted
    thr = res['LA']['thr']
    pl = []
    for p in range(5):
        name = 'PL%d' % p
        real = load(name, 0)
        if not real:
            continue
        D, sig, T = dossiers(name, real, thr)
        planted = [t['id'] for t in T if t.get('planted')]
        src = planted[0].split('_', 1)[1]
        best = max(D, key=lambda d: len(set(d['ids']) & set(planted))) if D else {'ids': []}
        hit = len(set(best['ids']) & set(planted))
        extra = [i for i in best['ids'] if i not in planted]
        pl.append({'name': name, 'src': src, 'hit': hit, 'extra': extra, 'dossier': best['ids']})
        rows.append('%s: planted from %s; recovered %d/6 in one dossier; extra members %s' % (name, src, hit, extra))
    res['PL'] = pl
    json.dump(res, open(os.path.join(K.CK, 'c1_dossiers.json'), 'w'), indent=0)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
