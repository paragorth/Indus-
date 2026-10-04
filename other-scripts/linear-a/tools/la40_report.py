#!/usr/bin/env python3
"""LA-40 report for a cycle checkpoint: forced roles, null correction, Linear B control, plants."""
import collections, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la40_common as C

THR = 0.9


def forced(r, minocc=2):
    return {w: m for w, m, f, a, n in zip(r['types'], r['mode'], r['freq'], r['agree'], r['nocc'])
            if n >= minocc and f >= THR and a >= THR}


def lb_score(r, lab, minocc=2, only_forced=True):
    F = forced(r, minocc) if only_forced else {w: m for w, m, n in zip(r['types'], r['mode'], r['nocc']) if n >= minocc}
    pairs = [(lab[w], m) for w, m in F.items() if w in lab]
    if not pairs:
        return dict(n=0, acc=float('nan'), best=float('nan'), maj=float('nan'))
    acc = np.mean([a == b for a, b in pairs])
    labs = sorted(set(a for a, _ in pairs))
    cnt = collections.Counter(pairs)
    # best many-to-one mapping from inferred role to label (generous; the null gets the same search)
    best = sum(max(cnt[(l, m)] for l in labs) for m in C.ROLES) / len(pairs)
    maj = collections.Counter(a for a, _ in pairs).most_common(1)[0][1] / len(pairs)
    rec = {l: (sum(cnt[(l, l)] for _ in [0]) / max(1, sum(v for (a, b), v in cnt.items() if a == l))) for l in labs}
    return dict(n=len(pairs), acc=float(acc), best=float(best), maj=float(maj), recall=rec, conf=sorted(cnt.items()))


def main(path):
    res = json.load(open(path))
    lab = C.lb_labels()
    by = collections.defaultdict(list)
    for r in res: by[r['kind']].append(r)
    out = []
    if 'LA' in by:
        la = by['LA'][0]
        F = forced(la)
        cnt = collections.Counter(F.values())
        out.append('LA: %d types (%d with >=2 occ); forced (>=%.1f of samples and chains) %d: %s' % (
            len(la['types']), sum(n >= 2 for n in la['nocc']), THR, len(F), dict(cnt)))
        nulls = by.get('NULL', [])
        if nulls:
            nc = [collections.Counter(forced(n).values()) for n in nulls]
            out.append('NULL (words shuffled within site, same search) forced per role, mean [max]: ' + ', '.join(
                '%s %.1f [%d]' % (r, np.mean([c[r] for c in nc]), max(c[r] for c in nc)) for r in C.ROLES))
            out.append('  real/null ratio per role: ' + ', '.join('%s %d/%.1f' % (r, cnt[r], np.mean([c[r] for c in nc])) for r in C.ROLES))
            # per-word: how often is the word forced to the same role after shuffling?
            nf = [forced(n) for n in nulls]
            rows = []
            for w, m in sorted(F.items(), key=lambda x: -la['nocc'][la['types'].index(x[0])]):
                same = sum(1 for f in nf if f.get(w) == m) / len(nf)
                i = la['types'].index(w)
                rows.append((w, m, la['nocc'][i], la['ndocs'][i], la['freq'][i], same))
            sig = [x for x in rows if x[5] <= 1 / len(nf) and x[1] != 'L']
            out.append('  forced roles that the shuffle null reproduces in <= 1/%d runs (non-default roles): %d' % (len(nf), len(sig)))
            for x in rows[:60]:
                out.append('   %-16s %s occ %3d docs %3d freq %.3f null-same %.2f' % x)
            json.dump(rows, open(path.replace('.json', '_forced.json'), 'w'))
    for kind in ('LB', 'LBNULL', 'LBFULL'):
        if kind in by:
            sc = [lb_score(r, lab) for r in by[kind]]
            sa = [lb_score(r, lab, only_forced=False) for r in by[kind]]
            out.append('%s (%d draws): labelled forced types n %s; strict acc %s; best-map acc %s; majority %s; all-types strict %s' % (
                kind, len(sc), [s['n'] for s in sc], ['%.2f' % s['acc'] for s in sc], ['%.2f' % s['best'] for s in sc],
                ['%.2f' % s['maj'] for s in sc], ['%.2f' % s['acc'] for s in sa]))
            if kind in ('LB', 'LBFULL'):
                out.append('   conf (truth, inferred) draw0: %s' % sc[0].get('conf'))
                for w in ['TO-SO', 'TO-SA', 'A-PU-DO-SI', 'O-PE-RO', 'PA-RO', 'E-KE', 'KO-NO-SO', 'PU-RO', 'PA-I-TO']:
                    xs = []
                    for r in by[kind]:
                        if w in r['types']:
                            i = r['types'].index(w); xs.append('%s%.2f' % (r['mode'][i], r['freq'][i]))
                    out.append('   %s: %s' % (w, ' '.join(xs)))
    if 'PLANT' in by:
        for r in by['PLANT']:
            s = []
            for w, tr in r['truth'].items():
                i = r['types'].index(w); s.append('%s truth %s got %s (%.2f)' % (w, tr, r['mode'][i], r['freq'][i]))
            if 'PLANT-E' in r['types']:
                i = r['types'].index('PLANT-E')
                s.append('PLANT-E got %s (%.2f); neighbours %s' % (r['mode'][i], r['freq'][i], r['E_expected']))
            out.append('PLANT %d: ' % r['k'] + '; '.join(s))
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.CK, 'c1.json'))
