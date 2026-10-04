#!/usr/bin/env python3
"""la34 arm R ("rhythm"): writer features from the GEOMETRY of SigLA's per-occurrence boxes only
(no shapes): sign-specific size and aspect residuals, spacing inside and between words, line pitch,
baseline slope and jitter, logogram/fraction size ratios, line length.  All ratios are scale-free.
Output: la34_ckpt/rhythm.json {unit: [features]}"""
import numpy as np, json, os, collections
from la34_common import load, CK

NAMES = ['asp_mean', 'asp_sd', 'size_sd', 'gap_in', 'gap_word', 'pitch', 'slope', 'jitter', 'logo_ratio',
         'frac_ratio', 'signs_per_line', 'occupancy']


def lines_of(occ):
    """group occurrences (reading order) into lines: a new line when the centre y jumps by > 0.6 median h"""
    hs = np.median([o['rect'][3] for o in occ])
    L = [[occ[0]]]
    for a, b in zip(occ, occ[1:]):
        ya = a['rect'][1] + a['rect'][3] / 2; yb = b['rect'][1] + b['rect'][3] / 2
        if abs(yb - ya) > 0.6 * hs or b['rect'][0] < a['rect'][0] - 0.5 * hs: L.append([b])
        else: L[-1].append(b)
    return L, hs


def main():
    occ, meta, cid = load()
    syl = [o for o in occ if o['role'] == 'syllabogram']
    by_unit_doc = collections.defaultdict(list)
    for o in occ: by_unit_doc[(o['unit'], o['doc'])].append(o)
    # sign-code means of relative log size and log aspect
    rel = {}
    for (u, d), os_ in by_unit_doc.items():
        s = [o for o in os_ if o['role'] == 'syllabogram']
        if not s: continue
        mh = np.median([o['rect'][3] for o in s])
        for o in os_:
            rel[id(o)] = (np.log(o['rect'][3] / mh), np.log(o['rect'][2] / o['rect'][3]))
    cm = collections.defaultdict(list)
    for o in syl:
        if id(o) in rel: cm[o['code']].append(rel[id(o)])
    cmean = {c: np.mean(v, 0) for c, v in cm.items() if len(v) >= 5}
    feats = collections.defaultdict(lambda: collections.defaultdict(list))
    for (u, d), os_ in by_unit_doc.items():
        os_ = sorted(os_, key=lambda o: o['n'])
        s = [o for o in os_ if o['role'] == 'syllabogram']
        if len(s) < 2: continue
        F = feats[u]
        mh = np.median([o['rect'][3] for o in s])
        for o in s:
            if o['code'] in cmean:
                r = np.array(rel[id(o)]) - cmean[o['code']]
                F['size'].append(r[0]); F['asp'].append(r[1])
        L, hs = lines_of(os_)
        for line in L:
            F['spl'].append(len(line))
            for a, b in zip(line, line[1:]):
                g = (b['rect'][0] - (a['rect'][0] + a['rect'][2])) / mh
                if a['role'] == b['role'] == 'syllabogram':
                    (F['gin'] if a['par'] == b['par'] else F['gword']).append(g)
            if len(line) >= 3:
                x = np.array([o['rect'][0] + o['rect'][2] / 2 for o in line]); y = np.array([o['rect'][1] + o['rect'][3] for o in line])
                if np.ptp(x) > 0:
                    p = np.polyfit(x, y, 1); F['slope'].append(np.degrees(np.arctan(p[0])))
                    F['jit'].append(np.std(y - np.polyval(p, x)) / mh)
        ys = [np.mean([o['rect'][1] + o['rect'][3] / 2 for o in line]) for line in L]
        F['pitch'] += list(np.diff(ys) / mh) if len(ys) > 1 else []
        for o in os_:
            if o['role'] == 'logogram': F['logo'].append(np.log(o['rect'][3] / mh))
            if o['role'] == 'fraction': F['frac'].append(np.log(o['rect'][3] / mh))
        if o['vb']: F['occ'].append(np.log(mh / o['vb'][1]))
    out = {}
    for u, F in feats.items():
        g = lambda k, f=np.mean: float(f(F[k])) if len(F[k]) else float('nan')
        out[u] = [g('asp'), g('asp', np.std) if len(F['asp']) > 1 else float('nan'), g('size', np.std) if len(F['size']) > 1 else float('nan'),
                  g('gin', np.median), g('gword', np.median), g('pitch', np.median), g('slope', np.median), g('jit', np.median),
                  g('logo'), g('frac'), g('spl'), g('occ')]
    json.dump({'names': NAMES, 'units': out}, open(os.path.join(CK, 'rhythm.json'), 'w'))
    print('units', len(out))


if __name__ == '__main__':
    main()
