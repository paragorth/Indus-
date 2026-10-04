"""v38 cycle 2: real-herbal positive control by the same pipeline.
Gerard, The Herball (1636; archive.org mobot31753000817756): woodcuts segmented after blanking
OCR word boxes; page OCR as text. Partial Mantel controlling page adjacency, log page distance,
text length and woodcut area; free page permutation. Also: a 'Voynich-ised' Gerard (each English
word type replaced by an arbitrary code word, so word identity carries meaning but spelling does
not) and a page-shuffled Gerard.
Writes data/v38_ckpt/c2.json and loops/v38_cycle2.txt.
"""
import os, sys, json, re
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *

NPERM = int(os.environ.get('NPERM', 2000))
CACHE = os.environ.get('V38_CACHE')


def gerard_setup(min_area=0.06):
    vis = json.load(open(os.path.join(DER, 'v38_vis_gerard.json')))
    keys = sorted([k for k in vis if vis[k]['area'] >= min_area], key=int)
    ocr = gerard_ocr(os.path.join(CACHE, 'gerard_djvu.xml'), set(int(k) - 1 for k in keys))
    words = []
    for k in keys:
        ws = [re.sub(r'[^a-z]', '', w[0].lower()) for w in ocr[int(k) - 1][2]]
        words.append([w for w in ws if len(w) >= 3])
    good = [i for i, w in enumerate(words) if len(w) >= 30]
    keys = [keys[i] for i in good]; words = [words[i] for i in good]
    n = len(keys)
    order = np.array([int(k) for k in keys], float)
    D = np.abs(order[:, None] - order[None, :])
    ln = np.log([len(w) for w in words])
    ar = np.log([vis[k]['area'] for k in keys])
    conf = dict(adj=(D <= 8).astype(float), logdist=np.log1p(D),
                lensum=ln[:, None] + ln[None, :], lendiff=np.abs(ln[:, None] - ln[None, :]),
                areasum=ar[:, None] + ar[None, :], areadiff=np.abs(ar[:, None] - ar[None, :]))
    return keys, words, vis, conf


if __name__ == '__main__':
    rng = np.random.default_rng(3838)
    keys, words, vis, conf = gerard_setup()
    n = len(keys)
    print('gerard pages', n)
    V = vis_sims(vis, keys)
    part = Partial(list(conf.values()), n)
    rows, out = [], {'n': n, 'keys': keys}
    T = text_sims(text_profiles(words))
    r0 = mantel_table(V, T, Partial([], n), NPERM // 2, rng)
    r1 = mantel_table(V, T, part, NPERM, rng)
    # page distance alone: how much is adjacency?
    far = Partial(list(conf.values()), n)
    out['raw'], out['partial'] = r0, r1
    fz = lambda r: ', '.join('%s %+.1f' % kv for kv in r['fam_z'].items())
    rows.append(('V-38.5', 'POSITIVE control, Gerard Herball 1636: %d pages with woodcuts (sampled every 8th scan, 45-1565), same 5 visual families (colour empty: woodcuts) x 5 text metrics on page OCR; raw Mantel' % n,
                 'omni %+.4f z %+.1f p %.4f; family z %s' % (r0['omni'], r0['omni_z'], r0['omni_p'], fz(r0)), 'raw link' if r0['omni_p'] < .01 else 'no raw link'))
    rows.append(('V-38.6', 'Gerard partial Mantel (adjacency, log page distance, text length, woodcut area), %d perms' % NPERM,
                 'omni %+.4f z %+.1f p %.4f; max cell %+.3f p_max %.3f; family z %s; text z %s' % (
                     r1['omni'], r1['omni_z'], r1['omni_p'], r1['max'], r1['max_p'], fz(r1),
                     ', '.join('%s %+.1f' % kv for kv in r1['met_z'].items())),
                 'pipeline sees text-tracks-plant in a real herbal' if r1['omni_p'] < .01 else 'CONTROL FAILS: pipeline blind'))
    # far pairs only: drop pairs closer than 40 scans by giving them a separate intercept dummy
    near = (np.abs(np.array([int(k) for k in keys])[:, None] - np.array([int(k) for k in keys])[None, :]) < 40).astype(float)
    r2 = mantel_table(V, T, Partial(list(conf.values()) + [near], n), NPERM // 2, rng)
    out['partial_far'] = r2
    rows.append(('V-38.7', 'Gerard, extra near-pair dummy (<40 scans apart) so the effect must come from far-apart pages',
                 'omni %+.4f z %+.1f p %.4f; family z %s' % (r2['omni'], r2['omni_z'], r2['omni_p'], fz(r2)), ''))
    # Voynich-ised Gerard: word types -> random code strings (meaning kept, spelling scrambled)
    types = sorted(set(w for ws in words for w in ws))
    code = {t: 'w%06d' % i for i, t in enumerate(rng.permutation(len(types)))}
    code = {t: c for t, c in zip(types, code.values())}
    wc = [[code[w] for w in ws] for ws in words]
    Tc = text_sims(text_profiles(wc))
    r3 = mantel_table(V, Tc, part, NPERM // 2, rng)
    out['coded'] = r3
    rows.append(('V-38.8', 'Gerard with every word type replaced by an arbitrary code word (word identity kept, spelling destroyed: the situation of an unread script)',
                 'omni %+.4f z %+.1f p %.4f; text z %s' % (r3['omni'], r3['omni_z'], r3['omni_p'], ', '.join('%s %+.1f' % kv for kv in r3['met_z'].items())),
                 'word-level metrics survive coding' if r3['omni_p'] < .01 else 'lost'))
    sh = []
    for rep in range(10):
        p = rng.permutation(n)
        rr = mantel_table(V, text_sims(text_profiles([words[i] for i in p])), part, 300, rng)
        sh.append(rr['omni_z'])
    out['shuffled'] = sh
    rows.append(('V-38.9', 'Gerard page-shuffled text (10 shuffles)', 'omni z ' + ' '.join('%+.1f' % z for z in sh), 'killed' if max(sh) < 2.5 else 'null leaks'))
    # subsample to Voynich size and power: how often p<0.01 with n=Voynich pages
    nv = int(os.environ.get('NV', 110))
    hits = []
    for rep in range(10):
        idx = np.sort(rng.choice(n, min(nv, n), replace=False))
        Vs = {f: M[np.ix_(idx, idx)] for f, M in V.items()}
        Ts = {m: M[np.ix_(idx, idx)] for m, M in T.items()}
        ps = Partial([C[np.ix_(idx, idx)] for C in conf.values()], len(idx))
        rr = mantel_table(Vs, Ts, ps, 300, rng)
        hits.append((rr['omni_z'], rr['omni_p']))
    out['sub'] = hits
    rows.append(('V-38.10', 'Gerard subsampled to %d pages (Voynich size), 10 draws' % nv,
                 'omni z ' + ' '.join('%.1f' % z for z, _ in hits) + '; p<0.01 in %d/10' % sum(p < .01 for _, p in hits), 'power at Voynich n'))
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v38_cycle2.txt'), 'w') as fh:
        fh.write('# v38 cycle 2 - real-herbal positive control (Gerard 1636) by the same pipeline\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v38_cycle2.txt')).read())
