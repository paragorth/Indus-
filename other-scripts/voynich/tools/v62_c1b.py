"""v62 cycle 1b: RHYME re-scored. Cycle 1's raw kappa maximum was dominated by 2-class, 1-glyph endings
(match rate near 1, noisy kappa). Here each definition is scored by z of (final-word kappa - second-word
kappa) at lags 1-3, uninformative definitions (expected match > 0.5) set to 0; same family-wise nulls;
held-out pages and IT2a; controls held-out; and the natural definitions by Voynich section/language.
"""
import sys, os, json, random, pickle, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L
from v62_c1 import split, A

NDEF = int(os.environ.get('NDEF', 2000)); NNULL = int(os.environ.get('NNULL', 3))
CTRL = ['Regimen(verse,rhymed)', 'Macer(verse,hexam)', 'Dante(verse,terza)', 'Litany(refrain)',
        'Hildegard(prose herbal)', 'Caesar(prose)', 'Dante-reflowed', 'Regimen-reflowed']


def zsearch(pages, defs):
    P = L.prep_positions(pages)
    return np.array([float(np.max(L.rhyme_z(P, d, 3)[0])) for d in defs])


def job(args):
    tag, pages, seed = args
    defs = L.random_defs(L.alphabet(pages), NDEF, random.Random(1000 + seed))
    return tag, defs, zsearch(pages, defs)


def main():
    t0 = time.time(); out = {}
    zl_tr, zl_te = split('V-ZL3b'); it_tr, it_te = split('V-IT2a')
    tasks = [('ZL-train', zl_tr, 1)]
    for i in range(NNULL):
        tasks += [(f'null-shuffle-{i}', L.shuffle_lines_within_page(zl_tr, random.Random(500 + i)), 1),
                  (f'null-reflow-{i}', L.reflow_page(zl_tr, random.Random(600 + i)), 1),
                  (f'null-markov-{i}', L.markov_line_generator(zl_tr, random.Random(700 + i)), 1)]
    for name in CTRL:
        tasks.append((name, split(name)[0], 2))
    with Pool(2) as pool:
        res = {t: (d, s) for t, d, s in pool.imap_unordered(job, tasks)}
    out['search'] = {t: dict(max=round(float(s.max()), 2), p99=round(float(np.percentile(s, 99)), 2),
                             best=L.def_str(d[int(s.argmax())])) for t, (d, s) in res.items()}
    for t in sorted(out['search']): print(t, out['search'][t], flush=True)
    nullmax = [v['max'] for t, v in out['search'].items() if t.startswith('null')]
    out['ZL_vs_nullmax'] = dict(real=out['search']['ZL-train']['max'], null_max=nullmax,
                                frac_null_ge=float(np.mean(np.array(nullmax) >= out['search']['ZL-train']['max'])))
    defs, sc = res['ZL-train']
    top = np.argsort(-sc)[:15]
    P = {k: L.prep_positions(v) for k, v in (('zl_te', zl_te), ('it_te', it_te), ('it_tr', it_tr))}
    out['heldout'] = [dict(df=L.def_str(defs[i]), train=round(float(sc[i]), 2),
                           **{k: [round(x, 2) for x in L.rhyme_z(P[k], defs[i], 3)[0]] for k in P}) for i in top]
    for r in out['heldout']: print(r, flush=True)
    out['controls_heldout'] = {}
    for name in CTRL:
        d, s = res[name]; i = int(s.argmax()); te = split(name)[1]
        out['controls_heldout'][name] = dict(df=L.def_str(d[i]), train=round(float(s[i]), 2),
                                             test=[round(x, 2) for x in L.rhyme_z(L.prep_positions(te), d[i], 3)[0]])
        print(name, out['controls_heldout'][name], flush=True)
    # natural definitions: all corpora and Voynich subsets
    nat = [(None, k, 0) for k in (1, 2, 3)] + [(None, 2, 1)]
    out['natural'] = {}
    groups = {}
    for tr in ('ZL3b', 'IT2a'):
        pg, meta = A['V-' + tr]['pages'], A['V-' + tr]['meta']
        for key in ('illus', 'lang'):
            for v in sorted(set(m[key] for m in meta)):
                sub = [p for p, m in zip(pg, meta) if m[key] == v]
                if sum(len(p) for p in sub) >= 150: groups[f'{tr}:{key}={v}'] = sub
        groups[f'{tr}:all'] = pg
    for name in A:
        if not name.startswith('V-'): groups[name] = A[name]['pages']
    for g, pages in groups.items():
        Pg = L.prep_positions(pages)
        out['natural'][g] = {L.def_str(d): [round(x, 2) for x in L.rhyme_z(Pg, d, 3)[0]] for d in nat}
        print('nat', g, out['natural'][g], flush=True)
    out['secs'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(L.CK, 'cycle1b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
