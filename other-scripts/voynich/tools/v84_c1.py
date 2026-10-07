"""v84 cycle 1: MEASURE THE PAGE VOCABULARY WITH THE FROZEN STATISTIC (v84_lib.PB, hash printed in the loop file).
Voynich ZL3b and IT2a (VOY_MODE=glyph), leaf swap, page bootstrap, by section and Currier language; plant dose curve
(page-shuffled ZL with r% of interior words replaced by one of 3 page-specific words drawn from the ZL lexicon);
real herbals and recipe books through the v72 surface and their page-shuffled twins; v77 generators fitted to ZL
(SELFCIT SC10 MK2 JUNC STACK) and the v82 kill and shape generators.
"""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v84_lib as K
from multiprocessing import Pool

V82_KILL = {'base': 'stack', 'p_vert': 0.10676249291844887, 'p_cite': 0.0, 'p_mod': 1.151453389807132, 'k': 2, 'M': 4,
            'mood_src': 'fit', 'mood_beta': 1.5968468745053763, 'mood_scope': 'passage', 'mood_rate': 0.052405993703425094,
            'agr': 'harm', 'agr_src': 1, 'agr_lam': 3.6831100288254857, 'H': 6, 'hseed': 109312019, 'bias_copies': False}
V82_SHAPE = {'base': 'stack', 'p_vert': 0.00586142203482235, 'p_cite': 0.11454999695803883, 'p_mod': 0.5853565057447743,
             'k': 2, 'M': 8, 'mood_src': 'fit', 'mood_beta': 0.6938967292942438, 'mood_scope': 'line',
             'mood_rate': 0.3884912551467466, 'agr': 'harm', 'agr_src': 1, 'agr_lam': 1.4063597742933909, 'H': 6,
             'hseed': 461439076, 'bias_copies': False}


def plant(pages, rate, seed):
    rng = random.Random(seed)
    lex = [w for p in pages for l in p['lines'] for w in l['w']]
    S = K.page_shuffle(pages, seed)
    out = []
    for p in S:
        sw = [rng.choice(lex) for _ in range(3)]
        nl = [dict(l, w=[w if (i == 0 or rng.random() >= rate) else rng.choice(sw) for i, w in enumerate(l['w'])]) for l in p['lines']]
        out.append(dict(p, lines=nl))
    return out


def corpora():
    import v77_lib as G, v82_lib as K82
    C = {}
    Z = K.voynich('ZL3b'); I = K.voynich('IT2a')
    C['VOY_ZL'] = Z; C['VOY_IT'] = I
    C['VOY_ZL_SHUF'] = K.page_shuffle(Z, 8401); C['VOY_IT_SHUF'] = K.page_shuffle(I, 8401)
    for r in (0.01, 0.02, 0.05):
        C['PLANT_%02d' % int(r * 100)] = plant(Z, r, 8402)
    C.update(K.controls())
    for g in ('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'):
        C['GEN_' + g] = G.generate('ZL3b', g, 8411)
    C['GEN_V82KILL'] = K82.gen_moodagr(Z, V82_KILL, 8412, {})
    C['GEN_V82SHAPE'] = K82.gen_moodagr(Z, V82_SHAPE, 8413, {})
    return C


C = None


def init():
    global C
    C = K.pload('c1_corpora.pkl')


def work(args):
    name, swap = args
    r = K.PB(C[name], variants=('PB', 'PB_far', 'PB_word'), swap=swap)
    ntok = sum(len(l['w']) for p in C[name] for l in p['lines'])
    return name, swap, r, ntok, len(C[name])


def boot(pages, nb=200, seed=843):
    """page bootstrap of PB on the test half: resample test pages within section (keeping train half) - via
    jackknife-like: recompute PB on random halves of test pages (dropped pages moved out of scoring)."""
    rng = random.Random(seed); vals = []
    test = [p for p in pages if K.leaf(p['id']) == 1]; train = [p for p in pages if K.leaf(p['id']) == 0]
    for b in range(nb):
        keep = [p for p in test if rng.random() < 0.5]
        vals.append(K.PB(train + keep, nnull=4, seed=b)['PB'][0])
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def by_group(pages, key):
    out = {}
    for g in sorted({key(p) for p in pages}):
        sub = [p for p in pages if key(p) == g]
        if sum(len(l['w']) for p in sub for l in p['lines']) < 3000: continue
        r = K.PB(sub)
        out[g] = (r['PB'][0], r['PB'][1], r['ntest'])
    return out


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'build':
        K.psave('c1_corpora.pkl', corpora())
    elif stage == 'run':
        init(); jobs = [(n, s) for n in C for s in (False, True)]
        res = {}
        with Pool(2, initializer=init) as P:
            for name, swap, r, ntok, npg in P.imap_unordered(work, jobs):
                res.setdefault(name, {})[swap] = dict(r=r, ntok=ntok, npg=npg)
                print(name, swap, {k: tuple(round(x, 4) for x in v) if isinstance(v, tuple) else v for k, v in r.items()}, flush=True)
        K.psave('c1_res.pkl', res)
    elif stage == 'extra':
        Z = K.voynich('ZL3b'); I = K.voynich('IT2a')
        ex = {}
        with Pool(2) as P:
            a = P.apply_async(boot, (Z,)); b = P.apply_async(boot, (I,))
            ex['boot_ZL'] = a.get(); ex['boot_IT'] = b.get()
        ex['sec_ZL'] = by_group(Z, lambda p: p['sec']); ex['sec_IT'] = by_group(I, lambda p: p['sec'])
        ex['lang_ZL'] = by_group(Z, lambda p: p['lang']); ex['lang_IT'] = by_group(I, lambda p: p['lang'])
        print(ex)
        K.psave('c1_extra.pkl', ex)
