"""v21 cycle 3: COPY-AND-VARY FORGERS. Can known mechanisms close what cycles 1-2 caught?

Cycle-1 survivors against F7 were local self-similarity (repeats in a line, near-repeats a line or more apart,
first/last glyph echoed two words later, positive word-length autocorrelation) and drift inside the page.
 F9  = F7 + rich junction key (last unit and length of the previous word, first unit of the word two back),
       line filling tuned so token counts match (end_room 6)
 F10 = F9 + Timm-Schinner self-citation: with prob cite copy a word from the last 15 tokens of the
       paragraph, edited with prob cite_edit by an edit drawn from real near-repeat pairs; paragraph-level
       topic tables (lam). (lam, cite, cite_edit, redup) grid-calibrated on the repeat / near-repeat profile
       ONLY (rep_*, near_*, ttr, vert_same); the other features are left free as the test.
Same battery as cycle 1 (ridge, boosting, 300 random 6-feature subsets, 5-fold CV by page, 5 seeds), on
Voynich ZL3b, Voynich IT2a (replication), Latin and Italian herbals; null, negative and planted controls.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_cycle1 import LAB, disc_battery, plant_longrange
from v21_lib import *
from multiprocessing import Pool

NSEED = int(os.environ.get('NSEED', 5))
CAL2 = ['rep_adj', 'rep_sameline', 'rep_prevline', 'rep_line2_4', 'rep_line5p', 'near_same', 'near_prev', 'near_far',
        'ttr', 'vert_same']


def corpora3():
    return {'V': voynich_pages('ZL3b'), 'VI': voynich_pages('IT2a'), 'LA': latin_herbal(), 'IT': italian_herbal()}


def mk(name, C, par):
    base = dict(scope='sec', pos=True, chain=True, width=True, rich=True, end_room=6)
    if name == 'F7': return Forger(C, scope='sec', pos=True, lam=par['lam'], chain=True, redup=par['redup'], cite=par['cite'], width=True, name=name)
    if name == 'F9': return Forger(C, lam=par['lam'], redup=par['redup'], cite=par['cite'], name=name, **base)
    if name == 'F10': return Forger(C, lam=par['lam10'], redup=par['redup10'], cite=par['cite10'], cite_window=15,
                                    cite_edit=par['edit10'], para_lam=True, name=name, **base)
    raise KeyError(name)


def calibrate10(C, FZ, seed=0):
    Xr, keys = FZ.matrix(C)
    idx = [keys.index(k) for k in CAL2]
    tr = Xr[:, idx].mean(0); sd = Xr[:, idx].std(0) / math.sqrt(len(C)) + 1e-9
    best = None
    for lam in (0.2, 0.35, 0.5):
        for cite in (0.03, 0.06, 0.1, 0.15):
            for edit in (0.4, 0.8):
                for redup in (0.0, 0.005):
                    par = dict(lam10=lam, cite10=cite, edit10=edit, redup10=redup)
                    Xf, _ = FZ.matrix(mk('F10', C, par).forge(C, random.Random(seed)), keys)
                    err = float((((Xf[:, idx].mean(0) - tr) / sd) ** 2).sum())
                    if best is None or err < best[0]: best = (err, par)
    return best


def get_pars(cn, C):
    p = load(f'c3_cal_{cn}.json')
    if p: return p
    FZ = Featurizer(C, LAB)
    p1 = load(f'c1_cal_{cn}.json') or load('c1_cal_V.json')
    err, p = calibrate10(C, FZ)
    p.update({k: p1[k] for k in ('lam', 'cite', 'redup')}); p['err10'] = err
    save(f'c3_cal_{cn}.json', p)
    return p


def job(args):
    tag, cname, fname, seed = args
    ck = f'c3_{tag}_{cname}_{fname}_{seed}.json'
    got = load(ck)
    if got: return got
    C = corpora3()[cname]; par = load(f'c3_cal_{cname}.json')
    FZ = Featurizer(C, LAB); rng = random.Random(seed)
    if tag == 'main':
        real, forged = C, mk(fname, C, par).forge(C, rng)
    elif tag == 'null':
        fg = mk(fname, C, par); real, forged = fg.forge(C, random.Random(10_000 + seed)), fg.forge(C, rng)
    elif tag == 'neg':
        S = mk(fname, C, par).forge(C, random.Random(20_000 + seed)); real, forged = S, mk(fname, S, par).forge(S, rng)
    elif tag == 'plant':
        S = mk(fname, C, par).forge(C, random.Random(30_000 + seed)); P = plant_longrange(S, random.Random(40_000 + seed))
        real, forged = P, mk(fname, P, par).forge(P, rng)
    Xr, keys = FZ.matrix(real); Xf, _ = FZ.matrix(forged, keys)
    res = disc_battery(Xr, Xf, seed, 300 if tag == 'main' else 100, keys)
    res.update(tag=tag, corpus=cname, forger=fname, seed=seed, keys=keys)
    save(ck, res)
    return res


def cal_job(cn):
    return cn, get_pars(cn, corpora3()[cn])


def main():
    t0 = time.time()
    with Pool(2) as pool:
        for cn, p in pool.imap_unordered(cal_job, ['V', 'LA', 'IT', 'VI']):
            print(cn, p, f'[{time.time() - t0:.0f}s]', flush=True)
        jobs = []
        for s in range(NSEED):
            jobs += [('main', 'V', 'F9', s), ('main', 'V', 'F10', s), ('main', 'VI', 'F7', s), ('main', 'VI', 'F10', s),
                     ('main', 'LA', 'F10', s), ('main', 'IT', 'F10', s),
                     ('null', 'V', 'F10', s), ('neg', 'V', 'F10', s), ('neg', 'LA', 'F10', s), ('neg', 'IT', 'F10', s),
                     ('plant', 'V', 'F10', s)]
        for r in pool.imap_unordered(job, jobs):
            print(f"{r['tag']:5s} {r['corpus']:2s} {r['forger']} s{r['seed']} ridge {r['lr']:.3f} gbm {r['gbm']:.3f} "
                  f"subMax {max(a for a, _ in r['sub']):.3f} [{time.time() - t0:.0f}s]", flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
