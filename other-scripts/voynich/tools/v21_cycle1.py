"""v21 cycle 1: the forger ladder against page-level feature discriminators.

Corpora: Voynich ZL3b (paragraph text, pages >= 40 words), Latin herbal/lapidary (Isidore XVI-XVII),
Italian herbal (Brumati 1844). Forgers (all keep the page/paragraph/line/words-per-line skeleton):
 F0 unigram   : section x position unigram (no junction)
 F1 junc      : whole-book junction (next word drawn from real successors of words with the same last unit)
 F2 junc+pos  : F1 with position-aware tables (2nd word, 3rd word, middle, last)
 F3 junc+sec  : F2 per section
 F4 slot      : in-word unit trigram per section x position class (slot grammar), no word links
 F5 junc+page : F3 + page topic (lam: draw from the page's own successor table)
 F6 kitchen   : F5 + line-initial chain + reduplication + self-citation, (lam, cite, redup) calibrated
 F7 kitchen+width : F6, but lines are filled to the real line's width in units (not its word count)
Discriminators: L2 logistic on all features, gradient boosting on all features, and 300 logistic models
on random 6-feature subsets per forger seed; 5-fold CV by page (paired real/forged).
Controls: null (forgery vs forgery), negative (F7 output re-forged by F7 refitted on it), positive
(Latin and Italian herbals through the same ladder), planted long-range dependency in F7 output.
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_lib import *
from multiprocessing import Pool

NSEED = int(os.environ.get('NSEED', 5))
NSUB = int(os.environ.get('NSUB', 300))
OUT = 'v21_cycle1.txt'

LAB = [U(d['label']) for d in json.load(open(os.path.join(vlib.DATA, 'derived', 'v2_labels.json'))) if d.get('label')]


def corpora():
    return {'V': voynich_pages('ZL3b'), 'LA': latin_herbal(), 'IT': italian_herbal()}


def make_forger(name, C, par=None):
    par = par or {}
    if name == 'F0': return Forger(C, scope='sec', pos=True, unigram=True, name=name)
    if name == 'F1': return Forger(C, scope='global', pos=False, name=name)
    if name == 'F2': return Forger(C, scope='global', pos=True, name=name)
    if name == 'F3': return Forger(C, scope='sec', pos=True, name=name)
    if name == 'F4': return SlotForger(C, name=name)
    if name == 'F5': return Forger(C, scope='sec', pos=True, lam=par.get('lam5', 0.3), name=name)
    if name == 'F6': return Forger(C, scope='sec', pos=True, lam=par['lam'], chain=True, redup=par['redup'],
                                   cite=par['cite'], name=name)
    if name == 'F7': return Forger(C, scope='sec', pos=True, lam=par['lam'], chain=True, redup=par['redup'],
                                   cite=par['cite'], width=True, name=name)
    raise KeyError(name)


CAL_KEYS = ['rep_adj', 'rep_sameline', 'rep_prevline', 'rep_line2_4', 'rep_line5p', 'ttr', 'vert_same']


def calibrate(C, FZ, rng_seed=0):
    """Grid search (lam, cite, redup) so that page means of the repeat features match the real corpus."""
    Xr, keys = FZ.matrix(C)
    idx = [keys.index(k) for k in CAL_KEYS]
    tr = Xr[:, idx].mean(0); sd = Xr[:, idx].std(0) / math.sqrt(len(C)) + 1e-9
    best = None
    for lam in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7):
        for cite in (0.0, 0.01, 0.03, 0.06):
            for redup in (0.0, 0.004, 0.008, 0.015):
                fg = Forger(C, scope='sec', pos=True, lam=lam, chain=True, redup=redup, cite=cite)
                Xf, _ = FZ.matrix(fg.forge(C, random.Random(rng_seed)), keys)
                err = float((((Xf[:, idx].mean(0) - tr) / sd) ** 2).sum())
                if best is None or err < best[0]: best = (err, dict(lam=lam, cite=cite, redup=redup))
    return best


def plant_longrange(C, rng, rate=1.0):
    """Planted long-range dependency: on each page, a word from the first line of each paragraph is
    re-used once in each later line of that paragraph with prob 0.35 (replacing a random medial word)."""
    out = []
    for p in C:
        paras = []
        for pa in p['paras']:
            pa = [list(l) for l in pa]
            if rng.random() < rate and len(pa) >= 2:
                key = rng.choice(pa[0])
                for l in pa[1:]:
                    if len(l) >= 3 and rng.random() < 0.35: l[rng.randrange(1, len(l) - 1)] = key
            paras.append(pa)
        q = dict(p); q['paras'] = paras; out.append(q)
    return out


def disc_battery(Xr, Xf, seed, nsub, keys):
    rng = np.random.RandomState(1000 + seed)
    res = {'lr': cv_auc(Xr, Xf, 'ridge', seed=seed, C=0.05), 'gbm': cv_auc(Xr, Xf, 'gbm', seed=seed)}
    subs = []
    for t in range(nsub):
        cols = rng.choice(len(keys), 6, replace=False)
        subs.append((cv_auc(Xr, Xf, 'ridge', seed=seed, cols=cols, C=0.05), cols.tolist()))
    res['sub'] = subs
    res['z'] = paired_z(Xr, Xf).tolist()
    return res


def job(args):
    tag, cname, fname, seed, par = args
    ck = f'c1_{tag}_{cname}_{fname}_{seed}.json'
    got = load(ck)
    if got: return got
    import warnings; warnings.filterwarnings('ignore')
    Cs = corpora(); C = Cs[cname]
    FZ = Featurizer(C, LAB)
    rng = random.Random(seed)
    if tag == 'main':
        real = C
        forged = make_forger(fname, C, par).forge(C, rng)
    elif tag == 'null':          # forgery A (as 'real') vs forgery B, same forger
        fg = make_forger(fname, C, par)
        real = fg.forge(C, random.Random(10_000 + seed)); forged = fg.forge(C, rng)
    elif tag == 'neg':           # F6 output as 'real', re-forged by F6 refitted on it
        S = make_forger(fname, C, par).forge(C, random.Random(20_000 + seed))
        real = S; forged = make_forger(fname, S, par).forge(S, rng)
    elif tag == 'plant':         # planted long-range dependency in F6 output, re-forged by F6 refitted
        S = make_forger(fname, C, par).forge(C, random.Random(30_000 + seed))
        P = plant_longrange(S, random.Random(40_000 + seed))
        real = P; forged = make_forger(fname, P, par).forge(P, rng)
    Xr, keys = FZ.matrix(real); Xf, _ = FZ.matrix(forged, keys)
    res = disc_battery(Xr, Xf, seed, NSUB if tag == 'main' else NSUB // 3, keys)
    res.update(tag=tag, corpus=cname, forger=fname, seed=seed, keys=keys)
    save(ck, res)
    return res


def main():
    t0 = time.time()
    Cs = corpora()
    pars = {}
    for cn, C in Cs.items():
        p = load(f'c1_cal_{cn}.json')
        if not p:
            FZ = Featurizer(C, LAB)
            err, p = calibrate(C, FZ)
            p['err'] = err; save(f'c1_cal_{cn}.json', p)
        pars[cn] = p
        print(cn, 'pages', len(C), 'tokens', ntok(C), 'cal', p, flush=True)
    jobs = []
    for s in range(NSEED):
        for f in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']:
            jobs.append(('main', 'V', f, s, pars['V']))
        for cn in ('LA', 'IT'):
            for f in ['F1', 'F3', 'F5', 'F6', 'F7']:
                jobs.append(('main', cn, f, s, pars[cn]))
        for cn in ('V', 'LA', 'IT'):
            jobs.append(('null', cn, 'F7', s, pars[cn]))
            jobs.append(('neg', cn, 'F7', s, pars[cn]))
        jobs.append(('null', 'V', 'F3', s, pars['V']))
        jobs.append(('plant', 'V', 'F7', s, pars['V']))
    with Pool(2) as pool:
        R = []
        for r in pool.imap_unordered(job, jobs):
            R.append(r)
            print(f"{r['tag']:5s} {r['corpus']:2s} {r['forger']} s{r['seed']} lr {r['lr']:.3f} gbm {r['gbm']:.3f} "
                  f"subMax {max(a for a, _ in r['sub']):.3f} [{time.time() - t0:.0f}s]", flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
