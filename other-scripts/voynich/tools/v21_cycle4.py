"""v21 cycle 4: CLOSE THE SURVIVORS. F11 = F10 + one known-style mechanism per cycle-3 survivor:
  first_eq_lag1 (adjacent words share their first glyph)  -> junction key also holds the previous word's FIRST unit
  vert_last (last glyph echoes the word above)             -> with prob vprob, redraw (<=10x) until the last unit
                                                              matches the word at the same position one line up
  para_first_in_rest (first-line words not re-used)        -> paragraph-first lines drawn from their own junction tables
  mg_line_var (about one m/g per line)                     -> with prob mgq, redraw words with m/g once the line has one
(vprob, mgq) grid-calibrated on those 4 targets plus the cycle-3 repeat profile. Fresh forger seeds 5-9, so the
survivors chosen on seeds 0-4 are tested out of sample. Same battery and controls as cycles 1 and 3.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_cycle1 import LAB, disc_battery, plant_longrange
from v21_cycle3 import corpora3, CAL2
from v21_lib import *
from multiprocessing import Pool

TGT = CAL2 + ['vert_last', 'mg_line_var', 'first_eq_lag1', 'para_first_in_rest']


def mk11(C, par):
    return Forger(C, scope='sec', pos=True, chain=True, width=True, end_room=6, lam=par['lam10'], redup=par['redup10'],
                  cite=par['cite10'], cite_window=15, cite_edit=par['edit10'], para_lam=True, rich2=True, pfl=True,
                  vprob=par['vprob'], mgq=par['mgq'], name='F11')


def cal_job(cn):
    p = load(f'c4_cal_{cn}.json')
    if p: return cn, p
    C = corpora3()[cn]; par = load(f'c3_cal_{cn}.json'); FZ = Featurizer(C, LAB)
    Xr, keys = FZ.matrix(C); idx = [keys.index(k) for k in TGT]
    tr = Xr[:, idx].mean(0); sd = Xr[:, idx].std(0) / math.sqrt(len(C)) + 1e-9
    best = None
    for v in (0.0, 0.15, 0.3, 0.5):
        for q in (0.0, 0.5, 0.9):
            pp = dict(par, vprob=v, mgq=q)
            Xf, _ = FZ.matrix(mk11(C, pp).forge(C, random.Random(0)), keys)
            err = float((((Xf[:, idx].mean(0) - tr) / sd) ** 2).sum())
            if best is None or err < best[0]: best = (err, pp)
    p = best[1]; p['err11'] = best[0]; save(f'c4_cal_{cn}.json', p)
    return cn, p


def job(args):
    tag, cname, seed = args
    ck = f'c4_{tag}_{cname}_F11_{seed}.json'
    got = load(ck)
    if got: return got
    C = corpora3()[cname]; par = load(f'c4_cal_{cname}.json')
    FZ = Featurizer(C, LAB); rng = random.Random(seed)
    if tag == 'main':
        real, forged = C, mk11(C, par).forge(C, rng)
    elif tag == 'null':
        fg = mk11(C, par); real, forged = fg.forge(C, random.Random(10_000 + seed)), fg.forge(C, rng)
    elif tag == 'neg':
        S = mk11(C, par).forge(C, random.Random(20_000 + seed)); real, forged = S, mk11(S, par).forge(S, rng)
    elif tag == 'plant':
        S = mk11(C, par).forge(C, random.Random(30_000 + seed)); P = plant_longrange(S, random.Random(40_000 + seed))
        real, forged = P, mk11(P, par).forge(P, rng)
    Xr, keys = FZ.matrix(real); Xf, _ = FZ.matrix(forged, keys)
    res = disc_battery(Xr, Xf, seed, 300 if tag == 'main' else 100, keys)
    res.update(tag=tag, corpus=cname, forger='F11', seed=seed, keys=keys)
    save(ck, res)
    return res


def main():
    t0 = time.time()
    with Pool(2) as pool:
        for cn, p in pool.imap_unordered(cal_job, ['V', 'VI', 'LA', 'IT']):
            print(cn, p, f'[{time.time() - t0:.0f}s]', flush=True)
        jobs = []
        for s in range(5, 10):
            jobs += [('main', 'V', s), ('main', 'VI', s), ('main', 'LA', s), ('main', 'IT', s), ('null', 'V', s),
                     ('neg', 'V', s), ('neg', 'LA', s), ('neg', 'IT', s), ('plant', 'V', s)]
        for r in pool.imap_unordered(job, jobs):
            print(f"{r['tag']:5s} {r['corpus']:2s} F11 s{r['seed']} ridge {r['lr']:.3f} gbm {r['gbm']:.3f} "
                  f"subMax {max(a for a, _ in r['sub']):.3f} [{time.time() - t0:.0f}s]", flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
