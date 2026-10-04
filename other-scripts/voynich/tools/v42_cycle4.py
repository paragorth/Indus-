"""v42 cycle 4: forgery detectability (v21) for a long asemic book.

The v21 forger ladder (F3 junction + section + position tables; F4 in-word slot grammar; F6 'kitchen' = F3 + page
topic + line-initial chain + reduplication + self-citation) re-draws every word of every page, keeping the
page/paragraph/line skeleton. Page-level v21 features; ridge and boosted-tree discriminators, 5-fold CV by page
(paired real/forged). AUC 0.5 = forgery indistinguishable.
Objects: Codex Seraphinianus (raw OCR, and repeats collapsed), Voynich ZL3b (clean and through the n18 channel),
Latin and Italian herbals (clean and n18). Null control: a forgery against a second forgery of the same corpus.
"""
import os, sys, json, random, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v21_lib as V

FN = 'v42_cycle4.txt'
SEEDS = (0, 1, 2)


def noise_pages(P, rate, seed):
    docs = [[l for pa in p['paras'] for l in pa] for p in P]
    tw = L.twins(docs); rng = random.Random(seed); out = []
    for p in P:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                nl = [v for v in (L.noisy_word(w, rng, tw, rate) for w in l) if v]
                if nl: npa.append(nl)
            if npa: q['paras'].append(npa)
        out.append(q)
    return out


def map_pages(P, f):
    return [dict(p, paras=[[[f(w) for w in l] for l in pa] for pa in p['paras']]) for p in P]


def corpora():
    C = {}
    cs = [p for p in L.cs_pages() if sum(len(l) for pa in p['paras'] for l in pa) >= 40]
    C['CS_raw'] = cs; C['CS_col'] = map_pages(cs, L.collapse)
    vz = V.voynich_pages('ZL3b'); C['V_clean'] = vz; C['V_n18'] = noise_pages(vz, 0.18, 1)
    la = V.latin_herbal(); it = V.italian_herbal()
    C['LA_clean'] = la; C['LA_n18'] = noise_pages(la, 0.18, 2)
    C['IT_n18'] = noise_pages(it, 0.18, 3)
    return C


def forger(name, P):
    if name == 'F3': return V.Forger(P, scope='sec', pos=True, name=name)
    if name == 'F4': return V.SlotForger(P)
    return V.Forger(P, scope='sec', pos=True, lam=0.3, chain=True, redup=0.008, cite=0.03, name=name)


def job(a):
    cname, fname, seed = a
    P = CORP[cname]
    F = forger(fname, P)
    Q = F.forge(P, random.Random(seed)); Q2 = F.forge(P, random.Random(seed + 100))
    FZ = V.Featurizer(P)
    Xr, keys = FZ.matrix(P); Xf, _ = FZ.matrix(Q, keys); Xf2, _ = FZ.matrix(Q2, keys)
    for M in (Xr, Xf, Xf2): M[~np.isfinite(M)] = 0
    r = {'ridge': V.cv_auc(Xr, Xf, 'ridge', seed=seed), 'gbm': V.cv_auc(Xr, Xf, 'gbm', seed=seed),
         'null_ridge': V.cv_auc(Xf2, Xf, 'ridge', seed=seed), 'null_gbm': V.cv_auc(Xf2, Xf, 'gbm', seed=seed)}
    z = V.paired_z(Xr, Xf); top = sorted(zip(np.abs(z), keys), reverse=True)[:5]
    r['top'] = [(k, round(float(v), 1)) for v, k in top]
    r['npages'] = len(P)
    return cname, fname, seed, r


CORP = None


def main():
    global CORP
    CORP = corpora()
    for k, P in CORP.items(): print(k, len(P), V.ntok(P), flush=True)
    J = [(c, f, s) for c in CORP for f in ('F3', 'F4', 'F6') for s in SEEDS]
    res = L.load('cycle4.json') or {}
    J = [j for j in J if f'{j[0]}|{j[1]}|{j[2]}' not in res]
    with Pool(2) as p:
        for c, f, s, r in p.imap_unordered(job, J):
            res[f'{c}|{f}|{s}'] = r; L.save('cycle4.json', res)
            print(c, f, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    n = 1
    for c in CORP:
        parts = []
        for f in ('F3', 'F4', 'F6'):
            rs = [res[f'{c}|{f}|{s}'] for s in SEEDS]
            parts.append(f"{f} ridge {np.mean([r['ridge'] for r in rs]):.3f} gbm {np.mean([r['gbm'] for r in rs]):.3f} (null {np.mean([r['null_ridge'] for r in rs]):.2f}/{np.mean([r['null_gbm'] for r in rs]):.2f}); top {rs[0]['top'][:3]}")
        L.row(FN, f'V-42.4.{n}', f'{c} ({rs[0]["npages"]} pages): v21 forgers F3 (junction), F4 (slot), F6 (kitchen) x 3 seeds; page features; ridge / boosted AUC, 5-fold by page; null = forgery vs forgery',
              ' | '.join(parts), 'see verdict'); n += 1


if __name__ == '__main__':
    main()
