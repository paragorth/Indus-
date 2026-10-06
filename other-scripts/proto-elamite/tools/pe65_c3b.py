#!/usr/bin/env python3
"""PE-65 cycle 3b: (a) chronology check on the held-out result (Banesh-phase outposts Yahya + Malyan only, from c2);
(b) excess-sharing read-off: which line strings reach more outposts than their frequency predicts under unit
shuffles (1,000 shuffles, documents re-dealt over units)? Scored on controls: Ur III titles, Linear B office words
(AUC among words with >= 3 tokens), planted Susa capital + offices (title/personnel classes) and planted null."""
import sys, os, json, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe65_common import *
from pe65_fit import auc
from pe65_c3 import UR_TIT, LB_TIT
from pe65_c1 import plant

NSH = 1000


def excess(docs, minn=3, tag='x'):
    words = sorted({w for d in docs for w in set(d['words'])})
    wi = {w: i for i, w in enumerate(words)}
    D = len(docs)
    M = np.zeros((D, len(words)), bool)
    for i, d in enumerate(docs):
        for w in set(d['words']):
            M[i, wi[w]] = True
    dfreq = M.sum(0)
    keep = np.where(dfreq >= minn)[0]
    M = M[:, keep]
    site = np.array([d['site'] for d in docs])
    def outposts(lab):
        o = np.zeros(M.shape[1])
        for u in range(NB, NB + K - 1):
            o += M[lab == u].any(0)
        hub = M[lab < NB].any(0)
        return o * hub
    obs = outposts(site)
    rng = np.random.RandomState(seed('pe65-c3b-' + tag) % 2 ** 31)
    sims = np.zeros((NSH, M.shape[1]))
    for s in range(NSH):
        sims[s] = outposts(rng.permutation(site))
    mu = sims.mean(0); sd = sims.std(0) + 0.25
    z = (obs - mu) / sd
    p = ((sims >= obs).sum(0) + 1) / (NSH + 1)
    return [dict(w=words[keep[j]], n=int(dfreq[keep[j]]), obs=int(obs[j]), exp=round(float(mu[j]), 2), z=round(float(z[j]), 2),
                 p=round(float(p[j]), 4)) for j in range(M.shape[1])]


def main():
    res = {}
    c2 = json.load(open(os.path.join(CK, 'c2_results.json')))
    ym = {}
    for nm, v in c2.items():
        ym[nm] = round(float(np.mean([v['held'][u]['prior'] - v['held'][u]['post'] for u in ('Yahya', 'Malyan')])), 3)
    res['banesh_gain'] = ym
    print('Yahya+Malyan gain', ym, flush=True)
    pe = pe_docs_all('pub')
    rows = excess(pe, tag='PE')
    rows.sort(key=lambda r: (r['p'], -r['z']))
    res['PE'] = rows
    print('PE top', rows[:15], flush=True)
    print('PE n words', len(rows), 'p<0.01:', sum(r['p'] < 0.01 for r in rows), 'below exp:', sum(r['obs'] < r['exp'] for r in rows),
          'mean obs-exp', round(float(np.mean([r['obs'] - r['exp'] for r in rows])), 3), flush=True)
    for nm, tit in (('UR', UR_TIT), ('LB', LB_TIT)):
        out = []
        for rep in range(3):
            docs = control_docs(nm, rep)
            rr = excess(docs, tag='%s%d' % (nm, rep))
            # a line string counts as a title line if any of its words is a title
            sep = ' '
            y = [any(t in tit for t in r['w'].split(sep)) for r in rr]
            out.append(dict(auc=auc(np.array([r['z'] for r in rr]), y), ntit=int(sum(y)), n=len(rr),
                            mean_obs_minus_exp=round(float(np.mean([r['obs'] - r['exp'] for r in rr])), 3),
                            sig=sum(r['p'] < 0.01 for r in rr)))
            print(nm, rep, out[-1], flush=True)
        res[nm] = out
    pl = []
    for kind in ('susa_capsat', 'susa_state', 'null'):
        for rep in range(3):
            _, th = plant(kind, 300 + rep)
            docs, _ = sim_docs(seed('pe65-c3b-%s-%d' % (kind, rep)), force_theta=th)
            cls = {}
            for d in docs:
                for w, c in zip(d['words'], d['wclass']):
                    cls[w] = c
            rr = excess(docs, tag='%s%d' % (kind, rep))
            y = [cls.get(r['w']) in (2, 4) for r in rr]
            pl.append(dict(kind=kind, auc=auc(np.array([r['z'] for r in rr]), y), ninst=int(sum(y)), n=len(rr),
                           mean_obs_minus_exp=round(float(np.mean([r['obs'] - r['exp'] for r in rr])), 3),
                           sig=sum(r['p'] < 0.01 for r in rr)))
            print('planted', pl[-1], flush=True)
    res['planted'] = pl
    jdump(res, 'c3b_results.json')


if __name__ == '__main__':
    main()
