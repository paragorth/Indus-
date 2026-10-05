#!/usr/bin/env python3
"""LA-57 readout: run a role ensemble on PLANT (control), Linear A and Linear A shuffles.
Models: dict role -> list of score functions f(X) -> scores.  Each model's scores are rank-normalised within a
corpus; the ensemble score is the mean.  Per Linear A word type (>= MINOCC occurrences) we report the mean
ensemble score and the agreement = share of models that put the type in the top 10 % of types for the role.
Shuffle controls: S1 (types shuffled over slots), S2 (tokens shuffled within documents), 20 each."""
import os, sys, pickle, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C

MINOCC = 3


def type_table(f, models):
    X, types = f['X'], f['types']
    if not models:
        return None
    R = np.column_stack([rankdata(m(X)) / len(X) for m in models])
    ens = R.mean(1)
    T = collections.defaultdict(list)
    for i, t in enumerate(types):
        T[t].append(i)
    keep = [t for t, ii in T.items() if len(ii) >= MINOCC]
    tm = np.array([ens[T[t]].mean() for t in keep])
    per = np.array([[R[T[t], j].mean() for j in range(R.shape[1])] for t in keep])  # types x models
    top = per >= np.quantile(per, 0.9, axis=0)[None, :]
    agree = top.mean(1)
    return dict(ens=ens, types=keep, tmean=tm, agree=agree, T=T)


def readout(models, F, roles, log, ntop=8):
    out = {}
    la = F[('LA', 0)]
    for role in roles:
        ms = models.get(role) or []
        if not ms:
            continue
        # planted control
        pl = []
        for j in range(3):
            f = F[('PLANT', j)]
            rows = [i for i, l in enumerate(f['labs']) if l is not None]
            if any(f['labs'][i] == role for i in rows):
                R = np.column_stack([rankdata(m(f['X'][rows])) for m in ms]).mean(1)
                pl.append(C.auc(R, np.array([f['labs'][i] == role for i in rows])))
        tt = type_table(la, ms)
        # concentration statistic: spread of type means (structure) and max agreement
        def stats(t):
            return float(np.std(t['tmean'])), float(np.max(t['agree'])), float(np.mean(t['agree'] >= 0.5))
        real = stats(tt)
        sh = {tag: [stats(type_table(F[('LA_' + tag, j)], ms)) for j in range(20)] for tag in ('S1', 'S2')}
        o = np.argsort(-tt['tmean'])
        top = [(tt['types'][i], round(float(tt['tmean'][i]), 3), round(float(tt['agree'][i]), 2), len(tt['T'][tt['types'][i]]))
               for i in o[:ntop]]
        # logogram check (COM only): LA logograms vs syllabic words, occurrence level
        lg = None
        if role == 'COM':
            y = np.array([t.startswith('L:') for t in la['types']])
            lg = C.auc(tt['ens'], y)
            lgs = []
            for j in range(20):
                f = F[('LA_S1', j)]
                tts = type_table(f, ms)
                lgs.append(C.auc(tts['ens'], np.array([t.startswith('L:') for t in f['types']])))
            lg = (lg, float(np.median(lgs)), float(np.quantile(lgs, 0.95)))
        p_sd = {tag: (1 + sum(1 for s in v if s[0] >= real[0])) / 21 for tag, v in sh.items()}
        p_ag = {tag: (1 + sum(1 for s in v if s[2] >= real[2])) / 21 for tag, v in sh.items()}
        out[role] = dict(n_models=len(ms), plant=pl, real=real, sh_med={k: np.median(np.array(v), 0).tolist() for k, v in sh.items()},
                         p_sd=p_sd, p_agree=p_ag, top=top, logo=lg)
        log('%s: %d models | PLANT AUC %s | LA spread %.3f (S1 med %.3f P %.2f, S2 med %.3f P %.2f) | share types agree>=0.5 %.3f (S1 %.3f P %.2f, S2 %.3f P %.2f)%s' % (
            role, len(ms), ' '.join('%.2f' % a for a in pl), real[0], out[role]['sh_med']['S1'][0], p_sd['S1'],
            out[role]['sh_med']['S2'][0], p_sd['S2'], real[2], out[role]['sh_med']['S1'][2], p_ag['S1'],
            out[role]['sh_med']['S2'][2], p_ag['S2'],
            (' | logograms-as-COM AUC %.3f (S1 med %.3f q95 %.3f)' % lg) if lg else ''))
        log('   top: ' + '; '.join('%s %.3f ag%.2f n%d' % x for x in top))
    return out
