"""pe15 cycle 3: predictions from the fitted food web.

(a) LINK HOLD-OUT: hide 20% of the distinct consumer-resource links (all their entries),
    refit the DC-SBM at the block numbers chosen in cycle 2, and rank the hidden links
    among all never-observed pairs of the same consumers.  Scorers: SBM P(r|c); DEG =
    resource popularity (configuration-model predictor); KNN = cosine neighbours'
    resources (structure without blocks).  AUC; 8 random hides per network.
    Nulls: the same on event-re-paired networks (fixed weighted degrees) and on a
    planted modular network.
(b) NEW NAMES: tablets split 80/20 (5 folds).  For held-out entries whose full middle
    string never occurs in the training tablets ("first appearance"), predict the
    resource (class sign | number system) from the signs it is spelled with.
    Scorers: DEG popularity; SIGN-HIST = mean of the signs' own smoothed resource
    profiles; SIGN-SBM = mean of the signs' block profiles (SBM fitted on the sign
    network of the training tablets).  Control: the same strings with their signs
    replaced by random training signs of the same frequency band (kills any real
    spelling -> goods link).  Controls corpora: Ur III names (syllables -> commodity),
    Linear B words (syllables -> ideogram).
"""
import json, os, sys, random
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe15_common import *

RESTARTS = int(os.environ.get('RESTARTS', 6))
SWEEPS = int(os.environ.get('SWEEPS', 150))


def chosen_k():
    K = {}
    p = os.path.join(CK, 'c2.json')
    if os.path.exists(p):
        for r in json.load(open(p)):
            if r['kind'] == 'real':
                K[r['net']] = tuple(map(int, r['SBM_k'].split('_')[1:]))
    return K


def link_job(args):
    name, ev, kc, kr, seed, kind = args
    rng = np.random.default_rng(seed)
    if kind == 'glob':
        rs = [e[1] for e in ev]
        random.Random(seed).shuffle(rs)
        ev = [(e[0], r, e[2]) for e, r in zip(ev, rs)]
    W, C, R = matrix(ev, min_c=2, min_r=2)
    B = W > 0
    cd, rd = B.sum(1), B.sum(0)
    links = [(i, j) for i, j in zip(*np.nonzero(B)) if cd[i] >= 2 and rd[j] >= 2]
    rng.shuffle(links)
    hide = links[:int(0.2 * len(links))]
    Wt = W.copy()
    for i, j in hide:
        Wt[i, j] = 0
    keep_r = Wt.sum(0) > 0
    out = {'net': name, 'kind': kind, 'seed': seed, 'hidden': len(hide), 'K': [kc, kr]}
    bc, br, L = sbm_fit(Wt, min(kc, Wt.shape[0]), min(kr, Wt.shape[1]), restarts=RESTARTS, sweeps=SWEEPS, seed=seed)
    P = cond_prob(Wt, bc, br, min(kc, Wt.shape[0]), min(kr, Wt.shape[1]))
    pop = Wt.sum(0).astype(float)
    X = (Wt > 0).astype(float)
    nx = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = nx @ nx.T
    np.fill_diagonal(S, 0)
    KN = S @ X
    hid = set(hide)
    cons = sorted({i for i, j in hide})
    sc = {'SBM': ([], []), 'DEG': ([], []), 'KNN': ([], [])}
    for i in cons:
        for j in range(W.shape[1]):
            if not keep_r[j] or Wt[i, j] > 0:
                continue
            if (i, j) in hid:
                k = 0
            elif W[i, j] == 0:
                k = 1
            else:
                continue
            sc['SBM'][k].append(P[i, j]); sc['DEG'][k].append(pop[j]); sc['KNN'][k].append(KN[i, j] + 1e-6 * pop[j])
    for m, (pos, neg) in sc.items():
        out[m] = float(auc(pos, neg))
    print('%-10s %-4s s%d K%s hidden %4d  AUC SBM %.3f DEG %.3f KNN %.3f' % (name, kind, seed, (kc, kr), len(hide), out['SBM'], out['DEG'], out['KNN']), flush=True)
    return out


def newname_job(args):
    """ev_str: (string, resource, tablet); strings are space- or hyphen-separated signs."""
    name, ev_str, split, seed, kc, kr, control = args
    rng = random.Random(seed)
    tabs = sorted({e[2] for e in ev_str})
    rng.shuffle(tabs)
    fo = {t: i % 5 for i, t in enumerate(tabs)}
    tot = defaultdict(float)
    n = 0
    for f in range(5):
        tr = [e for e in ev_str if fo[e[2]] != f]
        te = [e for e in ev_str if fo[e[2]] == f]
        seen = {e[0] for e in tr}
        sev = [(s, e[1], e[2]) for e in tr for s in split(e[0])]
        W, C, R = matrix(sev, min_c=1, min_r=1)
        ci = {c: i for i, c in enumerate(C)}; ri = {r: i for i, r in enumerate(R)}
        sfreq = W.sum(1)
        # frequency bands for the scramble control
        order = np.argsort(sfreq)
        band = {}
        nb = 10
        for b in range(nb):
            for i in order[b * len(order) // nb:(b + 1) * len(order) // nb]:
                band[C[i]] = b
        members = defaultdict(list)
        for c, b in band.items():
            members[b].append(c)
        Hs = hist_prob(W, 1.0)
        bc, br, L = sbm_fit(W, min(kc, W.shape[0]), min(kr, W.shape[1]), restarts=RESTARTS, sweeps=SWEEPS, seed=seed * 10 + f)
        Ps = cond_prob(W, bc, br, min(kc, W.shape[0]), min(kr, W.shape[1]))
        pop = (W.sum(0) + 0.1) / (W.sum() + 0.1 * W.shape[1])
        for s, r, t in te:
            if s in seen or r not in ri:
                continue
            sg = [x for x in split(s) if x in ci]
            if not sg:
                continue
            if control:
                sg = [rng.choice(members[band[x]]) for x in sg]
            j = ri[r]
            idx = [ci[x] for x in sg]
            n += 1
            tot['DEG'] += -np.log2(pop[j])
            tot['SIGN_HIST'] += -np.log2(Hs[idx].mean(0)[j])
            tot['SIGN_SBM'] += -np.log2(Ps[idx].mean(0)[j])
            tot['MIX'] += -np.log2(0.5 * Hs[idx].mean(0)[j] + 0.5 * Ps[idx].mean(0)[j])
            # top-1 accuracy
            tot['acc_DEG'] += float(np.argmax(pop) == j)
            tot['acc_SIGN_HIST'] += float(np.argmax(Hs[idx].mean(0)) == j)
            tot['acc_SIGN_SBM'] += float(np.argmax(Ps[idx].mean(0)) == j)
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out.update({'net': name, 'n': n, 'control': control, 'seed': seed})
    print('%-8s %-9s s%d n %4d bits: DEG %.3f SIGN_HIST %.3f SIGN_SBM %.3f MIX %.3f | top1 DEG %.3f HIST %.3f SBM %.3f' % (
        name, 'SCRAMBLE' if control else 'real', seed, n, out['DEG'], out['SIGN_HIST'], out['SIGN_SBM'], out['MIX'],
        out['acc_DEG'], out['acc_SIGN_HIST'], out['acc_SIGN_SBM']), flush=True)
    return out


def run_job(j):
    return link_job(j[1]) if j[0] == 'link' else newname_job(j[1])


def split_pe(s):
    return s.split(' ')


def split_syl(s):
    import re
    return [x for x in re.split(r'[- ]', re.sub(r'\{[^}]*\}', '', s)) if x]


if __name__ == '__main__':
    D = load_nets()['nets']
    K = chosen_k()
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    jobs = []
    if part in ('all', 'link'):
        nets = {k: [tuple(e) for e in D[k]] for k in ('PE_SIGN', 'PE_STR', 'PE_HDR', 'PE_HRES', 'UR3_DAB', 'LB_STR')}
        nets['UR3_STR'] = [tuple(e) for e in subsample_tablets(D['UR3_STR'], len(D['PE_STR']), 0)]
        nets['UR3_SIGN'] = [tuple(e) for e in subsample_tablets(D['UR3_SIGN'], len(D['PE_SIGN']), 0)]
        ev, _ = planted(480, 41, 6700, 5, 0)
        nets['PLANT_MOD'] = ev
        for nm, ev in nets.items():
            kc, kr = K.get(nm, (6, 4))
            if (kc, kr) == (1, 1):
                kc, kr = 4, 3
            for s in range(8):
                jobs.append(('link', (nm, ev, kc, kr, s, 'real')))
            for s in range(4):
                jobs.append(('link', (nm, ev, kc, kr, 100 + s, 'glob')))
    if part in ('all', 'new'):
        sets = {'PE': ([tuple(e) for e in D['PE_STR']], split_pe, K.get('PE_SIGN', (6, 4))),
                'UR3': ([tuple(e) for e in subsample_tablets(D['UR3_STR'], 6000, 0)], split_syl, K.get('UR3_SIGN', (6, 4))),
                'LB': ([tuple(e) for e in D['LB_STR']], split_syl, K.get('LB_SIGN', (6, 4)))}
        for nm, (ev, sp, (kc, kr)) in sets.items():
            kc, kr = max(kc, 2), max(kr, 2)
            for s in range(3):
                for ctl in (False, True):
                    jobs.append(('new', (nm, ev, sp, s, kc, kr, ctl)))
    with Pool(2) as p:
        res = p.map(run_job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'c3_%s.json' % part), 'w'))
