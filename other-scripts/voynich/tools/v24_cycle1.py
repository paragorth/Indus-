"""v24 cycle 1: what do corrections enforce?  Random-edit null at the same place, ~1,900 candidate rules,
family-wise max-|z| correction; conditional tests identity|legality and legality|identity.
Datasets: planted generator (template+junction repair), planted Latin (word-identity repair),
real Latin (SCTA Plaoul diplomatic <del>/<add>, single-letter edits), Voynich ZL correction notes
(before reconstructable: n=4), Voynich ZL [a:b] single-glyph alternatives (transcriber ambiguity, not
corrections: control for transcriber preference).  Power at small n by subsampling the planted sets.
"""
import os, sys, json, pickle, math, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L

NSIM = 2000


def load(name):
    return pickle.load(open(os.path.join(L.CK, name), 'rb'))


def analyse(tag, lines, sites, slot=None, nsim=NSIM, nrand=1500):
    M = L.Model(lines, *(slot or (None, None)))
    RS = L.RuleSet(M, n_random=nrand, slot=slot is not None)
    P = L.prepare(sites, RS)
    out = dict(tag=tag, n=len(P), kinds=dict(Counter(p['site'].kind for p in P)))
    r0 = L.run_test(P, nsim=nsim, seed=1)
    out['core'] = L.summarize(r0, RS.names, core_only=True)
    out['top'] = L.summarize(r0, RS.names, top=8)
    out['zmax'] = float(r0['zmax']) if r0 else None
    out['pfw_min'] = float(np.nanmin(r0['pfw'])) if r0 else None
    out['n_fw05'] = int(np.nansum(r0['pfw'] < 0.05)) if r0 else None
    rI = L.run_test(P, nsim=nsim, seed=2, match=L.match_legal)
    rL = L.run_test(P, nsim=nsim, seed=3, match=L.match_vocab)
    out['id_given_legal'] = L.summarize(rI, RS.names, core_only=True)[:2] if rI else []
    out['legal_given_id'] = [r for r in (L.summarize(rL, RS.names, core_only=True) if rL else []) if r[0] in ('big1', 'big5', 'bilp', 'tri1', 'slot', 'junc', 'firstlp', 'lastlp')]
    out['n_id_given_legal'] = rI['n'] if rI else 0
    out['n_legal_given_id'] = rL['n'] if rL else 0
    # descriptive: real before is a word / bigram-legal, vs null expectation
    vb = np.array([p['vb'][[0, 2]] for p in P]); nb = np.array([(p['pm'][:, [0, 2]] * p['pw'][:, None]).sum(0) for p in P])
    out['before_word'] = float(vb[:, 0].mean()); out['null_word'] = float(nb[:, 0].mean())
    out['before_legal'] = float(vb[:, 1].mean()); out['null_legal'] = float(nb[:, 1].mean())
    return out, P, RS


def power(P, RS, n, reps=100, nsim=400, seed=5):
    """fraction of size-n subsamples where identity|legal (vocab) and legality|identity (big1/slot) reach p<0.05."""
    rng = random.Random(seed); hits = Counter()
    for r in range(reps):
        sub = rng.sample(P, min(n, len(P)))
        rI = L.run_test(sub, nsim=nsim, seed=r, match=L.match_legal)
        rL = L.run_test(sub, nsim=nsim, seed=r + 999, match=L.match_vocab)
        r0 = L.run_test(sub, nsim=nsim, seed=r + 555)
        if rI and not math.isnan(rI['z'][0]) and rI['z'][0] > 0 and rI['p'][0] < 0.05: hits['id|legal'] += 1
        for nm, i in (('big1', 2), ('slot', 6), ('junc', 9)):
            if rL and not math.isnan(rL['z'][i]) and rL['z'][i] > 0 and rL['p'][i] < 0.05: hits[nm + '|id'] += 1
        if r0 and np.nanmin(r0['pfw']) < 0.05: hits['any_fw'] += 1
    return {k: v / reps for k, v in hits.items()}


def row_core(o):
    return L.fmt_core([r for r in o['core'] if r[0] in ('vocab', 'logf', 'big1', 'bilp', 'tri1', 'slot', 'junc', 'lineinit', 'chsh2', 'firstlp', 'lastlp')])


if __name__ == '__main__':
    V = load('voy.pkl'); PL = load('plaoul.pkl'); PG = load('pgen.pkl'); PK = load('plang.pkl')
    sm = json.load(open(os.path.join(L.DATA, 'derived', 'v7_slotmodels.json')))['voynich_ZL3b_K4']
    vslot = (sm['order'], sm['cuts'])
    res = {}
    jobs = [('planted_gen', PG['lines'], PG['sites'], tuple(PG['slot'])),
            ('planted_lang', PK['lines'], PK['sites'], None),
            ('latin_plaoul', PL['lines'], PL['sites'], None),
            ('voy_T1', V['corpus'], V['T1'], vslot),
            ('voy_T2alt', V['corpus'], V['T2'], vslot)]
    Ps = {}
    for tag, lines, sites, slot in jobs:
        o, P, RS = analyse(tag, lines, sites, slot)
        res[tag] = o; Ps[tag] = (P, RS)
        print(tag, 'n', o['n'], o['kinds'], 'zmax %.1f pfw %.3g nFW %d' % (o['zmax'], o['pfw_min'], o['n_fw05']), flush=True)
        print('  core:', row_core(o)); print('  top:', L.fmt_top(o['top']))
        print('  id|legal (n=%d):' % o['n_id_given_legal'], L.fmt_core(o['id_given_legal']))
        print('  legal|id (n=%d):' % o['n_legal_given_id'], L.fmt_core(o['legal_given_id']))
        print('  before is a word %.2f (null %.2f); before bigram-legal %.2f (null %.2f)' % (o['before_word'], o['null_word'], o['before_legal'], o['null_legal']), flush=True)
    pw = {}
    for tag in ('planted_gen', 'planted_lang', 'latin_plaoul'):
        P, RS = Ps[tag]
        for n in (4, 24, 100):
            if n <= len(P):
                pw[(tag, n)] = power(P, RS, n, reps=60 if n > 24 else 100)
                print('power', tag, n, pw[(tag, n)], flush=True)
    res['power'] = {'%s_%d' % k: v for k, v in pw.items()}
    json.dump(res, open(os.path.join(L.CK, 'c1_results.json'), 'w'), indent=1, default=str)
