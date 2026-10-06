#!/usr/bin/env python3
"""LA-61 cycle 1: typed-glossary search, held-out decoding gain, calibration.
usage: la61_c1.py COND [COND ...]   COND = LA | LAshuf<k> | LB<k> | LBshuf<k> | UR3<k> | UR3shuf<k>
For each condition: 5-fold outer CV by tablet (search on 4 folds with internal 2-fold CV objective,
frozen, scored on the 5th) against controls (types permuted among words, one type for all words,
types without affinities, random assignments), plus a full-data fit with restarts for typing
accuracy against calibration truth (LB, UR3) with a permutation null."""
import sys, time
from la61_common import *

R_OUT = int(os.environ.get('R_OUT', 4)); R_FULL = int(os.environ.get('R_FULL', 8))
ITERS = int(os.environ.get('ITERS', 4000)); NRAND = int(os.environ.get('NRAND', 2000))
OUT = os.path.join(LOOPS, 'la61_cycle1.txt')


def docs_for(cond):
    base = 'UR3' if cond.startswith('UR3') else cond[:2]
    k = int(re.sub(r'\D', '', cond.replace('UR3', '')) or 0)
    docs = load(base)
    pyr = random.Random(seed('la61-draw-%s-%d' % (base, k)))
    if base != 'LA':
        docs = draw(docs, ntok(load('LA')), pyr)
    if 'shuf' in cond:
        docs = shuffle_words(docs, random.Random(seed('la61-shuf-' + cond)))
    return base, docs


def run(cond):
    t0 = time.time()
    base, docs = docs_for(cond)
    rng = np.random.default_rng(seed('la61-c1-' + cond))
    folds = split_tabs(docs, 5, 'la61-outer-' + cond)
    res = dict(cond=cond, n_docs=len(docs), n_tok=ntok(docs), folds=[])
    nscored = 0
    for k in range(5):
        te = folds[k]; tr = [d for j, f in enumerate(folds) if j != k for d in f]
        V = Vocab(tr)
        fits, C1, C2 = fit(V, tr, rng, R_OUT, ITERS, 'la61-inner-%s-%d' % (cond, k))
        nscored += R_OUT * (ITERS + 2 * V.W * (NT + V.K + V.S + 1))
        # massive random phase (recorded, not used for the frozen assignment)
        rs = [cv_objective(random_asg(V, rng), C1, C2) for _ in range(NRAND)]
        nscored += NRAND
        pooled_cv = ll_pooled(C1, C2).sum() + ll_pooled(C2, C1).sum()
        best = best_asg(fits)
        Ctr, Cte = V.counts(tr), V.counts(te)
        g = gain(best, Ctr, Cte)
        perm = []
        for _ in range(20):
            p = rng.permutation(best[0])
            perm.append(gain((p, best[1], best[2]), Ctr, Cte))
        one = [gain((np.full(V.W, t), best[1], best[2]), Ctr, Cte) for t in range(NT)]
        noaff = gain((best[0], np.full(V.W, -1), np.full(V.W, -1)), Ctr, Cte)
        rnd = [gain(random_asg(V, rng), Ctr, Cte) for _ in range(200)]
        hit, hit_sd, n = com_accuracy(best, V, Ctr, te, tr)
        n_te_occ = int(Cte['slot'].sum())
        res['folds'].append(dict(
            W=V.W, n_te_occ=n_te_occ, cv_best=max(f[1] for f in fits), cv_pooled=float(pooled_cv),
            cv_rand_max=float(max(rs)), cv_rand_mean=float(np.mean(rs)),
            gain=g, perm_core=[x['core'] for x in perm], perm_all=[x['all'] for x in perm],
            one_core=max(x['core'] for x in one), one_all=max(x['all'] for x in one),
            noaff=noaff, rnd_core=[x['core'] for x in rnd], rnd_all=[x['all'] for x in rnd],
            com_hit=hit, com_sd=hit_sd, com_n=n))
        print(cond, k, 'core %.1f all %.1f perm %.1f one %.1f noaff %.1f rnd %.1f com %d/%d sd %d' % (
            g['core'], g['all'], np.mean(res['folds'][-1]['perm_core']), res['folds'][-1]['one_core'],
            noaff['core'], np.mean(res['folds'][-1]['rnd_core']), hit, n, hit_sd), flush=True)
    # full-data fit: glossary + typing accuracy against truth
    V = Vocab(docs)
    fits, C1, C2 = fit(V, docs, rng, R_FULL, ITERS, 'la61-full-' + cond)
    nscored += R_FULL * (ITERS + 2 * V.W * (NT + V.K + V.S + 1))
    cons = consensus(fits, V)
    typ = {V.words[w]: TYPES[cons[w][0]] for w in range(V.W)}
    stable = {V.words[w]: TYPES[cons[w][0]] for w in range(V.W) if cons[w][1] >= 0.75}
    res['glossary'] = {V.words[w]: dict(type=TYPES[c[0]], share=c[1],
                                        com=(V.coms[c[2]] if c[2] >= 0 else None), com_share=c[3],
                                        site=(V.sites[c[4]] if c[4] >= 0 else None), site_share=c[5])
                       for w, c in enumerate(cons)}
    res['type_counts'] = dict(Counter(typ.values()))
    res['n_stable'] = len(stable)
    if base != 'LA':
        tr_ = truth(base)
        prng = random.Random(seed('la61-perm-' + cond))
        res['truth_all'] = truth_acc(typ, tr_, prng)
        res['truth_stable'] = truth_acc(stable, tr_, prng)
    res['n_scored'] = nscored; res['sec'] = time.time() - t0
    json.dump(res, open(os.path.join(CK, 'c1_%s.json' % cond), 'w'), default=str)
    F = res['folds']
    core = sum(f['gain']['core'] for f in F); allg = sum(f['gain']['all'] for f in F)
    perm = np.sum([f['perm_core'] for f in F], 0); rnd = np.sum([f['rnd_core'] for f in F], 0)
    one = sum(f['one_core'] for f in F); noaff = sum(f['noaff']['core'] for f in F)
    ch = {c: sum(f['gain'][c] for f in F) for c in CHANNELS}
    ntest = sum(f['n_te_occ'] for f in F)
    chit = sum(f['com_hit'] for f in F); csd = sum(f['com_sd'] for f in F); cn = sum(f['com_n'] for f in F)
    tline = ''
    if 'truth_all' in res and res['truth_all'].get('n'):
        ta = res['truth_all']; ts = res['truth_stable']
        tline = ('; truth: all typed %d/%d right (null %.1f +- %.1f, P %.4f)' % (ta['acc'], ta['n'], ta['null_mean'], ta['null_sd'], ta['P']) +
                 ((', stable %d/%d (null %.1f, P %.4f)' % (ts['acc'], ts['n'], ts['null_mean'], ts['P'])) if ts.get('n') else '') +
                 ', per class ' + '; '.join('%s %d/%d top %s' % (r, a, b, ','.join('%s%d' % x for x in top)) for r, (a, b, top) in ta['per'].items()))
    row = ('| LA-61.1-%s | %d docs, %d tokens; 5-fold outer CV by tablet; search = %d random restarts x %d annealed proposals + greedy polish per fold, '
           'internal 2-fold CV objective, + %d fully random assignments per fold (%s assignments scored in all); frozen, scored on the held-out fold vs the pooled scaffold class. '
           'Controls: types permuted among words (20/fold), one type for all, types without affinities, 200 random assignments/fold | '
           'held-out occurrences %d; gain bits core (slot+size+first line+commodity) %.1f [slot %.1f, size %.1f, first line %.1f, commodity %.1f], site %.1f, all %.1f; '
           'types permuted core %.1f (max of 20 = %.1f), one type %.1f, types w/o affinities %.1f, random %.1f (max %.1f); '
           'commodity in scope right %d/%d (site default %d); full fit: %d words, %d stable (>= 0.75 of %d restarts), types %s%s |' % (
               cond, len(docs), ntok(docs), R_OUT, ITERS, NRAND, format(nscored, ','), ntest, core, ch['slot'], ch['nb'], ch['fl'], ch['com'],
               ch['site'], allg, perm.mean(), perm.max(), one, noaff, rnd.mean(), rnd.max(), chit, cn, csd, V.W, len(stable), R_FULL,
               dict(sorted(res['type_counts'].items())), tline))
    wlog(OUT, row + ' - |')
    print(row, flush=True)


if __name__ == '__main__':
    for c in sys.argv[1:]:
        run(c)
