"""v26 evaluation of evolved genomes (cycle 3).

For a corpus: pick the winner among every restart's final population (top 3 per restart re-forged with 3 new
seeds), then:
  A. fitted half: ridge and boosting page AUC (v21 cv_auc, 5 seeds), null (genome vs itself), negative
     (genome output re-forged by the genome refitted on it = this generator's own re-forge floor);
     the same for the base junction genome (sec+pos tables = v21 F3).
  B. minimal genome: backward elimination of mechanisms while the 3-seed ridge AUC rises by < 0.01.
  C. held-out half (Voynich only needed, run for all): tables from the fitted half, forge the held-out pages'
     skeletons; ridge / boosting AUC; survivor statistics (paired z, real minus forged, page means over 5 seeds);
     per-glyph perplexity of the held-out text vs n-gram baselines.
Usage: python3 v26_eval.py CORPUS
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v26_lib import *
from v26_evo import get_corpus, CSU, LAMBDA, PLANT
from multiprocessing import Pool

BASE = dict(OFF, sec=1, pos=1)
SURV = ['near_far', 'CS_switch', 'para_first_in_rest', 'vert_last', 'mg_line_var', 'first_eq_lag1', 'wlen_ac1',
        'rep_sameline', 'last_eq_lag2', 'rep_line2_4', 'ttr', 'rep_prevline', 'near_prev']
_W = {}


def winit(name):
    import warnings; warnings.filterwarnings('ignore')
    C = get_corpus(name); tr, te = split_half(C)
    S = Scribe(tr, CSU[name]); FZ = FastFZ(tr)
    Xr, keys = FZ.matrix(tr); Xte, _ = FZ.matrix(te, keys)
    _W.update(name=name, tr=tr, te=te, S=S, FZ=FZ, Xr=Xr, Xte=Xte, keys=keys)


def j_train(args):
    """(genome, seed) -> ridge, gbm AUC on the fitted half and the forged feature matrix."""
    g, seed, gb = args
    W = _W
    Xf, _ = W['FZ'].matrix(W['S'].forge(W['tr'], g, random.Random(seed)), W['keys'])
    r = cv_auc(W['Xr'], Xf, 'ridge', seed=seed, C=0.05)
    b = cv_auc(W['Xr'], Xf, 'gbm', seed=seed) if gb else None
    return r, b, Xf.tolist()


def j_null(args):
    g, seed = args
    W = _W
    A = W['S'].forge(W['tr'], g, random.Random(seed)); B = W['S'].forge(W['tr'], g, random.Random(seed + 777))
    Xa, _ = W['FZ'].matrix(A, W['keys']); Xb, _ = W['FZ'].matrix(B, W['keys'])
    return cv_auc(Xa, Xb, 'ridge', seed=seed, C=0.05), cv_auc(Xa, Xb, 'gbm', seed=seed)


def j_neg(args):
    g, seed = args
    W = _W
    A = W['S'].forge(W['tr'], g, random.Random(seed + 555))
    S2 = Scribe(A, CSU[W['name']]); FZ2 = FastFZ(A)
    B = S2.forge(A, g, random.Random(seed))
    Xa, keys = FZ2.matrix(A); Xb, _ = FZ2.matrix(B, keys)
    return cv_auc(Xa, Xb, 'ridge', seed=seed, C=0.05), cv_auc(Xa, Xb, 'gbm', seed=seed)


def j_held(args):
    g, seed, kind = args
    W = _W
    if kind == 'bigram':
        wb = WordBigram(W['tr']); rng = random.Random(seed)
        F = [wb.generate(p, rng) for p in W['te']]
    else:
        F = W['S'].forge(W['te'], g, random.Random(seed))
    Xf, _ = W['FZ'].matrix(F, W['keys'])
    return cv_auc(W['Xte'], Xf, 'ridge', seed=seed, C=0.05), cv_auc(W['Xte'], Xf, 'gbm', seed=seed), Xf.tolist()


def mech_groups(g):
    """Top-level mechanisms (dependent genes go with their parent)."""
    return [k for k in active(g) if k not in DEPENDS]


def drop(g, k):
    h = dict(g); h[k] = OFF[k]
    for c, par in DEPENDS.items():
        if par == k: h[c] = OFF[c]
    return h


def perplexity(name, tr, te, gs):
    """bits per glyph event (units + word end) on the held-out half; slot escape eps from an internal split."""
    A, B = split_half(tr, seed=27)
    VA = set(w for p in A for w in tokens(p))
    eps = max(0.005, float(np.mean([w not in VA for p in B for w in tokens(p)])))
    S = Scribe(tr, CSU[name])
    out = {'eps': eps}
    for lab, g in gs.items():
        lp = n = 0
        for p in te:
            a, b = S.walk(p, g, eps=eps); lp += a; n += b
        out[lab] = -lp / n
    # likelihood-tuned version of the winner's mechanisms (coordinate search on the internal split A -> B)
    SA = Scribe(A, CSU[name])

    def bpg(g):
        lp = n = 0
        for p in B:
            a, b = SA.walk(p, g, eps=eps); lp += a; n += b
        return -lp / n
    g = dict(gs['evolved']); cur = bpg(g)
    for it in range(2):
        for k in [x for x in active(g) if GENES[x][0] == 'num']:
            lo, hi = GENES[k][1]
            for v in np.linspace(lo, hi, 6):
                h = dict(g); h[k] = float(v)
                if k.startswith('pi_') and sum(h[x] for x in ('pi_copy', 'pi_urn', 'pi_redup')) > 0.9: continue
                b = bpg(h)
                if b < cur: cur, g = b, h
    lp = n = 0
    for p in te:
        a, b = S.walk(p, g, eps=eps); lp += a; n += b
    out['evolved_LT'] = -lp / n; out['LT_genome'] = gstr(g)
    # baselines
    for lam in (0.0, 0.05, 0.1, 0.2):
        wbA = WordBigram(A, lam); lpA = sum(wbA.logprob(p, SA, eps)[0] for p in B)
        out.setdefault('_cache_tune', {})[lam] = lpA
    lam = max(out['_cache_tune'], key=out['_cache_tune'].get)
    for lab, l in (('bigram', 0.0), ('bigram+cache', lam)):
        wb = WordBigram(tr, l); lp = n = 0
        for p in te:
            a, b = wb.logprob(p, S, eps); lp += a; n += b
        out[lab] = -lp / n
    out['cache_lam'] = lam
    for o in (3, 4, 5, 6):
        gm = GlyphWB(tr, o); lp = n = 0
        for p in te:
            a, b = gm.logprob(p); lp += a; n += b
        out[f'glyph{o}'] = -lp / n
    del out['_cache_tune']
    return out


def main():
    name = sys.argv[1]
    ck = f'eval_{name}.json'
    res = load(ck) or {}
    t0 = time.time()
    pool = Pool(2, initializer=winit, initargs=(name,))
    # 1. winner
    if 'winner' not in res:
        cands = []
        for R in range(10):
            st = load(f'evo_{name}_r{R}.json')
            if st: cands += [e['g'] for e in st['pop'][:3]]
        sc = []
        for g in cands:
            rs = pool.map(j_train, [(g, 900 + s, False) for s in range(3)])
            sc.append(float(np.mean([r[0] for r in rs])) + LAMBDA * bits(g))
        i = int(np.argmin(sc))
        res['winner'] = cands[i]; res['cand_scores'] = [(gstr(g), s) for g, s in zip(cands, sc)]
        save(ck, res); print('winner', gstr(cands[i]), sc[i], f'[{time.time() - t0:.0f}s]', flush=True)
    W = res['winner']
    # 2. battery on fitted half
    for lab, g in (('win', W), ('base', BASE)):
        if lab + '_train' in res: continue
        rs = pool.map(j_train, [(g, s, True) for s in range(5)])
        nu = pool.map(j_null, [(g, s) for s in range(3)])
        ne = pool.map(j_neg, [(g, s) for s in range(3)])
        Xf = np.mean([np.array(r[2]) for r in rs], 0)
        winit(name)
        z = paired_z(_W['Xr'], Xf)
        res[lab + '_train'] = dict(ridge=[r[0] for r in rs], gbm=[r[1] for r in rs], null=nu, neg=ne,
                                   z={k: float(v) for k, v in zip(_W['keys'], z)})
        save(ck, res); print(lab, 'train', np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]), 'null', nu, 'neg', ne,
                             f'[{time.time() - t0:.0f}s]', flush=True)
    # 3. minimal genome
    if 'minimal' not in res:
        g = dict(W)
        def m3(g):
            return float(np.mean([r[0] for r in pool.map(j_train, [(g, 1200 + s, False) for s in range(4)])]))
        cur = m3(g); steps = [(gstr(g), cur)]
        while True:
            ms = mech_groups(g)
            if not ms: break
            tries = [(m3(drop(g, k)), k) for k in ms]
            a, k = min(tries)
            steps.append(dict(tries=[(kk, aa) for aa, kk in tries]))
            if a - cur < 0.01:
                g = drop(g, k); cur = min(cur, a) if a < cur else a; steps.append((gstr(g), a))
            else:
                break
        res['minimal'] = g; res['min_steps'] = steps
        save(ck, res); print('minimal', gstr(g), cur, f'[{time.time() - t0:.0f}s]', flush=True)
    # 4. held-out
    if 'held' not in res:
        out = {}
        winit(name)
        for lab, g in (('win', W), ('minimal', res['minimal']), ('base', BASE), ('bigram', None)):
            rs = pool.map(j_held, [(g, s, 'bigram' if g is None else 'scribe') for s in range(5)])
            Xf = np.mean([np.array(r[2]) for r in rs], 0)
            z = paired_z(_W['Xte'], Xf)
            out[lab] = dict(ridge=[r[0] for r in rs], gbm=[r[1] for r in rs],
                            z={k: float(v) for k, v in zip(_W['keys'], z)},
                            surv={k: [float(_W['Xte'][:, _W['keys'].index(k)].mean()), float(Xf[:, _W['keys'].index(k)].mean()),
                                      float(z[_W['keys'].index(k)])] for k in SURV})
            print('held', lab, np.mean(out[lab]['ridge']), np.mean(out[lab]['gbm']), f'[{time.time() - t0:.0f}s]', flush=True)
        res['held'] = out; save(ck, res)
    if 'ppl' not in res:
        winit(name)
        res['ppl'] = perplexity(name, _W['tr'], _W['te'], {'evolved': W, 'minimal': res['minimal'], 'base': BASE})
        save(ck, res); print('ppl', res['ppl'], f'[{time.time() - t0:.0f}s]', flush=True)
    pool.close()


if __name__ == '__main__':
    main()
