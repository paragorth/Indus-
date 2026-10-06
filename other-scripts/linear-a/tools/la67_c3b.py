#!/usr/bin/env python3
"""LA-67 cycle 3 (part 2): read-outs of the fresh-seed re-runs of earlier machinery, each against its
stated kill line, plus the la31 'without HT' kill test.  usage: la67_c3b.py [ole pade ws qa la44 la31]"""
from la67_lib import *
OUT = os.path.join(LOOPS, 'la67_cycle3.txt')
K = os.path.join(CK)


def ole():
    a = {x[1]: x[4] for x in json.load(open(os.path.join(K, 'la52', 'la67a_LA_wordbest.json')))}
    b = {x[1]: x[4] for x in json.load(open(os.path.join(K, 'la52', 'la67b_LA_wordbest.json')))}
    sh = sorted(set(a) & set(b))
    from scipy.stats import spearmanr
    rho = spearmanr([a[w] for w in sh], [b[w] for w in sh]).correlation
    lowa = [w for w in sh if a[w] < -2]; both = [w for w in lowa if b[w] < -2]
    wlog(OUT, '| LA-67.3g | la52 C: OLE+RI dies under copying faster than its frequency predicts (z -3.4, full corpus). Kill line: z above -1 with a new seed. Re-run: la52 gentle '
              'single-learner chains (n-gram, template, topic; 120 real + 120 shuffled-start chains each, 4 generations) with two NEW tags/seeds; same frequency-matched z. Decoys: '
              'every other scored word (seed-to-seed stability of z) | OLE+RI z %.2f (seed a) and %.2f (seed b); words < -2: %s (a), %s (b); z rank correlation between seeds %.2f '
              '(%d words); %d of %d words below -2 in seed a are below -2 in seed b | %s |' % (
                  a.get('OLE+RI', float('nan')), b.get('OLE+RI', float('nan')), ', '.join(w for w in sh if a[w] < -2), ', '.join(w for w in sh if b[w] < -2), rho, len(sh),
                  len(both), len(lowa), 'KILLED (z above -1 with a new seed)' if max(a.get('OLE+RI', 0), b.get('OLE+RI', 0)) > -1 else 'SURVIVES'))


def pade():
    r = json.load(open(os.path.join(K, 'la53', 'c3.json')))['LA_rank']
    d = {x['type']: x for x in r}
    p = d['PA-DE']['p']; rk = [x['type'] for x in r].index('PA-DE') + 1
    dec = [x for x in r if x['n'] == d['PA-DE']['n'] and x['type'] != 'PA-DE']
    share = np.mean([x['p'] <= p for x in dec])
    m = len(r); nsig = sum(x['p'] < 0.01 for x in r)
    wlog(OUT, '| LA-67.3h | la53 C: PA-DE is a specially marked entry (p 0.010, n 3). Kill line: an ordinary crowded entry on a new tablet. Existing-data test: la53 cycle 3 re-run with '
              'FRESH seeds (6,000 specs chosen on Linear B + Ur III, 1,000 within-stratum permutations); decoys = the %d other types with the same n | PA-DE p %.4f (rank %d of %d; q %.3f); '
              'decoys with p <= that: %.2f; types at p < 0.01: %d (chance %.1f) | %s |' % (
                  len(dec), p, rk, m, d['PA-DE']['q'], share, nsig, 0.01 * m,
                  'SURVIVES (p <= 0.01 again and beyond same-n decoys)' if p <= 0.01 and share <= 0.05 else 'KILLED (inside the decoy range / not significant after the search)'))


def ws():
    import glob
    real = [json.load(open(f))['agg']['bits_gain'] for f in glob.glob(os.path.join(K, 'la54', 'c1_LA_*.json'))]
    new = [json.load(open(f))['agg']['bits_gain'] for f in glob.glob(os.path.join(K, 'la54', 'c3_WS_*.json'))]
    old = [json.load(open(f))['agg']['bits_gain'] for f in glob.glob(os.path.join(HERE, '..', 'data', 'la54_ckpt', 'c3_WS_*.json'))]
    allw = np.array(old + new); rv = float(np.mean(real)) if real else 0.331
    p_new = (np.sum(np.array(new) >= rv) + 1) / (len(new) + 1)
    p_all = (np.sum(allw >= rv) + 1) / (len(allw) + 1)
    wlog(OUT, '| LA-67.3i | la54 C: ~0.16 bits per document about the unwritten commodity beyond site (P 1/9). Kill line: P > 0.2 with more shuffles. Re-run with fresh seeds: the real '
              'pipeline (%d seeds) and %d more within-site label permutations | real bits %s (mean %.3f; original 0.331); new nulls %.3f +- %.3f (max %.3f); all %d nulls P %.3f, new nulls only P %.3f; '
              'beyond-site surplus %.3f bits | %s |' % (
                  len(real), len(new), ', '.join('%.3f' % x for x in real), rv, np.mean(new) if new else float('nan'), np.std(new) if new else float('nan'), max(new) if new else float('nan'),
                  len(allw), p_all, p_new, rv - allw.mean(), 'KILLED (P > 0.2)' if p_all > 0.2 else 'SURVIVES (P <= 0.05)' if p_all <= 0.05 else 'NOT KILLED, NOT SUPPORTED (0.05 < P <= 0.2)'))


def qa():
    def rate(fn, sign, partners=('A', 'U', 'I')):
        z = np.load(fn); s = list(z['signs']); R = z['rows']
        if sign not in s:
            return None
        i = s.index(sign)
        return float(np.mean([np.mean(R[:, i] == R[:, s.index(p)]) for p in partners if p in s]))
    real = os.path.join(K, 'la21', 'la67_LA.npz')
    sh = [os.path.join(K, 'la21', 'la67_LAshW%d.npz' % k) for k in (11, 12, 13)]
    sh = [f for f in sh if os.path.exists(f)]
    q = rate(real, 'QA'); qs = [rate(f, 'QA') for f in sh]
    z = np.load(real); s = list(z['signs'])
    # decoys: every other non-vowel sign's co-assignment with A/U/I in the real run
    dv = {x: rate(real, x) for x in s if x not in ('A', 'U', 'I', 'E', 'O', 'QA')}
    p = pct_rank(q, list(dv.values()))
    top = sorted(dv.items(), key=lambda kv: -kv[1])[:5]
    wlog(OUT, '| LA-67.3j | la21 C: QA behaves like a pure vowel (vowel-row rate 0.70). Kill line: QA at its shuffle level in a larger run. Re-run: la21 annealed grid, %d restarts (fresh tag), '
              'and %d within-word shuffles (400 restarts each); rate = share of restarts with QA in the row of A, U, I. Decoys: every other non-vowel sign in the real run | QA %.2f; shuffles %s; '
              'decoy signs: P %.3f (top: %s) | %s |' % (
                  z['rows'].shape[0], len(sh), q, ', '.join('%.2f' % x for x in qs if x is not None), p, ', '.join('%s %.2f' % kv for kv in top),
                  'SURVIVES (above every shuffle and the decoy signs)' if qs and q > max(qs) and p <= 0.05 else 'KILLED (at shuffle level or inside the decoy signs)'))


def la44():
    r = json.load(open(os.path.join(K, 'la44', 'abc_la67cond.json')))
    T = r['targets']
    la = T['LA']['morph']['post']['FUS_SUF']
    sg = [v['morph']['post']['FUS_SUF'] for k, v in T.items() if k.startswith('LA_shuf')]
    lb = [v['morph']['post']['FUS_SUF'] for k, v in T.items() if k.startswith('LB_sub')]
    p = (np.sum(np.array(sg) >= la) + 1) / (len(sg) + 1)
    wlog(OUT, '| LA-67.3k | la44 C: given affixation, LA leans to Greek-type suffixing (FUS_SUF 0.57; 6/60 shuffles reach it, P 0.10). Kill line: P >= 0.2 with more shuffles. Re-run: a NEW '
              'simulation bank (v2, %d histories, fresh seeds), ISOL removed, robust panel, %d shuffled-LA targets | LA FUS_SUF %.2f (modal %s); LB subsamples %s; shuffles reaching it %d/%d, '
              'P %.3f | %s |' % (
                  r['nsims'], len(sg), la, max(T['LA']['morph']['post'], key=T['LA']['morph']['post'].get), ', '.join('%.2f' % x for x in lb), int(np.sum(np.array(sg) >= la)), len(sg), p,
                  'KILLED (P >= 0.2)' if p >= 0.2 else 'SURVIVES (P <= 0.05)' if p <= 0.05 else 'NOT KILLED, NOT SUPPORTED (0.05 < P < 0.2)'))


def la31():
    import la31_common as L
    from la31_stats import mantel
    travel = json.load(open(os.path.join(L.OUT, 'travel.json')))
    codes = ['HT', 'KH', 'PH', 'KN', 'ZA', 'PK', 'MA', 'TH', 'IO', 'AR', 'PE', 'SY']
    Zs = np.load(os.path.join(L.CKPT, 'c1_Z_big12.npy'))
    rng = np.random.default_rng(seed('la31'))
    def close(A):
        return -np.log(np.where(np.isfinite(A), A, 1e6) + 0.1)
    res = {}
    for nm, keep in (('big12', codes), ('big11 without HT', [c for c in codes if c != 'HT'])):
        ix0 = [codes.index(c) for c in keep]
        tc = travel['codes']; ix = [tc.index(c) for c in keep]
        A = np.array(travel['wind_sailing'], float)[np.ix_(ix, ix)]
        km = close(L.euclid(keep))
        Z = Zs[0][np.ix_(ix0, ix0)]
        r, p, _ = mantel(Z, close(np.minimum(A, A.T)), nperm=5000, rng=rng, covar=km)
        # decoy legs: the same mantel with the sailing matrix replaced by random asymmetric re-pairings of its entries
        dr = []
        for _ in range(200):
            B = A.copy(); iu = np.triu_indices(len(keep), 1); v = np.r_[A[iu], A.T[iu]]; rng.shuffle(v)
            B[iu] = v[:len(iu[0])]; B.T[iu] = v[len(iu[0]):]
            dr.append(mantel(Z, close(np.minimum(B, B.T)), nperm=1, rng=rng, covar=km)[0])
        res[nm] = (r, p, float(np.mean(np.array(dr) >= r)))
    w = res['big11 without HT']
    wlog(OUT, '| LA-67.3l | la31 C: among the big archives, words follow the easier (faster) sailing leg (partial r|km +0.20, P 0.097). Kill line: r <= 0 without HT. Test: the same partial '
              'Mantel (words layer, sailing season, faster leg, 5,000 fresh site permutations) on the 12 big archives and on the 11 without HT; decoys = 200 random re-pairings of the travel '
              'times | %s | %s |' % ('; '.join('%s r %+.2f P %.3f (decoy legs >= r: %.2f)' % (k, *v) for k, v in res.items()),
                                    'KILLED (r <= 0 without HT)' if w[0] <= 0 else 'SURVIVES the literal line (r > 0 without HT)' + ('' if w[1] <= 0.05 else ', but not significant')))


if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()[f]()
        print(f, 'done', flush=True)
