#!/usr/bin/env python3
"""LA-43 cycle 1: do the borrowed values predict words at other sites and in later publications?

Pre-registered (written before any test score was computed):
  primary score  bits (perplexity of new word types at the test part), gain over the sound-free model,
                 position of the true map among relabelings R2a, R2b, R3; secondary mask, mrr, auc.
  LA splits      SITE  train Hagia Triada, test all other sites
                 TIME  train GORILA 1-5 (published 1976-1985), test post-1985 publications ('blank' source excluded)
  controls       LB  KN -> PY full; LB drawn at the LA split sizes (10 draws each split, KN train docs, PY test docs);
                 Cypriot Idalion tablet, first half -> second half and back;
                 planted correct-values language on the LA grid (PLx, phonotactic generator, 5 draws) must be detected;
                 planted no-phonotactics language (NPx, i.i.d. syllables, 5 draws) must not be.
Predictions of the true map and of the sound-free model are hashed (with the test-set digest) before scoring.
"""
import sys, os, json, time, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la43_common as M
from multiprocessing import Pool

TIERS = ['R2a', 'R2b', 'R3']
LA_SIZES = dict(SITE=(3019, 2539), TIME=(4759, 162))


def la_split(kind):
    U = M.la_units()
    if kind == 'SITE':
        return [u for u in U if u['site'] == 'Haghia Triada'], [u for u in U if u['site'] != 'Haghia Triada']
    if kind == 'TIME':
        return [u for u in U if u['pub'].startswith('G')], [u for u in U if u['pub'] == 'post']
    raise ValueError(kind)


def planted(seed, phon=True, ntr=3019, nte=2539):
    """a synthetic language written on the LA grid with the LA (borrowed) values as the true values.
    phon=True: vowel harmony, consonant dissimilation, onset preference, Zipfian consonants and suffixes;
    phon=False: syllables drawn i.i.d. from the LA sign frequencies (no sound structure)."""
    rng = np.random.default_rng(77_000 + seed + (0 if phon else 500))
    U = M.la_units()
    cnt = {}
    for u in U:
        for s in u['w']:
            cnt[s] = cnt.get(s, 0) + 1
    cells = {}
    for s in cnt:
        v = M.L38.cv_of(s)
        if v is not None:
            cells.setdefault(v, s)
    Cs = sorted({c for c, v in cells}); Vs = sorted({v for c, v in cells})
    unval = sorted([s for s in cnt if M.L38.cv_of(s) is None], key=lambda s: -cnt[s])[:15]
    signs = sorted(cells.values(), key=lambda s: -cnt[s]); fr = np.array([cnt[s] for s in signs], float); fr /= fr.sum()
    cw = rng.permutation(len(Cs)); cw = 1.0 / (1 + np.argsort(cw)) ** 0.8; cw /= cw.sum()
    harm = {'i': 'ie', 'e': 'ie', 'u': 'uo', 'o': 'uo', 'a': 'aeiou'}

    def syl(prev, first):
        for _ in range(50):
            if first and rng.random() < 0.2:
                c = ''
            else:
                c = Cs[rng.choice(len(Cs), p=cw)]
                if c == '' and not first and rng.random() < 0.85:
                    continue
            if prev is not None and c == prev[0] and c != '' and rng.random() < 0.9:
                continue
            if prev is not None and rng.random() < 0.7:
                v = rng.choice(list(harm[prev[1]]))
            else:
                v = rng.choice(Vs)
            if (c, v) in cells:
                return (c, v)
        return list(cells)[rng.integers(len(cells))]

    def word():
        if not phon:
            n = rng.choice([2, 3, 4], p=[.5, .35, .15])
            return tuple(signs[i] for i in rng.choice(len(signs), n, p=fr))
        n = rng.choice([1, 2, 3], p=[.3, .5, .2]); out = []; prev = None
        for k in range(n):
            prev = syl(prev, k == 0); out.append(prev)
        out = [cells[x] for x in out]
        return tuple(out)
    suff = [word()[:1] for _ in range(6)]
    lex = []
    for _ in range(3000):
        w = word()
        if phon and rng.random() < 0.5:
            w = w + suff[rng.choice(6, p=[.35, .25, .15, .1, .1, .05])]
        if rng.random() < 0.05:
            w = w + (unval[rng.integers(len(unval))],)
        lex.append(w)
    zipf = 1.0 / np.arange(1, 3001) ** 1.0

    def corpus(lexidx, ntok, tag):
        p = zipf[:len(lexidx)] / zipf[:len(lexidx)].sum()
        out, n, d = [], 0, 0
        while n < ntok:
            for _ in range(8):
                w = lex[lexidx[rng.choice(len(lexidx), p=p)]]
                out.append(dict(w=w, doc=f'{tag}{d}', site=tag)); n += len(w)
            d += 1
        return out
    perm = rng.permutation(3000)
    A = perm[:1500]; B = np.concatenate([perm[:400], perm[1500:2600]]); rng.shuffle(B)
    return corpus(A, ntr, 'A'), corpus(B, nte, 'B')


def make(tag):
    """experiment tag -> (train, test)."""
    if tag in ('LA_SITE', 'LA_TIME'):
        return la_split(tag[3:])
    if tag == 'LB_KNPY':
        L = M.lb_units()
        return [u for u in L if u['site'] == 'KN'], [u for u in L if u['site'] == 'PY']
    if tag.startswith('LB_SITE_') or tag.startswith('LB_TIME_'):
        kind, d = tag.split('_')[1], int(tag.split('_')[2])
        L = M.lb_units(); ntr, nte = LA_SIZES[kind]
        return (M.draw_docs([u for u in L if u['site'] == 'KN'], ntr, 100 + d),
                M.draw_docs([u for u in L if u['site'] == 'PY'], nte, 200 + d))
    if tag.startswith('CYP'):
        W = M.L38.cyp_words(); h = len(W) // 2
        a = [dict(w=w, doc=f'c{i // 10}', site='I') for i, w in enumerate(W[:h])]
        b = [dict(w=w, doc=f'd{i // 10}', site='I') for i, w in enumerate(W[h:])]
        return (a, b) if tag == 'CYP_AB' else (b, a)
    if tag.startswith('PL_') or tag.startswith('NP_'):
        return planted(int(tag[3:]), phon=tag.startswith('PL'))
    raise ValueError(tag)


def experiment(tag):
    tr, te = make(tag)
    if tag.startswith('PL_') or tag.startswith('NP_') or tag.startswith('LA_'):
        return M.Experiment(tr, te)
    return M.Experiment(tr, te)


def job(args):
    tag, tier, N, seed = args
    E = experiment(tag)
    Pt, wt, Pn, wn, h = E.frozen()
    st = E.test.score(Pt); sn = E.test.score(Pn)
    t = time.time()
    null = E.null(tier, N, seed)
    res = M.summarise(sn, st, null)
    return dict(tag=tag, tier=tier, N=N, hash=h, test=E.test.digest(), ntr=E.ntr, ntypes=E.nte_types,
                w=wt.round(3).tolist(), res=res, sec=round(time.time() - t))


def main():
    out_p = os.path.join(M.CK, 'c1.json')
    done = json.load(open(out_p)) if os.path.exists(out_p) else []
    have = {(d['tag'], d['tier']) for d in done}
    jobs = []
    for i, tag in enumerate(['LA_SITE', 'LA_TIME']):
        for k, t in enumerate(TIERS):
            jobs.append((tag, t, 10000, 10 * i + k))
    for k, t in enumerate(TIERS):
        jobs.append(('LB_KNPY', t, 2000, 50 + k))
        for c in ('CYP_AB', 'CYP_BA'):
            jobs.append((c, t, 2000, 60 + k))
    for d in range(10):
        for kind in ('SITE', 'TIME'):
            for k, t in enumerate(TIERS):
                jobs.append((f'LB_{kind}_{d}', t, 500, 1000 + 10 * d + k))
    for d in range(5):
        for p in ('PL', 'NP'):
            for k, t in enumerate(TIERS):
                jobs.append((f'{p}_{d}', t, 500, 2000 + 10 * d + k))
    # freeze: hashes of the true-map and sound-free predictions and of the test sets, before scoring
    fz = os.path.join(M.CK, 'c1_freeze.json')
    if not os.path.exists(fz):
        F = {}
        for tag in sorted({j[0] for j in jobs}):
            E = experiment(tag); h = E.frozen()[4]
            F[tag] = dict(pred=h, test=E.test.digest(), seeds=sorted(j[3] for j in jobs if j[0] == tag))
        json.dump(F, open(fz, 'w'), indent=1)
        print('frozen', hashlib.sha256(open(fz, 'rb').read()).hexdigest()[:16], flush=True)
    jobs = [j for j in jobs if (j[0], j[1]) not in have]
    # cheap jobs first so controls arrive early, LA last
    jobs.sort(key=lambda j: j[2])
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            done.append(r)
            json.dump(done, open(out_p, 'w'))
            print(r['tag'], r['tier'], r['ntr'], r['ntypes'], M.fmt(r['res'], ['bits', 'mask']), r['sec'], 's', flush=True)


if __name__ == '__main__':
    main()
