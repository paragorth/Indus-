"""v40 cycle 1: which family of predictors decides each twin choice?
Unique held-out gain (bits/token, page-grouped 5-fold) of each family = LL(full without family) - LL(full).
Nulls: y permuted within (frame x hand) strata (keeps word identity and scribe, breaks layout and pen links),
10 draws -> z for LAYOUT and PEN unique gain.
Controls: DTA Simplicissimus 1669 + Goethe Faust 1808 (long s / round s, r / r rotunda: known context rules),
Caesar (c / g, n / m: real letters = meaning), PLANTED tall-under-tall rule on Voynich k/t at 3 strengths."""
import sys, json, os, time
import numpy as np
from scipy.sparse import hstack
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v40_lib import *

FAMS = ['SCRIBE', 'CTX', 'FRAME', 'LAYOUT', 'PEN']


def vpairs():
    def m(c):
        return 'G' if c in 'kt' else ('Q' if c in 'pf' else ('X' if c in 'CS' else ('B' if c in 'KT' else '?')))
    def mb(c):
        return {'k': 'k~', 'K': 'k~', 't': 't~', 'T': 't~', 'p': 'p~', 'P': 'p~', 'f': 'f~', 'F': 'f~'}[c]
    return {'KT': (set('k'), set('t'), m), 'PF': (set('p'), set('f'), m), 'CS': (set('C'), set('S'), m),
            'BENCH': (set('ktpf'), set('KTPF'), mb), 'BKT': (set('K'), set('T'), m)}


_TA = {}


def build_extra(pages, toks, tall, y):
    key = (id(pages), len(toks))
    if key not in _TA:
        _TA[key] = (above_any(pages, toks, tall, -1, 1), above_any(pages, toks, tall, -3, 3))
    e = dict(tall_above=_TA[key][0], tall_above2=_TA[key][1])
    pl, pp = pen_feats(toks, y)
    e['pen_line'], e['pen_prev'] = pl, pp
    return e


def same_above(pages, toks, pairs):
    out = np.zeros(len(toks), dtype=np.int8)
    for i, t in enumerate(toks):
        A, B, _ = pairs[t['pair']]
        if t['line'] == 0:
            continue
        g = pages[t['page']]['lines'][t['line'] - 1]['g']
        for c in g[max(0, t['off'] - 1): t['off'] + 2]:
            if c in B:
                out[i] = 1; break
            if c in A:
                out[i] = -1; break
    return out


def family_gains(toks, y, extra, groups, fams=FAMS, only=None):
    mats = {f: onehot(family_cols(toks, f, extra)) for f in fams}
    if only is not None:  # null draws: just the unique LAYOUT / PEN gains and LAYOUT-alone
        full = hstack([mats[f] for f in fams]).tocsr()
        pf, _ = oof_logloss(full, y, groups); Lf = bits(y, pf); out = {}
        for f in only:
            X = hstack([mats[g] for g in fams if g != f]).tocsr()
            p, _ = oof_logloss(X, y, groups); out['u_' + f] = bits(y, p) - Lf
        pS, _ = oof_logloss(mats['SCRIBE'], y, groups)
        p, _ = oof_logloss(hstack([mats['SCRIBE'], mats['LAYOUT']]).tocsr(), y, groups)
        out['a_LAYOUT'] = bits(y, pS) - bits(y, p)
        return out
    full = hstack([mats[f] for f in fams]).tocsr()
    pf, fold = oof_logloss(full, y, groups)
    Lfull = bits(y, pf)
    out = dict(H0=bits(y, np.full(len(y), np.clip(y.mean(), 1e-4, 1 - 1e-4))), full=Lfull)
    for f in fams:
        rest = [g for g in fams if g != f]
        X = hstack([mats[g] for g in rest]).tocsr()
        p, _ = oof_logloss(X, y, groups)
        out['u_' + f] = bits(y, p) - Lfull
    for f in ['CTX', 'FRAME', 'LAYOUT', 'PEN']:
        X = hstack([mats['SCRIBE'], mats[f]]).tocsr()
        p, _ = oof_logloss(X, y, groups)
        pS, _ = oof_logloss(mats['SCRIBE'], y, groups)
        out['a_' + f] = bits(y, pS) - bits(y, p)
    return out


def perm_within(toks, y, rng, keys=('frame', 'hand')):
    strata = defaultdict(list)
    for i, t in enumerate(toks):
        strata[tuple(t[k] for k in keys)].append(i)
    y2 = y.copy()
    for idx in strata.values():
        idx = np.array(idx)
        y2[idx] = y[rng.permutation(idx)]
    return y2


def run_corpus(name, pages, pairs, tall, nperm=10, rng=None, plant=None, log=None):
    toks_all = extract(pages, pairs, tall)
    res = {}
    for pname in pairs:
        toks = [t for t in toks_all if t['pair'] == pname]
        if len(toks) < 200:
            continue
        y = np.array([t['y'] for t in toks])
        if y.mean() < 0.01 or y.mean() > 0.99:
            continue
        groups = np.array([t['page'] for t in toks])
        sa = same_above(pages, toks, pairs)
        def ex(yy):
            e = build_extra(pages, toks, tall, yy); e['same_above'] = sa; return e
        t0 = time.time()
        r = family_gains(toks, y, ex(y), groups)
        r['n'] = len(y); r['rateB'] = float(y.mean())
        nl = defaultdict(list)
        for k in range(nperm):
            y2 = perm_within(toks, y, rng)
            r2 = family_gains(toks, y2, ex(y2), groups, only=('LAYOUT', 'PEN'))
            for key in ('u_LAYOUT', 'u_PEN', 'a_LAYOUT'):
                nl[key].append(r2[key])
        for key, v in nl.items():
            v = np.array(v)
            r['null_' + key] = (float(v.mean()), float(v.std()))
            r['z_' + key] = float((r[key] - v.mean()) / (v.std() + 1e-9))
        res[pname] = r
        msg = '%s %s n=%d B=%.3f H0=%.4f full=%.4f | uniq CTX %.4f FRAME %.4f LAYOUT %.4f (z %.1f) PEN %.4f (z %.1f) SCRIBE %.4f | alone CTX %.4f FRAME %.4f LAYOUT %.4f (z %.1f) PEN %.4f  [%.0fs]' % (
            name, pname, r['n'], r['rateB'], r['H0'], r['full'], r['u_CTX'], r['u_FRAME'], r['u_LAYOUT'], r['z_u_LAYOUT'],
            r['u_PEN'], r['z_u_PEN'], r['u_SCRIBE'], r['a_CTX'], r['a_FRAME'], r['a_LAYOUT'], r['z_a_LAYOUT'], r['a_PEN'], time.time() - t0)
        print(msg, flush=True)
        if log:
            log.write(msg + '\n'); log.flush()
    return res


def planted(pages, pairs, tall, strength, rng, nperm=10, log=None):
    """resample k/t from the frame model, then impose: tall glyph directly above (+-1) -> t with prob strength."""
    pp = {'KT': pairs['KT']}
    toks = extract(pages, pp, tall)
    y = np.array([t['y'] for t in toks])
    groups = np.array([t['page'] for t in toks])
    X = hstack([onehot(family_cols(toks, f)) for f in ('SCRIBE', 'CTX', 'FRAME')]).tocsr()
    p0, _ = oof_logloss(X, y, groups)
    ta = above_any(pages, toks, tall, -1, 1)
    ys = (rng.random(len(y)) < p0).astype(int)
    force = (ta == 1) & (rng.random(len(y)) < strength)
    ys[force] = 1
    for i, t in enumerate(toks):
        t['y'] = int(ys[i])
    sa = same_above(pages, toks, pp)
    def ex(yy):
        e = build_extra(pages, toks, tall, yy); e['same_above'] = sa; return e
    r = family_gains(toks, ys, ex(ys), groups)
    nl = []
    for k in range(nperm):
        y2 = perm_within(toks, ys, rng)
        nl.append(family_gains(toks, y2, ex(y2), groups, only=('LAYOUT',))['u_LAYOUT'])
    nl = np.array(nl)
    r['z_u_LAYOUT'] = float((r['u_LAYOUT'] - nl.mean()) / (nl.std() + 1e-9))
    msg = 'PLANT tall-under-tall s=%.2f: uniq LAYOUT %.4f (z %.1f) FRAME %.4f CTX %.4f PEN %.4f; share tall-above %.3f' % (
        strength, r['u_LAYOUT'], r['z_u_LAYOUT'], r['u_FRAME'], r['u_CTX'], r['u_PEN'], ta.mean())
    print(msg, flush=True)
    if log:
        log.write(msg + '\n'); log.flush()
    return r


if __name__ == '__main__':
    rng = np.random.default_rng(40)
    log = open(os.path.join(CKPT, 'cycle1.log'), 'a')
    out = {}
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'voy'):
        for nm in ('ZL3b', 'IT2a'):
            out[nm] = run_corpus(nm, load_voynich(nm), vpairs(), V_TALL, rng=rng, log=log)
        json.dump(out, open(os.path.join(CKPT, 'cycle1_voy.json'), 'w'), indent=1)
    if which in ('all', 'ctl'):
        dtall = set('bdfhklſtßꝛ') | set('ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ')
        dp = {'SS': (set('s'), set('ſ'), lambda c: 'S'), 'RR': (set('r'), set('ꝛ'), lambda c: 'R')}
        for f in ('grimmelshausen_simplicissimus_1669', 'goethe_faust01_1808'):
            out[f] = run_corpus(f, load_dta(os.path.join(SCR, f + '.txt')), dp, dtall, rng=rng, log=log)
        cp = {'CG': (set('c'), set('g'), lambda c: 'C'), 'NM': (set('n'), set('m'), lambda c: 'N')}
        out['caesar'] = run_corpus('caesar', load_gutenberg('pg218.txt'), cp, set('bdfhklt'), rng=rng, log=log)
        json.dump(out, open(os.path.join(CKPT, 'cycle1_ctl.json'), 'w'), indent=1)
    if which in ('all', 'plant'):
        pages = load_voynich('ZL3b')
        out['plant'] = {s: planted(pages, vpairs(), V_TALL, s, rng, log=log) for s in (0.0, 0.1, 0.2, 0.4)}
        json.dump(out, open(os.path.join(CKPT, 'cycle1_plant.json'), 'w'), indent=1)
