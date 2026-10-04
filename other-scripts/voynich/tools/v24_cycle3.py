"""v24 cycle 3: image route.  Uses data/v24_ckpt/c3_ink.pkl from v24_ink.py + v24_ink_test.py.
(a) Word-level: are ink-flagged words (top 2% local anomaly) richer in transcriber disagreements, with the
    flag permuted within page x glyph-count strata (longer words have more segments and more chances)?
(b) Rule-position test (only meaningful if (a) or the ZL-note validation passes): at the flagged glyph,
    constraint C_R = R(word) - E[R(random edit at that glyph)]; compared with the same statistic at a random
    glyph of the same word (within-word permutation) and at the anomaly glyph of unflagged words with the
    same glyph count (between-word permutation), 2,000 permutations, named core rules.
"""
import os, sys, json, pickle, math, random
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L
import v24_data as D
from v24_cycle1 import load

CORE = ['vocab', 'logf', 'big1', 'bilp', 'q3lp', 'slot', 'junc', 'lineinit', 'firstlp', 'lastlp']


def main():
    I = pickle.load(open(os.path.join(L.CK, 'c3_ink.pkl'), 'rb'))
    real, thr, alt, itd = I['real'], I['thr'], I['alt'], I['itd']
    rng = np.random.default_rng(0)
    flag = np.array([r['zmax'] >= thr for r in real])
    dis = np.array([((r['folio'], r['n'], r['k']) in alt) or ((r['folio'], r['n'], r['k']) in itd) for r in real])
    strata = defaultdict(list)
    for i, r in enumerate(real):
        strata[(r['folio'], min(len(r['glyphs']), 9))].append(i)
    obs = dis[flag].mean()
    null = []
    for _ in range(5000):
        f2 = flag.copy()
        for idx in strata.values():
            idx = np.array(idx); f2[idx] = rng.permutation(flag[idx])
        null.append(dis[f2].mean())
    null = np.array(null)
    out = dict(n_flag=int(flag.sum()), dis_flag=float(obs), dis_null=float(null.mean()), dis_sd=float(null.std()),
               p=float(((null >= obs).sum() + 1) / 5001))
    print('(a) disagreement rate in flagged words %.3f vs stratified null %.3f +- %.3f, p %.4f (n flagged %d)' % (
        obs, null.mean(), null.std(), out['p'], flag.sum()), flush=True)
    # (b) rule position
    V = load('voy.pkl')
    sm = json.load(open(os.path.join(L.DATA, 'derived', 'v7_slotmodels.json')))['voynich_ZL3b_K4']
    M = L.Model(V['corpus'], sm['order'], sm['cuts'])
    RS = L.RuleSet(M, n_random=0, slot=True)
    idx = [RS.names.index(c) for c in CORE]
    zl = {(r['folio'], r['n']): [L.U(D.reading(t, 0)) for t in D.zl_variants(r['raw'])] for r in D.zl_lines()}
    prevfirst = {}
    pf = None
    for r in D.zl_lines():
        prevfirst[(r['folio'], r['n'])] = pf
        if r['ltype'] == 'P':
            ws = zl[(r['folio'], r['n'])]
            if ws and ws[0]:
                pf = ws[0][0]
    rows = []   # per word: C matrix (positions x rules), kmax, flag
    sample = [i for i in range(len(real)) if flag[i]] + list(rng.choice(np.nonzero(~flag)[0], size=1500, replace=False))
    for i in sample:
        r = real[i]
        ws = zl.get((r['folio'], r['n']))
        if not ws or r['k'] >= len(ws):
            continue
        w = ws[r['k']]
        if len(w) != len(r['glyphs']) or not w:
            continue
        k = r['k']
        site = L.Site(w, w, prev=ws[k - 1] if k > 0 and ws[k - 1] else None, nxt=ws[k + 1] if k + 1 < len(ws) and ws[k + 1] else None,
                      prev2=ws[k - 2] if k > 1 and ws[k - 2] else None, li=(k == 0), pl=prevfirst.get((r['folio'], r['n'])))
        va = RS.score(w, site)[idx]
        C = []
        for j in range(len(w)):
            site.kind, site.pos = 'sub', j
            pool, pw = L.null_pool(site, M)
            pm = np.stack([RS.score(p, site)[idx] for p in pool])
            C.append(va - (pm * pw[:, None]).sum(0))
        rows.append(dict(C=np.array(C), k=r['kmax'], flag=bool(flag[i]), g=len(w)))
    F = [x for x in rows if x['flag']]; U = [x for x in rows if not x['flag']]
    obs = np.nanmean([x['C'][x['k']] for x in F], 0)
    within = []
    for _ in range(2000):
        within.append(np.nanmean([x['C'][rng.integers(x['g'])] for x in F], 0))
    within = np.array(within)
    byg = defaultdict(list)
    for x in U:
        byg[x['g']].append(x)
    between = []
    for _ in range(2000):
        vals = []
        for x in F:
            pool = byg.get(x['g']) or U
            y = pool[rng.integers(len(pool))]
            vals.append(y['C'][y['k']])
        between.append(np.nanmean(vals, 0))
    between = np.array(between)
    res = []
    for c, o_, wm, ws_, bm, bs in zip(CORE, obs, np.nanmean(within, 0), np.nanstd(within, 0), np.nanmean(between, 0), np.nanstd(between, 0)):
        res.append((c, float(o_), float((o_ - wm) / (ws_ + 1e-9)), float((o_ - bm) / (bs + 1e-9))))
    out['rule_pos'] = res; out['n_flag_words_used'] = len(F); out['n_ctrl_words'] = len(U)
    # family-wise: max |z| over the 10 rules for within-word permutations
    zw = (within - np.nanmean(within, 0)) / (np.nanstd(within, 0) + 1e-9)
    mx = np.nanmax(np.abs(zw), 1)
    out['fw_within'] = float(((mx >= max(abs(r[2]) for r in res)).sum() + 1) / (len(mx) + 1))
    zb = (between - np.nanmean(between, 0)) / (np.nanstd(between, 0) + 1e-9)
    mxb = np.nanmax(np.abs(zb), 1)
    out['fw_between'] = float(((mxb >= max(abs(r[3]) for r in res)).sum() + 1) / (len(mxb) + 1))
    print('(b) flagged glyphs used %d, control words %d' % (len(F), len(U)))
    for r in res:
        print('   %-8s C %.3f  z(within-word) %+.2f  z(between-word) %+.2f' % r)
    print('   family-wise p within %.3f, between %.3f' % (out['fw_within'], out['fw_between']))
    pos = Counter('first' if x['k'] == 0 else 'last' if x['k'] == x['g'] - 1 else 'mid' for x in F)
    exp = Counter();
    for x in F:
        exp['first'] += 1 / x['g']; exp['last'] += 1 / x['g']; exp['mid'] += (x['g'] - 2) / x['g'] if x['g'] > 2 else 0
    out['pos'] = dict(pos); out['pos_exp'] = {k: round(v, 1) for k, v in exp.items()}
    print('   flagged glyph position', dict(pos), 'uniform expectation', out['pos_exp'])
    json.dump(out, open(os.path.join(L.CK, 'c3_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
