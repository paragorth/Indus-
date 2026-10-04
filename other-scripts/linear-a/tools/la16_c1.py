#!/usr/bin/env python3
"""LA-16 cycle 1: hand-contrastive sign-pair search.
For every sign pair (X,Y): T = sum over one-sign-different word pairs of the share of token pairs written by
different hands. Null: hands permuted over documents within stratum (site+series for LB, site for LA), 2,000
permutations -> per-pair z, raw P and max-z FWER.
Positive control (LB, KN+PY, DAMOS hands): top cross-hand sign pairs should be phonetically related
(doublet / same consonant / same vowel) more than the base rate; most WITHIN-hand pairs should look like
inflection (same consonant, vowel differs). Planted control: in LA (HT Scribe 9) and LB-at-LA-size,
a sign X is rewritten as Y in one hand (rate 1.0 / 0.5) and must come back as the top pair.
"""
import sys, os, json, collections, random, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la16_common import *

NP = int(os.environ.get('NP', 2000))
say = say_to(os.path.join(OUT, 'c1_report.txt'))
res = {}


def strat_series(docs):
    for d in docs: d['ss'] = d['site'] + ':' + (d.get('support') or '')
    return docs


def analyse(tag, docs, field='hand', strat='site', top=12, seed=1):
    E = Engine(docs, field=field, strat=strat)
    real = E.stat(E.lab0); N = E.null(NP, seed)
    z, exc, rows, zmax = rank_report(E, real, N, top=top)
    ok = E.npairs_k >= 2
    say(f'\n== {tag}: docs {len(E.docs)} labels {len(E.labs)} types {len(E.T)} pairs {len(E.P)} sign-pairs {len(E.keys)} (n>=2: {ok.sum()})')
    say(f'   total cross weight real {real.sum():.1f} null {N.sum(1).mean():.1f} +- {N.sum(1).std():.1f}  P(<=) {(N.sum(1) <= real.sum()).mean():.4f}')
    for r in rows:
        say(f'   {r["key"][0]:>6}~{r["key"][1]:<6} n {r["n"]:3d} real {r["real"]:6.2f} null {r["null"]:6.2f} z {r["z"]:5.2f} P {r["P"]:.4f} FWER {r["FWER"]:.3f}')
    low = [k for k in np.argsort(z) if ok[k]][:8]
    say('   most WITHIN-hand: ' + ', '.join(f'{E.keys[k][0]}~{E.keys[k][1]}(n{E.npairs_k[k]},z{z[k]:.1f})' for k in low))
    return E, real, N, z, rows


def lb_classes(E, z, tag, K=(10, 20, 40)):
    ok = [k for k in range(len(E.keys)) if E.npairs_k[k] >= 2]
    base = collections.Counter(lb_class(*E.keys[k]) for k in ok)
    tot = sum(base.values())
    say(f'   {tag} base rate (n>=2 pairs, {tot}): ' + ', '.join(f'{c} {base[c]/tot:.3f}' for c in ('doublet', 'sameC', 'sameV', 'other')))
    out = {}
    hi = sorted(ok, key=lambda k: -z[k]); lo = sorted(ok, key=lambda k: z[k])
    for kk in K:
        for nm, lst in (('cross', hi[:kk]), ('within', lo[:kk])):
            c = collections.Counter(lb_class(*E.keys[k]) for k in lst)
            out[f'{nm}{kk}'] = {x: c[x] for x in ('doublet', 'sameC', 'sameV', 'other')}
            say(f'   {tag} top{kk} {nm}: ' + ', '.join(f'{x} {c[x]}' for x in ('doublet', 'sameC', 'sameV', 'other')))
    # rank test: does z separate related (doublet+sameV) from other?
    rel = np.array([lb_class(*E.keys[k]) in ('doublet', 'sameV') for k in ok]); zz = np.array([z[k] for k in ok])
    rnd = np.random.default_rng(5); d0 = zz[rel].mean() - zz[~rel].mean()
    perm = [(lambda r: zz[r].mean() - zz[~r].mean())(rnd.permutation(rel)) for _ in range(5000)]
    p = float((np.array(perm) >= d0).mean())
    say(f'   {tag} mean z (doublet+sameV) - (others) = {d0:.3f}, label-perm P {p:.4f}')
    dc = np.array([lb_class(*E.keys[k]) == 'sameC' for k in ok]); d1 = zz[dc].mean() - zz[~dc].mean()
    perm = [(lambda r: zz[r].mean() - zz[~r].mean())(rnd.permutation(dc)) for _ in range(5000)]
    say(f'   {tag} mean z (sameC) - (others) = {d1:.3f}, P(<=) {float((np.array(perm) <= d1).mean()):.4f}')
    out['d_rel'] = d0; out['p_rel'] = p
    return out


def plant(docs, hand, x, y, rate, rnd):
    D = copy.deepcopy(docs)
    for d in D:
        if d.get('hand') != hand: continue
        nw = []
        for w in d['words']:
            if x in w and rnd.random() < rate: w = tuple(y if s == x else s for s in w)
            nw.append(w)
        d['words'] = nw
    return D


def planted_runs(tag, docs, hand, strat, rnd, ndraw=3, nperm=500):
    cnt = collections.Counter(s for d in docs if d.get('hand') == hand for w in d['words'] for s in set(w))
    allc = collections.Counter(s for d in docs if d.get('hand') for w in d['words'] for s in w)
    cand = [s for s, c in cnt.items() if c >= 3]
    rows = []
    for rate in (1.0, 0.5):
        for _ in range(ndraw):
            x = rnd.choice(cand)
            ys = [s for s, c in allc.items() if s != x and 0.5 * allc[x] <= c <= 2 * allc[x] + 3] or [s for s in allc if s != x]
            y = rnd.choice(ys)
            D = plant(docs, hand, x, y, rate, rnd)
            E = Engine(D, strat=strat); real = E.stat(E.lab0); N = E.null(nperm, rnd.randrange(10 ** 6))
            z, exc, rr, zmax = rank_report(E, real, N, top=len(E.keys), minpairs=1)
            key = tuple(sorted((x, y)))
            rk = next((i + 1 for i, r in enumerate(rr) if r['key'] == key), None)
            fw = next((r['FWER'] for r in rr if r['key'] == key), None)
            n = int(E.npairs_k[E.keys.index(key)]) if key in E.keys else 0
            rows.append((rate, x, y, n, rk, fw))
            say(f'   {tag} plant {x}->{y} rate {rate}: pairs {n} rank {rk} FWER {fw}')
    ok = sum(1 for r in rows if r[4] == 1 and r[5] is not None and r[5] < 0.05)
    top5 = sum(1 for r in rows if r[4] and r[4] <= 5)
    say(f'   {tag} planted recovered (rank 1, FWER<0.05) {ok}/{len(rows)}; in top 5: {top5}/{len(rows)}')
    return rows, ok, top5


def main():
    rnd = random.Random(16)
    # ---- Linear B positive control, full
    lb = strat_series(lb_corpus(('KN', 'PY')))
    E, real, N, z, rows = analyse('LB KN+PY hands (strata site+series)', lb, strat='ss')
    res['LB_full'] = dict(rows=rows, cls=lb_classes(E, z, 'LB'))
    E2, r2, N2, z2, rows2 = analyse('LB KN+PY hands (strata site only)', lb, strat='site')
    res['LB_site'] = dict(rows=rows2, cls=lb_classes(E2, z2, 'LBsite'))
    # ---- Linear A
    la = la_corpus()
    E, real, N, z, rows = analyse('LA scribal hands (strata site)', la, strat='site')
    res['LA_hand'] = dict(rows=rows)
    for d in la:
        d['grp'] = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}.get(d['site'], 'OTH')
    E, real, N, z, rows = analyse('LA site groups, all docs (strata support)', la, field='grp', strat='support')
    res['LA_site'] = dict(rows=rows)
    # ---- planted controls
    say('\n== Planted controls')
    res['plant_LA'] = planted_runs('LA HT9', la, 'HT Scribe 9', 'site', rnd)
    # LB at LA size: random documents of KN+PY with hands until ~612 attributed word tokens
    att = [d for d in lb if d['hand']]
    rnd.shuffle(att); sub = []; n = 0
    for d in att:
        if n >= 612: break
        sub.append(d); n += len(d['words'])
    big = collections.Counter(d['hand'] for d in sub for w in d['words']).most_common(1)[0][0]
    E, real, N, z, rows = analyse(f'LB at LA size ({len(sub)} docs, {n} tokens), unplanted', sub, strat='ss', top=5)
    res['LB_small'] = dict(rows=rows, cls=lb_classes(E, z, 'LBsmall', K=(10,)))
    res['plant_LBsmall'] = planted_runs(f'LBsmall {big}', sub, big, 'ss', rnd)
    pyhand = 'PY:1'
    res['plant_LBfull'] = planted_runs('LBfull PY:1', lb, pyhand, 'ss', rnd, ndraw=2, nperm=300)
    json.dump(res, open(os.path.join(OUT, 'c1.json'), 'w'), default=str, indent=1)


if __name__ == '__main__':
    main()
