"""pe21 cycle 3: the token hypothesis.

If lists were transcribed from piles of tokens, numerals should show token-like limits:
(a) exchange failures: a pile of 10+ unit tokens written as it lies (count above the
    normal ceiling of a denomination).  Ceiling per code = largest count reached by >= 0.5%
    of that code's groups.  Planted: PE entries where with probability q an N14+N01 pair is
    written unexchanged (one N14 -> ten N01); power curve and an upper bound on q.
(b) few denominations per count (share of entries with 1, 2, 3+ codes).
(c) a hand-counting limit: a drop in the count distribution after 4-5 (subitizing), shown
    as ratios P(k+1)/P(k); planted 'pile counted by eye' corpus caps at 5.
(d) clay link (same null as cycles 1-2): does the tablet's area predict how large its
    counts are, beyond its amount of text?  Spearman of residual log area (after log glyphs)
    with mean log count per entry; null re-deals texts among tablets of the same system.
Outgroup: proto-cuneiform (Uruk IV-III administrative, pe2_pc_corpus.json), also the
clay elasticity beta on its intact tablets with CDLI dims.
"""
import csv, json, os, sys, collections, math
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe21_common import pe_tablets, dom_sys, CK, DATA, SCRATCH, write_rows
from pe21_cycle1 import beta, boot_beta

rng = np.random.default_rng(213)


def groups_from(entries):
    """entries: list of numeral lists -> list of Counter(code->count)."""
    out = []
    for e in entries:
        d = collections.Counter()
        for n, c in e:
            d[c.replace('N1@', 'N01@')] += n
        if d:
            out.append(d)
    return out


def ceilings(G, share=0.005):
    by = collections.defaultdict(collections.Counter)
    for d in G:
        for c, n in d.items():
            by[c][n] += 1
    ceil = {}
    for c, cnt in by.items():
        tot = sum(cnt.values())
        if tot < 30:
            continue
        ceil[c] = max(k for k, v in cnt.items() if v / tot >= share)
    return ceil, by


def overflow(G, ceil):
    n = o = 0
    for d in G:
        for c, k in d.items():
            if c in ceil:
                n += 1
                o += k > ceil[c]
    return o, n


def plant_unexchanged(G, q):
    out = []
    for d in G:
        d = collections.Counter(d)
        if d.get('N14', 0) >= 1 and d.get('N01', 0) >= 1 and rng.random() < q:
            d['N14'] -= 1
            d['N01'] += 10
            if d['N14'] == 0:
                del d['N14']
        out.append(d)
    return out


def ratios(by, code, kmax=9):
    c = by[code]
    return [round(c.get(k + 1, 0) / c[k], 2) if c.get(k) else None for k in range(1, kmax)]


def pc_load():
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    ent = []
    for t in T:
        for l in t['lines']:
            if l['signs'] and l['numerals'] and not l['lacuna']:
                ent.append(l['numerals'])
    return T, ent


def pc_dims(ids):
    csv.field_size_limit(10 ** 9)
    out = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        pid = 'P%06d' % int(row['id_text']) if row['id_text'].isdigit() else None
        if pid in ids:
            try:
                h, w = float(row['height']), float(row['width'])
                if h > 0 and w > 0:
                    out[pid] = (h, w, row['genre'])
            except Exception:
                pass
    return out


def main():
    res = {}
    P = pe_tablets()
    PG = groups_from([e for t in P for e in t['ent_nums']])
    PCT, pce = pc_load()
    CG = groups_from(pce)
    ceilP, byP = ceilings(PG)
    ceilC, byC = ceilings(CG)
    res['ceil_pe'] = ceilP
    res['ceil_pc'] = ceilC
    oP = overflow(PG, ceilP)
    oC = overflow(CG, ceilC)
    oPC = overflow(PG, {k: v for k, v in ceilC.items()})   # PE judged by PC ceilings
    res['overflow'] = {'pe': oP, 'pc': oC, 'pe_by_pc_ceil': oPC}
    # N01 overflow specifically (most relevant: unit tokens)
    n01 = lambda G: (sum(1 for d in G if d.get('N01', 0) > 9), sum(1 for d in G if d.get('N01', 0) > 0))
    res['n01_over9'] = {'pe': n01(PG), 'pc': n01(CG)}
    elig = sum(1 for d in PG if d.get('N14', 0) and d.get('N01', 0))
    res['eligible_pairs'] = elig
    pw = []
    for q in [0.005, 0.01, 0.02, 0.05, 0.1]:
        det = 0
        rates = []
        for rep in range(50):
            G2 = plant_unexchanged(PG, q)
            o, n = n01(G2)
            rates.append(o)
            det += o > 4 + 2 * math.sqrt(4) + 1   # beyond the real count's rough 97.5% band
        pw.append({'q': q, 'mean_over9': float(np.mean(rates)), 'power': det / 50})
    res['planted_unexchanged'] = pw
    # upper bound on q: real over9 = observed; Poisson 95% upper limit on events / eligible
    ob = res['n01_over9']['pe'][0]
    ul = {0: 3.0, 1: 4.74, 2: 6.3, 3: 7.75, 4: 9.15, 5: 10.51}.get(ob, ob + 2 * math.sqrt(ob) + 2)
    res['q_upper95'] = ul / elig
    # (b) denominations per count
    nd = lambda G: collections.Counter(min(len(d), 3) for d in G)
    res['ndenom'] = {'pe': nd(PG), 'pc': nd(CG)}
    # (c) count-distribution ratios
    res['ratios'] = {'pe_N01': ratios(byP, 'N01'), 'pc_N01': ratios(byC, 'N01'),
                     'pe_N14': ratios(byP, 'N14'), 'pc_N14': ratios(byC, 'N14')}
    # planted subitizing corpus: N01 counts > 5 are re-written as 5 + (k-5) split into a 2nd line (cap 5)
    capped = collections.Counter(min(d['N01'], 5) for d in PG if d.get('N01'))
    res['ratios']['planted_cap5_N01'] = [round(capped.get(k + 1, 0) / capped[k], 2) if capped.get(k) else None for k in range(1, 9)]
    # (d) clay link
    T = [t for t in P if t['intact'] and t['n_ent'] >= 2]
    A = np.log([t['h'] * t['w'] for t in T])
    Cn = np.log([max(t['glyphs'], 1) for t in T])
    sl, ic = np.polyfit(Cn, A, 1)
    resid = A - (ic + sl * Cn)

    def meancount(t):
        v = [sum(n for n, _ in e) for e in t['ent_nums']]
        return float(np.mean(np.log(v))) if v else 0.0
    mc = np.array([meancount(t) for t in T])
    rho = spearmanr(resid, mc).correlation
    sysl = np.array([dom_sys(t) for t in T])
    nulls = []
    for _ in range(1000):
        perm = np.arange(len(T))
        for s in set(sysl):
            ii = np.where(sysl == s)[0]
            perm[ii] = rng.permutation(ii)
        # re-deal texts (glyphs and counts together) across tablets of the same system
        Cp, mp = Cn[perm], mc[perm]
        r2 = A - np.polyval(np.polyfit(Cp, A, 1), Cp)
        nulls.append(spearmanr(r2, mp).correlation)
    nulls = np.array(nulls)
    res['clay_counts'] = {'n': len(T), 'rho': float(rho), 'null_mean': float(nulls.mean()),
                          'null_sd': float(nulls.std()), 'z': float((rho - nulls.mean()) / nulls.std())}
    # PC clay elasticity
    ids = {t['id'] for t in PCT}
    dims = pc_dims(ids)
    rowsC = []
    for t in PCT:
        if t['id'] not in dims or dims[t['id']][2] != 'Administrative':
            continue
        L = t['lines']
        if len(L) < 2 or any(l['lacuna'] for l in L):
            continue
        g = sum(len(l['signs']) + sum(n for n, _ in l['numerals']) for l in L)
        rowsC.append((math.log(dims[t['id']][0] * dims[t['id']][1]), math.log(max(len(L), 1)), math.log(max(g, 1))))
    R = np.array(rowsC)
    res['pc_beta'] = {'n': len(R), 'beta_lines': beta(R[:, 0], R[:, 1]), 'ci_lines': list(boot_beta(R[:, 0], R[:, 1])),
                      'beta_glyphs': beta(R[:, 0], R[:, 2]), 'ci_glyphs': list(boot_beta(R[:, 0], R[:, 2]))}
    m5 = R[:, 1] >= math.log(5)
    res['pc_beta']['beta_lines_min5'] = beta(R[m5, 0], R[m5, 1])
    res['pc_beta']['n_min5'] = int(m5.sum())
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=lambda o: dict(o))

    ov = res['overflow']
    rows = [
        ['PE-21.3a', 'TOKEN exchange failures: count above the per-code ceiling (largest count with >= 0.5% of groups); PE entries vs proto-cuneiform admin (PC); planted unexchanged N14 -> 10 N01 at rate q (50 reps each)',
         'ceilings PE %s; PC N01 %s, N14 %s; overflow PE %d/%d, PC %d/%d; N01 > 9: PE %d/%d, PC %d/%d; planted power %s; PE upper bound q < %.3f (95%%, %d eligible N14+N01 entries)' % (
             {k: v for k, v in ceilP.items() if k in ('N01', 'N14', 'N34', 'N39B', 'N30C', 'N24', 'N45')}, ceilC.get('N01'), ceilC.get('N14'),
             ov['pe'][0], ov['pe'][1], ov['pc'][0], ov['pc'][1], *res['n01_over9']['pe'], *res['n01_over9']['pc'],
             ', '.join('q %.3f: %.0f%% (mean %.1f)' % (p['q'], 100 * p['power'], p['mean_over9']) for p in pw), res['q_upper95'], elig), ''],
        ['PE-21.3b', 'Denominations per count (1 / 2 / 3+ codes) PE vs PC',
         'PE %s; PC %s' % (dict(sorted(res['ndenom']['pe'].items())), dict(sorted(res['ndenom']['pc'].items()))), ''],
        ['PE-21.3c', 'Hand-counting limit: ratios P(k+1)/P(k), k = 1..8, for N01 and N14; planted cap-at-5 corpus',
         'PE N01 %s; PC N01 %s; PE N14 %s; PC N14 %s; planted cap5 %s' % (res['ratios']['pe_N01'], res['ratios']['pc_N01'],
                                                                         res['ratios']['pe_N14'], res['ratios']['pc_N14'], res['ratios']['planted_cap5_N01']), ''],
        ['PE-21.3d', 'Clay vs count size: Spearman(residual log area after log glyphs, mean log count per entry); null re-deals texts across tablets of the same number system (1,000)',
         'rho %+.3f, null %+.3f +- %.3f, z %+.1f, n %d' % (res['clay_counts']['rho'], res['clay_counts']['null_mean'], res['clay_counts']['null_sd'], res['clay_counts']['z'], res['clay_counts']['n']), ''],
        ['PE-21.3e', 'Outgroup clay elasticity: proto-cuneiform admin tablets, no lacunae, CDLI dims',
         'beta lines %.2f [%.2f,%.2f], glyphs %.2f [%.2f,%.2f], n %d; >= 5 lines %.2f (n %d)' % (
             res['pc_beta']['beta_lines'], *res['pc_beta']['ci_lines'], res['pc_beta']['beta_glyphs'], *res['pc_beta']['ci_glyphs'],
             res['pc_beta']['n'], res['pc_beta']['beta_lines_min5'], res['pc_beta']['n_min5']), ''],
    ]
    write_rows(os.path.join(CK, 'c3_rows.txt'), rows)
    for r in rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
