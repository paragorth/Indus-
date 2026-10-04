"""v15 cycle 3: stranger length channels and the leftovers of cycle 1.

3a  Position-preserving nulls for the two cycle-1 leftovers: words (or gap
    types) swapped only between lines of the same page at the same index in
    the line. If the period-8 peak and the ',' clustering survive this null,
    they are not position-in-line effects.
3b  One letter per LINE: line totals (glyph units, EVA chars, strokes, words)
    mod k (k = 20..30) as cipher symbols; random codebooks + annealing for 7
    languages; best over k; nulls = words shuffled within page (line totals
    change, glyph stats kept) and line order shuffled within page. Positive
    control: Caesar planted in line glyph totals mod 23 (last word of each
    line replaced by a Voynich word of the length that sets the total).
    Negative control: Caesar / Manzoni typeset lines.
3c  Delta channel: (L_t - L_t-1) mod k for k = 6, 8, 10 with words in reading
    order, same search; nulls = within-line shuffle.
"""
import os, sys, json, zlib
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V
import v15_cycle1 as C1

OUT = os.path.join(V.RES, 'cycle3')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 3a
def pos_shuffle(lines, rng, what='words'):
    pages = defaultdict(list)
    for i, L in enumerate(lines):
        pages[L['folio']].append(i)
    out = [dict(L) for L in lines]
    for f, idx in pages.items():
        slots = defaultdict(list)
        key = 'words' if what == 'words' else 'gaps'
        for i in idx:
            for j, _ in enumerate(lines[i][key]):
                slots[j].append(i)
        for k in out:
            pass
        for j, li in slots.items():
            vals = [lines[i][key][j] for i in li]
            rng.shuffle(vals)
            for i, v in zip(li, vals):
                if out[i][key] is lines[i][key]:
                    out[i][key] = list(lines[i][key])
                out[i][key][j] = v
    return out


def job3a(_):
    path = os.path.join(OUT, '3a.json')
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(31)
    lines = V.load_v('ZL3b')
    res = {}
    for unit, fn in (('g', V.glen), ('c', V.clen)):
        _, rs = C1.stats(lines, fn)
        specs = [C1.stats(pos_shuffle(lines, rng), fn)[1] for _ in range(30)]
        specs = np.array(specs); mu = specs.mean(0)
        null_max = [float((sp / np.mean(np.delete(specs, i, 0), 0)).max()) for i, sp in enumerate(specs)]
        ratio = rs / mu
        per = 1 / np.linspace(1 / 64, 0.5, 121)[:120]
        b8 = int(np.argmin(np.abs(per - 8)))
        res['spec_' + unit] = {'real_max': float(ratio.max()), 'period': float(per[int(np.argmax(ratio))]),
                               'ratio_at_8': float(ratio[b8]), 'null_max_mean': float(np.mean(null_max)),
                               'null_max_max': float(np.max(null_max)),
                               'p': float((1 + sum(m >= ratio.max() for m in null_max)) / 31)}
        # neighbour MI with position-preserving null
        seqs = C1.line_codes(lines, lambda w: min(fn(w), 10))
        real = C1.mi_in(seqs)
        nul = [C1.mi_in(C1.line_codes(pos_shuffle(lines, rng), lambda w: min(fn(w), 10))) for _ in range(30)]
        res['mi_in_' + unit] = {'real': real, 'null': float(np.mean(nul)), 'sd': float(np.std(nul)),
                                'z': float((real - np.mean(nul)) / np.std(nul))}
    # gaps: position profile and position-preserving null
    def gmi(ls):
        seqs = [np.array([1 if g == ',' else 0 for g in L['gaps'] if g != '-']) for L in ls]
        seqs = [s for s in seqs if len(s) > 1]
        x = np.concatenate([s[:-1] for s in seqs]); y = np.concatenate([s[1:] for s in seqs])
        return C1.mi_pairs(x, y, 2)
    real = gmi(lines)
    nul = [gmi(pos_shuffle(lines, rng, 'gaps')) for _ in range(200)]
    prof = defaultdict(lambda: [0, 0])
    for L in lines:
        for j, g in enumerate(L['gaps']):
            if g == '-':
                continue
            prof[min(j, 12)][0] += g == ','; prof[min(j, 12)][1] += 1
    # line-level clustering: lines with >= 2 commas vs binomial expectation from line rates
    res['gap_pos'] = {'real': real, 'null': float(np.mean(nul)), 'sd': float(np.std(nul)),
                      'z': float((real - np.mean(nul)) / np.std(nul)),
                      'profile': {j: [a, n, a / n] for j, (a, n) in sorted(prof.items())}}
    # page-level clustering: comma rate per page, dispersion vs binomial
    pg = defaultdict(lambda: [0, 0])
    for L in lines:
        for g in L['gaps']:
            if g != '-':
                pg[L['folio']][0] += g == ','; pg[L['folio']][1] += 1
    a = np.array([v[0] for v in pg.values()]); n = np.array([v[1] for v in pg.values()])
    p = a.sum() / n.sum()
    chi = float((((a - n * p) ** 2) / (n * p * (1 - p))).sum())
    res['gap_page_dispersion'] = {'chi2': chi, 'df': int(len(a) - 1), 'top_pages': sorted(
        [(f, v[0], v[1]) for f, v in pg.items()], key=lambda t: -t[1] / max(t[2], 1))[:8]}
    json.dump(res, open(path, 'w'), indent=1)
    return path


# ---------------------------------------------------------------- 3b
def line_totals(lines, unit):
    if unit == 'w':
        return np.array([len(L['words']) for L in lines])
    fn = {'g': V.glen, 'c': V.clen, 's': V.slen}[unit]
    return np.array([sum(fn(w) for w in L['words']) for L in lines])


def plant_lines_total(lines, msg, k, rng):
    toks = [w for L in lines for w in L['words']]
    by_len = defaultdict(list)
    for w in toks:
        by_len[V.glen(w)].append(w)
    alpha = sorted(set(msg)); li = {c: i for i, c in enumerate(alpha)}
    out = []
    for i, L in enumerate(lines):
        target = li[msg[i % len(msg)]] % k
        ws = list(L['words'])
        lens = [l for l in by_len if l <= 12]
        if len(ws) >= 2:
            rest = sum(V.glen(w) for w in ws[:-2])
            g1, g2 = V.glen(ws[-2]), V.glen(ws[-1])
            best = min(((a, b) for a in lens for b in lens if (rest + a + b) % k == target),
                       key=lambda ab: abs(ab[0] - g1) + abs(ab[1] - g2))
            ws[-2] = by_len[best[0]][rng.integers(len(by_len[best[0]]))]
            ws[-1] = by_len[best[1]][rng.integers(len(by_len[best[1]]))]
        else:
            opts = [l for l in by_len if l % k == target]
            if opts:
                l = min(opts); ws[-1] = by_len[l][rng.integers(len(by_len[l]))]
        d = dict(L); d['words'] = ws; out.append(d)
    return out


def shuffle_line_order(lines, rng):
    pages = defaultdict(list)
    for i, L in enumerate(lines):
        pages[L['folio']].append(i)
    out = list(lines)
    for f, idx in pages.items():
        p = list(idx); rng.shuffle(p)
        for i, j in zip(idx, p):
            out[i] = lines[j]
    return out


def job3b(args):
    src, unit, lang = args
    path = os.path.join(OUT, f'3b_{src}_{unit}_{lang}.json')
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    if src == 'ZL3b':
        lines = V.load_v('ZL3b')
    elif src == 'plant_total':
        lines = plant_lines_total(V.load_v('ZL3b'), V.caesar_letters(), 23, np.random.default_rng(5))
    else:
        lines = V.ref_lines({'Caesar': 'Latin-Caesar', 'Manzoni': 'Italian-Manzoni'}[src])
        unit = 'c' if unit == 'g' else unit
    lm = V.get_lm(lang)
    ks = list(range(20, 31)) if unit != 'w' else [12]
    res = {'src': src, 'unit': unit, 'lang': lang, 'runs': []}
    kinds = [('real', 1), ('page', 5), ('order', 3)]
    for kind, n in kinds:
        for r in range(n):
            if kind == 'real':
                ls = lines
            elif kind == 'page':
                ls = V.shuffle_lines(lines, 'page', rng)
            else:
                ls = shuffle_line_order(lines, rng)
            tot = line_totals(ls, unit)
            best = None
            for k in ks:
                sym = (tot % k) if unit != 'w' else np.clip(tot, 1, 12) - 1
                o = V.solve2(lm, sym, k, rng, n_random=1000, n_starts=2, sweeps=10)
                o['k'] = k
                if best is None or o['test'] > best['test']:
                    best = o
            best['kind'] = kind
            if kind != 'real':
                best.pop('key')
            res['runs'].append(best)
    json.dump(res, open(path, 'w'))
    return path


# ---------------------------------------------------------------- 3c
def job3c(args):
    src, k, lang = args
    path = os.path.join(OUT, f'3c_{src}_{k}_{lang}.json')
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    if src == 'ZL3b':
        lines, fn = V.load_v('ZL3b'), V.glen
    elif src == 'plant_delta':
        lines, fn = plant_delta(k), V.glen
    else:
        lines, fn = V.ref_lines({'Caesar': 'Latin-Caesar', 'Manzoni': 'Italian-Manzoni'}[src]), V.clen
    lm = V.get_lm(lang)
    res = {'src': src, 'unit': 'delta%d' % k, 'scheme': 'delta%d' % k, 'lang': lang, 'runs': []}
    for kind, n in (('real', 1), ('line', 6)):
        for r in range(n):
            ls = lines if kind == 'real' else V.shuffle_lines(lines, 'line', rng)
            L = np.array(V.length_stream(ls, fn))
            sym = np.diff(L) % k
            o = V.solve2(lm, sym, k, rng)
            o['kind'] = kind; o['n_sym'] = int(len(sym))
            if kind != 'real':
                o.pop('key'); o.pop('sample')
            res['runs'].append(o)
    json.dump(res, open(path, 'w'))
    return path


def plant_delta(k):
    """Caesar in (L_t - L_t-1) mod k: each letter -> one residue (many-to-one if
    k < 21), next length = previous + residue (mod k) brought into 1..10."""
    rng = np.random.default_rng(9)
    lines = V.load_v('ZL3b')
    msg = V.caesar_letters()
    alpha = sorted(set(msg)); li = {c: i % k for i, c in enumerate(alpha)}
    toks = [w for L in lines for w in L['words']]
    by_len = defaultdict(list)
    for w in toks:
        by_len[V.glen(w)].append(w)
    prev, j, out = 5, 0, []
    for L in lines:
        ws = []
        for _ in L['words']:
            r = li[msg[j % len(msg)]]; j += 1
            cands = [l for l in range(1, 13) if (l - prev) % k == r and by_len.get(l)]
            l = min(cands, key=lambda x: abs(x - 5))
            ws.append(by_len[l][rng.integers(len(by_len[l]))]); prev = l
        d = dict(L); d['words'] = ws; out.append(d)
    return out


def run(j):
    kind = j[0]
    return {'3a': job3a, '3b': job3b, '3c': job3c}[kind](j[1])


if __name__ == '__main__':
    J = [('3a', None)]
    for lang in V.LANGS:
        for src in ('plant_total', 'ZL3b', 'Caesar', 'Manzoni'):
            for unit in ('g', 'w') if src == 'ZL3b' else ('g',):
                J.append(('3b', (src, unit, lang)))
        for k in (6, 8, 10):
            for src in ('plant_delta', 'ZL3b', 'Caesar', 'Manzoni'):
                J.append(('3c', (src, k, lang)))
    J.sort(key=lambda j: (j[0] != '3a', j[1] is not None and j[1][2] != 'la'))
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(run, J)):
            print(i, len(J), os.path.basename(r), flush=True)
