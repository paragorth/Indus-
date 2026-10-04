"""v15 cycle 2: massive random codebooks + annealing on length channels.

Channels: word length in glyph units (g), EVA characters (c), pen strokes (s,
rank-binned to 1..10 for 'len'/'pair'); ZL3b gap type ('.' vs ','); words per line.
Schemes (v15_lib.scheme_stream): len, pair, bacon2, baconT, tri3, mod4pair.
For each (stream, scheme, language): 2,000 random codebooks scored, the best 3
annealed (heat-bath Gibbs, 12 sweeps) on the first half; the key is scored on
the held-out second half (mean log2 P(letter | 2 previous) under the language
trigram model). Same search on nulls: words shuffled within line (8x) and the
whole length stream shuffled (4x). Positive controls: Caesar planted in the
lengths of a Voynich-like text (len / pair / bacon2). Negative controls: Caesar
and Manzoni's own word lengths (+ their within-line shuffles).
Checkpoints: one JSON per (stream, scheme, language) under results/v15/cycle2/.
"""
import os, sys, json, zlib
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V

OUT = os.path.join(V.RES, 'cycle2')
os.makedirs(OUT, exist_ok=True)
_CACHE = {}


def base_lines(src):
    if src not in _CACHE:
        if src == 'ZL3b':
            _CACHE[src] = V.load_v('ZL3b')
        elif src.startswith('plant_'):
            _CACHE[src] = V.plant(V.load_v('ZL3b'), V.caesar_letters(), src[6:], np.random.default_rng(7))
        elif src == 'Caesar':
            _CACHE[src] = V.ref_lines('Latin-Caesar')
        elif src == 'Manzoni':
            _CACHE[src] = V.ref_lines('Italian-Manzoni')
    return _CACHE[src]


def rank_bin(vals, k=10):
    v = np.asarray(vals)
    qs = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(qs, v, side='right') + 1


def raw_stream(lines, unit):
    fn = {'g': V.glen, 'c': V.clen, 's': V.slen}[unit]
    return np.array(V.length_stream(lines, fn))


def symbols(lines, unit, scheme):
    if unit == 'gap':
        bits = np.array([1 if g == ',' else 0 for L in lines for g in L['gaps'] if g != '-'])
        bits = bits[:len(bits) // 5 * 5].reshape(-1, 5)
        return bits @ np.array([16, 8, 4, 2, 1]), 32
    if unit == 'wpl':
        x = np.clip([len(L['words']) for L in lines], 1, 12) - 1
        return x, 12
    L = raw_stream(lines, unit)
    if unit == 's' and scheme in ('len', 'pair'):
        L = rank_bin(L)
    return V.scheme_stream(L, scheme)


def null_lines(lines, unit, kind, rng):
    if kind == 'real':
        return lines
    if unit == 'gap':
        out = []
        for L in lines:
            g = L['gaps'][:]; rng.shuffle(g); d = dict(L); d['gaps'] = g; out.append(d)
        return out
    if unit == 'wpl':
        idx = rng.permutation(len(lines)) if kind == 'glob' else None
        if idx is not None:
            return [lines[i] for i in idx]
        # within page
        from collections import defaultdict
        pages = defaultdict(list)
        for i, L in enumerate(lines):
            pages[L['folio']].append(i)
        out = list(lines)
        for f, ix in pages.items():
            p = list(ix); rng.shuffle(p)
            for i, j in zip(ix, p):
                out[i] = lines[j]
        return out
    if kind == 'line':
        return V.shuffle_lines(lines, 'line', rng)
    if kind == 'glob':
        words = [w for L in lines for w in L['words']]
        rng.shuffle(words)
        out, k = [], 0
        for L in lines:
            d = dict(L); d['words'] = words[k:k + len(L['words'])]; k += len(L['words']); out.append(d)
        return out
    raise ValueError(kind)


def run(job):
    src, unit, scheme, lang, n_line, n_glob = job
    path = os.path.join(OUT, f'{src}_{unit}_{scheme}_{lang}.json')
    if os.path.exists(path):
        return path
    seed = zlib.crc32(os.path.basename(path).encode())
    rng = np.random.default_rng(seed)
    lm = V.get_lm(lang)
    lines = base_lines(src)
    res = {'src': src, 'unit': unit, 'scheme': scheme, 'lang': lang, 'runs': []}
    for kind, n in (('real', 1), ('line', n_line), ('glob', n_glob)):
        for r in range(n):
            s, S = symbols(null_lines(lines, unit, kind, rng), unit, scheme)
            o = V.solve2(lm, s, S, rng)
            o['kind'] = kind; o['n_sym'] = int(len(s))
            if kind != 'real':
                o.pop('key'); o.pop('sample')
            res['runs'].append(o)
    json.dump(res, open(path, 'w'))
    return path


def jobs():
    J = []
    for lang in V.LANGS:
        for scheme in V.SCHEMES:
            J.append(('ZL3b', 'g', scheme, lang, 8, 4))
            J.append(('ZL3b', 'c', scheme, lang, 6, 2))
            J.append(('ZL3b', 's', scheme, lang, 6, 2))
            for p in ('plant_len', 'plant_pair', 'plant_bacon2'):
                J.append((p, 'g', scheme, lang, 4, 0))
            for n in ('Caesar', 'Manzoni'):
                J.append((n, 'c', scheme, lang, 4, 2))
        J.append(('ZL3b', 'gap', 'bacon', lang, 8, 0))
        J.append(('ZL3b', 'wpl', 'wpl', lang, 4, 4))
    # positive controls first so a broken pipeline is caught early
    J.sort(key=lambda j: (not j[0].startswith('plant'), j[3] != 'la'))
    return J


if __name__ == '__main__':
    J = jobs()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print(run(('plant_pair', 'g', 'pair', 'la', 1, 0)))
        sys.exit()
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(run, J)):
            print(i, len(J), os.path.basename(r), flush=True)
