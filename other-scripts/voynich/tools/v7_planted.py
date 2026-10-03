"""v7 positive control: real-style numeric tables written in a Voynich-like
4-slot positional script (digit 0 = empty slot, other digits = a hidden
injective filler map per slot)."""
import math, random
import numpy as np
from v7_numlib import chop, add_para

SLOT_FILLERS = [
    ['', 'q', 'o', 'qo', 'oo', 'qoo', 'y', 'oy', 'qy', 'qoy'],
    ['', 'k', 't', 'p', 'f', 'C', 'S', 'K', 'T', 'P'],
    ['', 'e', 'ee', 'd', 's', 'ed', 'es', 'eee', 'ds', 'eed'],
    ['', 'n', 'in', 'iin', 'l', 'r', 'al', 'ar', 'm', 'ain'],
]

def hidden_map(seed):
    rng = random.Random(seed)
    maps = []
    for fl in SLOT_FILLERS:
        rest = fl[1:]; rng.shuffle(rest)
        maps.append([''] + rest)          # maps[k][digit] = filler
    return maps

def write_num(v, maps):
    s = '%04d' % v
    w = ''.join(maps[k][int(d)] for k, d in enumerate(s))
    return w if w else 'o'                 # a bare zero still needs ink: rare

def tables(n_lines, rng):
    """Lines of integers (< 10000), paragraphs of 10 lines of one table type."""
    out = []
    while len(out) < n_lines:
        typ = rng.choice(['arith', 'sun', 'dose', 'rows'])
        if typ == 'arith':
            for _ in range(10):
                M = rng.choice([360, 60, 1000, 10000]); d = rng.choice([1, 7, 12, 13, 24, 29, 30, 59])
                s = rng.randrange(M); k = rng.randint(6, 10)
                out.append([(s + i * d) % M for i in range(k)])
        elif typ == 'sun':
            L0 = rng.uniform(0, 360); g0 = rng.uniform(0, 360)
            for r in range(10):
                vals = []
                for i in range(8):
                    t = r * 8 + i
                    g = math.radians(g0 + 0.9856 * t)
                    vals.append(int(round(((L0 + 0.9856 * t + 1.915 * math.sin(g)) % 360) * 10)) % 3600)
                out.append(vals)
        elif typ == 'dose':
            for _ in range(10):
                k = rng.randint(3, 7)
                ds = []
                for _ in range(k):
                    x = max(1, int(math.exp(rng.gauss(2.3, 0.9))))
                    if rng.random() < 0.5: x = max(5, 5 * round(x / 5))
                    ds.append(min(x, 999))
                out.append(ds + [sum(ds)])
        else:
            n0 = rng.randint(1, 100); steps = [1, 13, 59, 30, 7, 365]
            mods = [10000, 360, 60, 360, 60, 10000]
            for r in range(10):
                n = n0 + r
                out.append([(n * s) % m for s, m in zip(steps, mods)] + [n * 10 % 1000])
    return out[:n_lines]

def planted(n_lines=4100, seed=11, noise=0.0, map_seed=5):
    rng = random.Random(seed)
    maps = hidden_map(map_seed)
    rows = tables(n_lines, rng)
    lines = []
    for j, r in enumerate(rows):
        ws = []
        for v in r:
            if noise and rng.random() < noise:
                v = rng.randrange(1, 10000)
            ws.append(write_num(v, maps))
        lines.append({'words': ws, 'para_start': j % 10 == 0, 'folio': 'p%d' % (j // 40)})
    return add_para(lines), maps
