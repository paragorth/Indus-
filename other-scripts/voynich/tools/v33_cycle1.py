"""v33 cycle 1: ecology profile of every corpus x block x segmentation rule (36 rules), curveball nulls.
Writes data/v33_ckpt/c1_<corpus>_<block>.json.  Usage: python3 v33_cycle1.py [nworkers]"""
import sys, time, zlib
from multiprocessing import Pool
from v33_lib import *

NNULL = 30


def build_corpora():
    C = {}
    V = voy_lines('ZL3b')
    for nm, L in [('V-ZL', V), ('V-IT', voy_lines('IT2a')), ('V-A', voy_lines('ZL3b', 'A')), ('V-B', voy_lines('ZL3b', 'B'))]:
        for i, b in enumerate(blocks(L, seed=33)): C[f'{nm}_{i}'] = b
    for c in ('la', 'cs', 'de', 'it', 'hu', 'tr', 'he'):
        for i, b in enumerate(blocks(lang_lines(c), seed=33)): C[f'L-{c}_{i}'] = b
    LA = lang_lines('la')
    for src, L in (('V', V), ('la', LA)):
        for i in range(2):
            rng = random.Random(100 + i)
            C[f'G-slot-{src}_{i}'] = SlotGen(L).gen(NTOK, rng)
            C[f'G-tri-{src}_{i}'] = Trigram(L).gen(NTOK, rng)
            C[f'G-table-{src}_{i}'] = TableGrille(L, seed=i).gen(NTOK, rng)
    for i in range(2):
        t = v26_tokens('ZL3b', seed=i); s = random.Random(i).randrange(0, len(t) - NTOK)
        C[f'G-v26-V_{i}'] = t[s:s + NTOK]
    zl = blocks(V, seed=33)
    for i, b in enumerate(zl):
        C[f'P-plant30_{i}'] = plant_paradigm(b, 0.30, seed=i)[0]
        C[f'P-plant15_{i}'] = plant_paradigm(b, 0.15, seed=i)[0]
        rng = random.Random(i)
        C[f'P-gshuf_{i}'] = [''.join(rng.sample(w, len(w))) for w in b]
    return C


def job(args):
    name, toks = args
    out = load(f'c1_{name}.json')
    if out: return name, 'cached'
    va = name.startswith(('V-', 'G-slot-V', 'G-tri-V', 'G-table-V', 'G-v26', 'P-'))
    res = {}
    for r in rule_set(20, voynich_alpha=va):
        r.fit(toks)
        M = matrix(toks, r)
        p = profile(M, NNULL, seed=zlib.crc32(r.name.encode()) % 1000)
        if name.startswith(('V-ZL', 'L-la', 'P-gshuf')):     # stem-ending re-pairing control
            Mr = matrix(toks, r, pairing_shuffle=random.Random(7))
            p['repair'] = profile(Mr, NNULL, seed=5)
        res[r.name] = p
    save(f'c1_{name}.json', res)
    return name, 'done'


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    C = load('corpora.json')
    if not C:
        C = build_corpora(); save('corpora.json', C)
    print('corpora', len(C), round(time.time() - t0), flush=True)
    with Pool(nw) as P:
        for n, s in P.imap_unordered(job, sorted(C.items())):
            print(n, s, round(time.time() - t0), flush=True)
