"""v7 cycle 2: sensitivity.  How much numeric table content could hide in the
Voynich without the cycle-1 search seeing it?  Background = Voynich ZL3b
paragraph text with words shuffled within each line (keeps vocabulary, line
make-up and the slot model, destroys any order).  A fraction p of paragraphs
is replaced by planted tables (v7_planted.tables) written with the Voynich
slot model's OWN fillers under a hidden digit map (digit 0 = empty filler),
so the planted words are ordinary-looking Voynich words.  Same held-out
annealing as cycle 1; detection = test J above the p = 0 runs.
Second part: the real (unshuffled) Voynich restricted to sections, held-out,
against within-line shuffles of the same section."""
import sys, os, json, time, random
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
from v7_numlib import *
from v7_planted import tables

CACHE = os.path.join(DATA, 'derived', 'v7_slotmodels.json')

def mix(lines, model, p, seed):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    F = model['fillers']; K = len(F)
    maps = []
    for k in range(K):
        z = F[k].index('') if '' in F[k] else 0
        rest = [i for i in range(10) if i != z]; rng.shuffle(rest)
        maps.append([F[k][z]] + [F[k][i] for i in rest])  # maps[k][digit]
    paras = sorted({L['para'] for L in lines})
    chosen = {q for q in paras if rng.random() < p}
    rows = tables(len(lines), rng); ri = 0
    out = []
    for L in shuffle_inline(lines, seed + 1):
        nl = dict(L)
        if L['para'] in chosen:
            ws = []
            for v in rows[ri]:
                w = ''.join(maps[k][int(d)] for k, d in enumerate('%04d' % v))
                if w: ws.append(w)
            ri += 1; nl['words'] = ws
        out.append(nl)
    frac = sum(len(L['words']) for L in out if L['para'] in chosen) / sum(len(L['words']) for L in out)
    return out, frac

def heldout(lines, model, seed, steps):
    E = encode(lines, model); npara = E['PA'].max() + 1
    even = np.arange(npara) % 2 == 0
    Etr, Ete = subset(E, even), subset(E, ~even)
    best, D = anneal(Etr, steps=steps, seed=seed)
    return score(Etr, values(Etr, D), True), score(Ete, values(Ete, D), True)

def job(a):
    kind, p, seed, steps = a
    if kind == 'mix':
        lines, frac = mix(BASE, MODEL, p, seed)
        tr, te = heldout(lines, MODEL, seed, steps)
        return (kind, p, seed, frac, tr, te)
    sec, variant = p
    lines = [L for L in BASE if L['illus'] in sec]
    if variant == 'shuf_inline': lines = shuffle_inline(lines, 300 + seed)
    tr, te = heldout(lines, MODEL, seed, steps)
    return (kind, p, seed, None, tr, te)

def init(b, m):
    global BASE, MODEL; BASE, MODEL = b, m

def main(steps=8000):
    base = voynich('ZL3b')
    model = json.load(open(CACHE))['voynich_ZL3b_K4']
    jobs = [('mix', p, s, steps) for p in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4) for s in range(3)]
    for sec in ('H', 'S', 'B', 'P', 'AC'):
        for v in ('real', 'shuf_inline'):
            for s in range(2):
                jobs.append(('sec', (sec, v), s, steps))
    out = []
    with Pool(4, initializer=init, initargs=(base, model)) as pool:
        for kind, p, seed, frac, tr, te in pool.imap_unordered(job, jobs):
            out.append({'kind': kind, 'p': p, 'seed': seed, 'numeric_token_frac': frac, 'train': tr, 'test': te})
            print('%-4s %-22s s%d frac=%s trainJ=%.3f testJ=%.3f mono=%.3f cd=%.4f cmono=%.3f ccd=%.4f sum=%.4f' % (
                kind, p, seed, ('%.3f' % frac) if frac is not None else '-', tr['J'], te['J'], te['mono'], te['cdiff'],
                te['colmono'], te['colcdiff'], te['sum']), flush=True)
    save('v7_cycle2', out)

if __name__ == '__main__':
    main()
