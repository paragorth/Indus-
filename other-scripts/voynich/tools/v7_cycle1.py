"""v7 cycle 1: does any glyph->digit reading of the slot grammar make Voynich
lines behave like numeric tables, beyond what the same search finds in
shuffled text?  Fit on even paragraphs, score on odd paragraphs (held out).
Corpora: planted tables (clean; 30% noise) = positive controls; Isidore under
verbose substitution = language control; Voynich ZL3b paragraph text.
Nulls for each: global word shuffle and within-line shuffle (same slot model)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
from v7_numlib import *
from v7_planted import planted

CACHE = os.path.join(DATA, 'derived', 'v7_slotmodels.json')

def get_model(name, lines, K=4):
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = '%s_K%d' % (name, K)
    if key not in c:
        c[key] = slot_model(lines, K)
        json.dump(c, open(CACHE, 'w'), indent=1)
    return c[key]

def corpora():
    pc, maps = planted()
    pn, _ = planted(noise=0.3)
    return {'planted_clean': (pc, maps), 'planted_noise30': (pn, maps),
            'latin_verbose': (latin_verbose(), None), 'voynich_ZL3b': (voynich('ZL3b'), None)}

def job(args):
    name, variant, rest, seed, rev, steps = args
    lines, model = WORK[name]
    if variant == 'shuf_global': lines = shuffle_global(lines, 100 + seed)
    elif variant == 'shuf_inline': lines = shuffle_inline(lines, 200 + seed)
    E = encode(lines, model)
    npara = E['PA'].max() + 1
    even = np.arange(npara) % 2 == 0
    Etr, Ete = subset(E, even), subset(E, ~even)
    best, D = anneal(Etr, steps=steps, seed=seed * 7 + rest, rev=rev)
    te = score(Ete, values(Ete, D, rev), True)
    tr = score(Etr, values(Etr, D, rev), True)
    return (name, variant, rest, seed, rev, tr, te, [d.tolist() for d in D])

def init(work):
    global WORK; WORK = work

def main(steps=10000):
    t0 = time.time()
    C = corpora(); work = {}; out = {'meta': {}, 'runs': [], 'randnull': {}, 'truth': {}}
    for name, (lines, maps) in C.items():
        m = get_model(name, lines)
        work[name] = (lines, m)
        out['meta'][name] = {'order': ''.join(m['order']), 'cuts': m['cuts'], 'coverage': m['coverage'],
                             'fillers': m['fillers'], 'tokens': sum(len(L['words']) for L in lines)}
        E = encode(lines, m); npara = E['PA'].max() + 1
        Ete = subset(E, np.arange(npara) % 2 == 1)
        rng = np.random.default_rng(1)
        rs = [score(Ete, values(Ete, random_D(4, rng)), True) for _ in range(200)]
        out['randnull'][name] = {k: [float(np.mean([r[k] for r in rs])), float(np.max([r[k] for r in rs]))]
                                 for k in ('J', 'mono', 'cdiff', 'colmono', 'colcdiff', 'sum', 'benford_fit', 'round')}
        if maps is not None:
            D = [np.array([maps[k].index(f) if f in maps[k] else 0 for f in m['fillers'][k]]) for k in range(4)]
            out['truth'][name] = score(Ete, values(Ete, D), True)
            out['truth_D'] = [d.tolist() for d in D]
        print(name, out['meta'][name]['order'], m['cuts'], round(m['coverage'], 3), 'rand', out['randnull'][name]['J'], flush=True)
    jobs = []
    for name in work:
        for variant in ('real', 'shuf_global', 'shuf_inline'):
            for rest in range(2):
                jobs.append((name, variant, rest, rest, False, steps))
    jobs.append(('voynich_ZL3b', 'real', 9, 9, True, steps))
    jobs.append(('voynich_ZL3b', 'shuf_inline', 9, 9, True, steps))
    with Pool(4, initializer=init, initargs=(work,)) as p:
        for r in p.imap_unordered(job, jobs):
            name, variant, rest, seed, rev, tr, te, D = r
            rec = {'corpus': name, 'variant': variant, 'restart': rest, 'rev': rev, 'train': tr, 'test': te, 'D': D}
            if name.startswith('planted') and not rev:
                T = out['truth_D']; rec['digit_acc'] = float(np.mean([D[k][i] == T[k][i] for k in range(4) for i in range(10)]))
            out['runs'].append(rec)
            print('%-16s %-12s r%d rev=%d  trainJ=%.3f testJ=%.3f mono=%.3f cd=%.4f cmono=%.3f ccd=%.4f sum=%.4f benf=%.2f rnd=%.2f %s' % (
                name, variant, rest, rev, tr['J'], te['J'], te['mono'], te['cdiff'], te['colmono'], te['colcdiff'], te['sum'],
                te['benford_fit'], te['round'], ('acc=%.2f' % rec['digit_acc']) if 'digit_acc' in rec else ''), flush=True)
    out['elapsed_s'] = time.time() - t0
    save('v7_cycle1', out)

if __name__ == '__main__':
    main()
