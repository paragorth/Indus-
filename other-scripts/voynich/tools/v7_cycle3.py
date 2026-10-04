"""v7 cycle 3: robustness of the cycle-1 verdict.
 (a) second transcription IT2a (own slot model, K = 4);
 (b) K = 3 and K = 5 digit slots on ZL3b (K = 5 = best single extra cut added
     to the K = 4 cuts);
 (c) token-level numerals: anneal for Benford fit + roundness only, on
     Voynich, verbose Latin, planted tables and a glyph-shuffled Voynich.
Each (a)/(b) run: fit on even paragraphs, score odd, real vs within-line
shuffle, 2 restarts."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
from v7_numlib import *
from v7_planted import planted
import gen

CACHE = os.path.join(DATA, 'derived', 'v7_slotmodels.json')

def cached(key, fn):
    c = json.load(open(CACHE))
    if key not in c:
        c[key] = fn(); json.dump(c, open(CACHE, 'w'), indent=1)
    return c[key]

def job(a):
    kind, name, variant, seed, steps = a
    lines, model = WORK[name]
    if variant == 'shuf_inline': lines = shuffle_inline(lines, 400 + seed)
    if variant == 'glyph_shuffle': lines = gen.char_shuffle(lines, seed=7)
    E = encode(lines, model)
    if kind == 'tok':
        best, D = anneal(E, steps=steps, seed=seed, obj=score_tok, T0=0.02, T1=0.0002)
        return (kind, name, variant, seed, score(E, values(E, D), True), None, float(E['valid'].mean()))
    npara = E['PA'].max() + 1; even = np.arange(npara) % 2 == 0
    Etr, Ete = subset(E, even), subset(E, ~even)
    best, D = anneal(Etr, steps=steps, seed=seed)
    return (kind, name, variant, seed, score(Etr, values(Etr, D), True), score(Ete, values(Ete, D), True), float(E['valid'].mean()))

def init(w):
    global WORK; WORK = w

def main(steps=8000):
    zl = voynich('ZL3b'); it = voynich('IT2a')
    m4 = json.load(open(CACHE))['voynich_ZL3b_K4']
    m_it = cached('voynich_IT2a_K4', lambda: slot_model(it, 4))
    m3 = cached('voynich_ZL3b_K3', lambda: slot_model(zl, 3, order=m4['order']))
    c4 = m4['cuts'][1:-1]; U = len(m4['order'])
    cand5 = [tuple(sorted(c4 + [x])) for x in range(1, U) if x not in c4]
    m5 = cached('voynich_ZL3b_K5', lambda: slot_model(zl, 5, order=m4['order'], cand=cand5))
    for nm, m in (('IT2a K4', m_it), ('ZL K3', m3), ('ZL K5', m5)):
        print(nm, ''.join(m['order']), m['cuts'], 'coverage %.3f' % m['coverage'], m['fillers'], flush=True)
    pc, _ = planted()
    lat = latin_verbose()
    c = json.load(open(CACHE))
    work = {'ZL_K4': (zl, m4), 'IT_K4': (it, m_it), 'ZL_K3': (zl, m3), 'ZL_K5': (zl, m5),
            'latin': (lat, c['latin_verbose_K4']), 'planted': (pc, c['planted_clean_K4'])}
    jobs = [('ord', n, v, s, steps) for n in ('IT_K4', 'ZL_K3', 'ZL_K5') for v in ('real', 'shuf_inline') for s in range(2)]
    jobs += [('tok', n, 'real', 0, steps) for n in ('ZL_K4', 'latin', 'planted')] + [('tok', 'ZL_K4', 'glyph_shuffle', 0, steps)]
    out = []
    with Pool(4, initializer=init, initargs=(work,)) as p:
        for kind, name, variant, seed, tr, te, val in p.imap_unordered(job, jobs):
            out.append({'kind': kind, 'corpus': name, 'variant': variant, 'seed': seed, 'train_or_all': tr, 'test': te, 'valid_frac': val})
            if kind == 'tok':
                print('tok %-8s %-13s valid=%.2f benford_fit=%.3f round=%.3f' % (name, variant, val, tr['benford_fit'], tr['round']), flush=True)
            else:
                print('ord %-8s %-12s s%d valid=%.2f trainJ=%.3f testJ=%.3f mono=%.3f cd=%.4f cmono=%.3f ccd=%.4f sum=%.4f' % (
                    name, variant, seed, val, tr['J'], te['J'], te['mono'], te['cdiff'], te['colmono'], te['colcdiff'], te['sum']), flush=True)
    save('v7_cycle3', out)

if __name__ == '__main__':
    main()
