"""v74 cycle 2b: score the frozen P2 on Glen Claston's transcription (GC2a, v101 -> EVA, tools/v74_gc.py).
Uses the v72 code unchanged (v72_lib, v72_c2.run, v72_c3.run, v72_c3b.run, v72_c2.arrows) with the frozen rule
read from data/v72_ckpt/frozen.json (E1c_keepd, hash checked against 2803cbeb0deb51bf before anything runs).
Corpus 'GC' is loaded like L.voynich (P lines, [a-z]+ words, pages >= 20 words, leaf-parity halves).
Controls: GC~SC10 (the 10%-copy generator fitted to GC), GC~JUNC (junction generator), GC~WSHUF.
Sensitivity corpus 'GCb': the three commonest family-fallback v101 symbols resolved to their ZL-majority glyph.
Outputs go to data/v74_ckpt/c2/ (never into v72's checkpoint)."""
import os, sys, re, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L
import v72_c2 as C2
import v72_c3 as C3
import v72_c3b as C3B
import v74_lib as X
import vlib

OUT = os.path.join(X.CK, 'c2'); os.makedirs(OUT, exist_ok=True)
C2.OUT = OUT; C3.OUT = OUT
assert L.rule_hash('E1c_keepd') == '2803cbeb0deb51bf' and C2.freeze()['hash'] == '2803cbeb0deb51bf'
_orig_corpus = C2.corpus


def gc_pages(fname):
    recs = X.jload(fname)
    pages = collections.OrderedDict()
    for r in recs:
        if r['ltype'] != 'P': continue
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if re.fullmatch(r'[a-z]+', w)]
        if not ws: continue
        p = pages.setdefault(r['folio'], dict(id=r['folio'], sec=r['illus'], lang=r['lang'] or '-', hand=r['hand'] or '-',
                                              quire=r['quire'], lines=[]))
        p['lines'].append(dict(w=ws, ps=bool(r['para_start'])))
    return [p for p in pages.values() if sum(len(l['w']) for l in p['lines']) >= 20]


def corpus(name):
    if name in C2._CACHE: return C2._CACHE[name]
    base, gen = name.split('~') if '~' in name else (name, None)
    if base in ('GC', 'GCb'):
        S = gc_pages('GC_lines.json' if base == 'GC' else 'GCb_lines.json')
        half = [L.leaf_half(p['id']) for p in S]
        if gen: S = L.GENS[gen](S, seed=2)
        C2._CACHE[name] = (S, None, half)
        return C2._CACHE[name]
    return _orig_corpus(name)


C2.corpus = corpus


def arrows_ctrl(name):
    fn = os.path.join(OUT, f'arrows_{name}.json')
    if os.path.exists(fn): return json.load(open(fn))
    fz = C2.freeze()
    S, _, half = corpus(name)
    Xp = [p for p, h in zip(L.extract(S, L.RULES[fz['rule']]), half) if h == 1]
    out = {}
    nops = lambda P: [dict(p, lines=[l for l in p['lines'] if not l['ps']]) for p in P]
    for k, f in [('real', lambda P: P), ('no_para_first', nops)]:
        out[k] = [C2.arrows(f(Xp), s) for s in (0, 1, 2)]
        print(name, k, [a['surv'] for a in out[k]], flush=True)
    json.dump(out, open(fn, 'w'), default=float)
    return out


def job(j):
    kind, name = j
    if kind == 'c2': C2.run((name, 'payload'))
    elif kind == 'c3': C3.run(name)
    elif kind == 'c3b':
        fn = os.path.join(OUT, f'c3b_{name}.json')
        if not os.path.exists(fn): json.dump(C3B.run(name), open(fn, 'w'), default=float)
    elif kind == 'arr': arrows_ctrl(name)
    return j


if __name__ == '__main__':
    names = sys.argv[1:] or ['GC']
    J = []
    for n in names:
        J += [('c2', n), ('c3', n), ('c3b', n), ('arr', n), ('c2', n + '~SC10'), ('c2', n + '~JUNC'), ('c3', n + '~JUNC')]
    from multiprocessing import Pool
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, J, chunksize=1): print('done', r, flush=True)
