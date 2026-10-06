"""v74 cycle 3a: a second outside witness for P1 - does Glen Claston, reading the vellum independently, also put a
space where ZL has one? For every ZL junction a.b on lines present in both files (glyph strings equal once spaces
are removed), check whether GC has a space at the same glyph offset. Compare the 7 frozen targets with the tight
control (same first word, same first glyph of the next word, other next word) and with all junctions; and the
ZL ',' vs '.' spaces. Out: data/v74_ckpt/c3.json"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v74_lib as X
import v74_c1 as C

if __name__ == '__main__':
    Z = X.ivtff(os.path.join(X.ROOT, 'data', 'ZL3b-n.txt'))
    G = {(r['folio'], r['n']): r for r in X.jload('GC_lines.json')}
    firsts = {a for a, _ in X.TARGETS}
    cnt = collections.defaultdict(lambda: [0, 0]); used = 0
    for key, z in Z.items():
        g = G.get(key)
        if not g or ''.join(z['words']) != ''.join(g['words']) or '?' in ''.join(z['words']): continue
        used += 1
        cuts, o = set(), 0
        for w in g['words'][:-1]: o += len(w); cuts.add(o)
        o = 0
        for k in range(len(z['words']) - 1):
            a, b = z['words'][k], z['words'][k + 1]; o += len(a)
            hit = o in cuts
            cls = []
            if (a, b) in X.TSET: cls.append('target')
            elif a in firsts and C.first_g(b) in {C.first_g(bb) for aa, bb in X.TARGETS if aa == a}: cls.append('tight_ctl')
            if a in firsts and (a, b) not in X.TSET: cls.append('same_word_ctl')
            cls += ['all', 'zl' + z['seps'][k]]
            for c in cls: cnt[c][0] += hit; cnt[c][1] += 1
    out = dict(lines=used, **{k: dict(gc_split=v[0], n=v[1], share=v[0] / max(v[1], 1)) for k, v in cnt.items()})
    X.jsave('c3.json', out); print(json.dumps(out, indent=1))
