"""v85 cycle 4b: MOOD world control for cycle 4 = one order-2 glyph Markov per SECTION (fitted on all hands of that
section pooled, per line slot), used by every hand writing in that section: no dictionary, no meaning, no hand.
If it gives LEX(SS) > LEX(SH) like the Voynich, the cycle 4 ordering is not evidence of subject vocabulary."""
import sys
from multiprocessing import Pool
import v85_lib as L, v85_c4 as C4

NAME = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'


def mood_world(G):
    Cn = {k: L.canon(v) for k, v in G.items()}
    secs = {}
    for (s, h, l), P in Cn.items(): secs.setdefault((s, l), []).extend(P)
    M = {k: L.MK(P, 2) for k, P in secs.items()}
    W = {k: M[(k[0], k[2])].gen(P, 300 + i) for i, (k, P) in enumerate(Cn.items())}
    g = lambda s, h, l='B': W.get((s, h, l), [])
    return [('MOOD h2 herbal', g('H', '2'), g('B', '2'), g('H', '3') + g('H', '5'), g('S', '3'), 1000),
            ('MOOD h3 herbal', g('H', '3'), g('S', '3'), g('H', '2'), g('B', '2'), 1000),
            ('MOOD h3+h5 herbal', g('H', '3') + g('H', '5'), g('S', '3'), g('H', '2'), g('B', '2'), 1000),
            ('MOOD h2 text-only', g('T', '2'), g('H', '2'), g('T', '5'), g('S', '3'), 280)]


if __name__ == '__main__':
    P, G = L.voy_groups(NAME)
    with Pool(2) as pool:
        res = pool.map(C4.run, mood_world(G), chunksize=1)
    L.psave('c4b_%s.pkl' % NAME, res)
    for lab, N, d in res: print(C4.fmt(lab, N, d), flush=True)
