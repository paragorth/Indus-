"""v5 cycle 4 -- section-level 'mode' and 'tonic echo'.

E1 tonic echo: in modal music the final is a central pitch of the piece, so the closing symbol is
   over-represented in the unit's interior. Statistic: log ratio of the closing symbol's interior share in its
   unit over its mean interior share in other units of the same stratum (Voynich: section x Currier language;
   chant: none, since mode is what we want to see). Compared with the same for the final of line 2 and for a
   random interior symbol of a random line (a burstiness baseline: any symbol is enriched in its own unit).
E2 mode consistency: are units of one stratum more alike in their CLOSING symbol than in a mid-line word-final
   symbol? Statistic: excess MI(stratum; symbol) for closing vs line-2-final vs mid word-final, null permutes
   stratum labels over units (200). Chant stratum = mode (positive control); Voynich stratum = section.
"""
import sys, os, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib


def interior(u):
    c = Counter()
    for L in u['lines']:
        c.update([x for w in L for x in w][1:-1])
    return c


def e1(units, seed=0):
    rng = random.Random(seed)
    us = [u for u in units if len(u['lines']) >= 3]
    ints = [interior(u) for u in us]
    tots = [max(sum(c.values()), 1) for c in ints]
    by = defaultdict(list)
    for i, u in enumerate(us):
        by[u['stratum']].append(i)

    def enrich(i, s):
        own = (ints[i][s] + 0.5) / tots[i]
        others = [j for j in by[us[i]['stratum']] if j != i]
        if not others:
            return None
        o = np.mean([(ints[j][s] + 0.5) / tots[j] for j in others])
        return math.log(own / o)
    out = {}
    for lab in ['closing', 'line2_final', 'random_interior']:
        v = []
        for i, u in enumerate(us):
            if lab == 'closing':
                s = u['lines'][-1][-1][-1]
            elif lab == 'line2_final':
                s = u['lines'][1][-1][-1]
            else:
                L = rng.choice(u['lines']); seq = [x for w in L for x in w][1:-1]
                if not seq:
                    continue
                s = rng.choice(seq)
            e = enrich(i, s)
            if e is not None:
                v.append(e)
        v = np.array(v)
        out[lab] = {'n': len(v), 'mean_log_enrich': float(v.mean()), 'se': float(v.std() / math.sqrt(len(v)))}
    return out


def mi(a, b):
    n = len(a); cj = Counter(zip(a, b)); ca = Counter(a); cb = Counter(b)
    return sum(v / n * math.log2(v * n / (ca[x] * cb[y])) for (x, y), v in cj.items())


def e2(units, seed=0, R=200):
    rng = random.Random(seed)
    us = [u for u in units if len(u['lines']) >= 3 and len(u['lines'][1]) >= 3]
    st = [u['stratum'] for u in us]
    syms = {'closing': [u['lines'][-1][-1][-1] for u in us],
            'line2_final': [u['lines'][1][-1][-1] for u in us],
            'mid_wordfinal': [u['lines'][1][len(u['lines'][1]) // 2][-1] for u in us]}
    out = {}
    for k, s in syms.items():
        obs = mi(st, s); nn = []
        for _ in range(R):
            p = list(st); rng.shuffle(p); nn.append(mi(p, s))
        out[k] = {'excess': obs - np.mean(nn), 'z': (obs - np.mean(nn)) / (np.std(nn) + 1e-12)}
    return out


if __name__ == '__main__':
    C = {'V-ZL': vc.voynich('ZL3b'), 'V-IT': vc.voynich('IT2a'), 'chant': vc.chant(max_units=1500)}
    C['vs-Italian-Manzoni'] = vc.verbose_lang('Italian-Manzoni')
    C['vs-Latin-Caesar'] = vc.verbose_lang('Latin-Caesar')
    res = {}
    for k, units in C.items():
        if k.startswith('V-'):
            units1 = [dict(u, stratum=f"{u['stratum']}{u['lang']}") for u in units]
        else:
            units1 = units
        r = {'e1': e1(units1)}
        if k.startswith('V-') or k == 'chant':
            r['e2'] = e2(units)
        res[k] = r
        print(f"== {k}")
        for lab, v in r['e1'].items():
            print(f"  E1 {lab:16s} n={v['n']:4d} mean log enrichment {v['mean_log_enrich']:+.3f} +- {v['se']:.3f}")
        if 'e2' in r:
            print('  E2 ' + ' '.join(f"{a}: {b['excess']:.3f} (z {b['z']:.1f})" for a, b in r['e2'].items()))
    vlib.save('v5_cycle4', res)
