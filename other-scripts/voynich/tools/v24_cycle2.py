"""v24 cycle 2: the 'before-form' signature and its controls.
Datasets (500 sites max each, same rule family with 600 random rules):
  planted_gen   generator, frequency-weighted slips, rule-breaking slips corrected
  planted_lang  Caesar, frequency-weighted slips, non-words corrected (p .85), real-word slips (p .15)
  planted_copy  generator copied from an exemplar: neighbour-glyph slips, ALL corrected (exemplar restored)
  latin_plaoul  real Latin single-letter corrections (end-of-word a/b/c sigla insertions removed)
  voy_T1        Voynich ZL-marked corrections with a before-state (n=4)
  voy_T2        Voynich ZL [a:b] single-glyph alternatives (transcriber ambiguity; first reading as 'after')
Signature S = (before is a word) and (before is bigram-legal), each vs its random-edit expectation; plus
conditional identity|trigram-LP and legality|identity.  Small-n calibration: for every control, the share
of random 4-site subsets with >= 3/4 word befores, and the exact Poisson-binomial tail for the Voynich 4.
"""
import os, sys, json, pickle, math, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L
import v24_data as D
from v24_cycle1 import load

NSIM = 2000


def pb_tail(ps, k):
    """P(sum of Bernoulli(ps) >= k)."""
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p; dist[0] *= (1 - p)
    return float(dist[k:].sum())


def analyse(tag, lines, sites, slot=None, nmax=500, seed=0):
    rng = random.Random(seed)
    if len(sites) > nmax:
        sites = rng.sample(sites, nmax)
    M = L.Model(lines, *(slot or (None, None)))
    RS = L.RuleSet(M, n_random=600, slot=slot is not None)
    P = L.prepare(sites, RS)
    o = dict(tag=tag, n=len(P), kinds=dict(Counter(p['site'].kind for p in P)))
    r0 = L.run_test(P, nsim=NSIM, seed=1)
    o['core'] = L.summarize(r0, RS.names, core_only=True)
    o['top'] = L.summarize(r0, RS.names, top=6)
    o['zmax'] = float(r0['zmax']); o['pfw'] = float(np.nanmin(r0['pfw'])); o['nfw'] = int(np.nansum(r0['pfw'] < 0.05))
    rI = L.run_test(P, nsim=NSIM, seed=2, match=L.match_lp)
    rL = L.run_test(P, nsim=NSIM, seed=3, match=L.match_vocab)
    o['id_lp'] = (rI['n'], float(rI['z'][0]), float(rI['p'][0]), float(rI['z'][1])) if rI else None
    o['legal_id'] = (rL['n'], float(rL['z'][2]), float(rL['z'][15]), float(rL['z'][9]) if not math.isnan(rL['z'][9]) else None) if rL else None
    bw = np.array([p['vb'][0] for p in P]); nw = np.array([(p['pm'][:, 0] * p['pw']).sum() for p in P])
    bl = np.array([p['vb'][2] for p in P]); nl = np.array([(p['pm'][:, 2] * p['pw']).sum() for p in P])
    o['bword'] = float(bw.mean()); o['nword'] = float(nw.mean()); o['blegal'] = float(bl.mean()); o['nlegal'] = float(nl.mean())
    o['blp'] = float(np.mean([p['vb'][15] for p in P])); o['nlp'] = float(np.mean([(p['pm'][:, 15] * p['pw']).sum() for p in P]))
    # z of word-rate and legal-rate (Poisson-binomial normal approx)
    o['z_word'] = float((bw.sum() - nw.sum()) / math.sqrt((nw * (1 - nw)).sum() + 1e-9))
    o['z_legal'] = float((bl.sum() - nl.sum()) / math.sqrt((nl * (1 - nl)).sum() + 1e-9))
    o['before_word_pairs'] = [(''.join(p['site'].before), ''.join(p['site'].after)) for p in P[:12]]
    # small-n calibration
    if len(P) >= 8:
        hits = 0; R = 2000
        for _ in range(R):
            sub = rng.sample(range(len(P)), 4)
            hits += bw[sub].sum() >= 3
        o['share4_ge3'] = hits / R
    o['pb_tail'] = pb_tail(list(nw), int(bw.sum())) if len(P) <= 20 else None
    return o


def show(o):
    c = {r[0]: r for r in o['core']}
    print('%s n %d %s zmax %.1f pFW %.3g nFW %d' % (o['tag'], o['n'], o['kinds'], o['zmax'], o['pfw'], o['nfw']))
    print('  before is word %.2f (null %.2f, z %+.1f); bigram-legal %.2f (null %.2f, z %+.1f); trigram LP %.2f (null %.2f)' % (
        o['bword'], o['nword'], o['z_word'], o['blegal'], o['nlegal'], o['z_legal'], o['blp'], o['nlp']))
    print('  D-tests:', L.fmt_core([c[k] for k in ('vocab', 'big1', 'bilp', 'q3lp', 'slot', 'junc', 'lineinit', 'chsh2', 'firstlp', 'lastlp') if k in c]))
    print('  top:', L.fmt_top(o['top']))
    print('  identity|LP', o['id_lp'], ' legality|identity (n, big1 z, q3lp z, junc z)', o['legal_id'])
    print('  share of 4-subsets with >=3 word befores', o.get('share4_ge3'), ' PB tail', o['pb_tail'], flush=True)


if __name__ == '__main__':
    V = load('voy.pkl'); PL = load('plaoul.pkl')
    sm = json.load(open(os.path.join(L.DATA, 'derived', 'v7_slotmodels.json')))['voynich_ZL3b_K4']
    vslot = (sm['order'], sm['cuts'])
    pg_lines, pg_sites, pslot = D.planted_generator()
    pk_lines, pk_sites = D.planted_language()
    pc_lines, pc_sites, cslot = D.planted_copy()
    vocabL = Counter(w for Lw in PL['lines'] for w in Lw)
    lat = [s for s in PL['sites'] if not (s.kind == 'ins' and s.pos == len(s.after) - 1 and s.after[-1] in 'abc' and vocabL[s.after] <= 1)]
    print('planted gen', len(pg_sites), Counter(s.meta['broke'] for s in pg_sites), 'planted lang', len(pk_sites),
          Counter(s.meta['nonword'] for s in pk_sites), 'planted copy', len(pc_sites), 'latin', len(lat), flush=True)
    res = {}
    for tag, lines, sites, slot in [('planted_gen', pg_lines, pg_sites, tuple(pslot)),
                                    ('planted_lang', pk_lines, pk_sites, None),
                                    ('planted_copy', pc_lines, pc_sites, tuple(cslot)),
                                    ('latin_plaoul', PL['lines'], lat, None),
                                    ('voy_T1', V['corpus'], V['T1'], vslot),
                                    ('voy_T2', V['corpus'], V['T2'], vslot)]:
        o = analyse(tag, lines, sites, slot)
        res[tag] = o; show(o)
    json.dump(res, open(os.path.join(L.CK, 'c2_results.json'), 'w'), indent=1, default=str)
