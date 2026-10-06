"""v77 cycle 3b: the v38 picture-text lead (grade C) and the v18/v57 ink leads, on existing data.
M18 v38: 'a faint, diffuse picture-text resemblance in Currier A hand 1'. Would support: z >= 3 on held-out quires.
    Would kill: z < 1 with a CLIP-type network (none available offline) or loss when hand 1 is split by campaign.
    Strongest test now: the v38 neural composite (partial Mantel, v38 confounds) computed separately on two
    quire-disjoint halves of the hand-1 herbal pages, for each of the four networks (ResNet-18, DINO, EfficientNet-B0,
    DINOv2), and with the page text replaced by generator text fitted per section|language (SELFCIT, STACK, page-wise
    so that each page keeps its own size and position).
"""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L
from v38_lib import vis_sims, text_sims, text_profiles, Partial, mantel_table, cos_sim, DER
from v38_cycle1 import voynich_setup


def emb(vis2, keys, f):
    F = np.array([vis2[k][f] for k in keys], float)
    return cos_sim(F - F.mean(0))


def run():
    rng = np.random.default_rng(7738)
    pages, keys, words, vis, conf, strata = voynich_setup()
    vis2 = json.load(open(os.path.join(DER, 'v38_vis2_voynich.json')))
    A1 = [i for i, s in enumerate(strata) if s == 'A1']
    quires = sorted({pages[i]['quire'] for i in A1})
    halves = {0: [i for i in A1 if quires.index(pages[i]['quire']) % 2 == 0], 1: [i for i in A1 if quires.index(pages[i]['quire']) % 2 == 1]}
    V = vis_sims(vis, keys, ['r18', 'dino'])
    V['effb0'] = emb(vis2, keys, 'effb0'); V['dinov2'] = emb(vis2, keys, 'dinov2')
    texts = {'ZL3b': words}
    gw = {}
    for g in ('SELFCIT', 'STACK'):
        for s in (1, 2):
            G = L.generate('ZL3b', g, 7700 + s)
            m = {p['id']: [w for l in p['lines'] for w in l['w']] for p in G}
            # generator glyph units -> EVA strings like the v38 text (only identity matters for the text metrics)
            texts['%s:%d' % (g, s)] = [m.get(k, words[i]) for i, k in enumerate(keys)]
    out = {}
    for tname, W in texts.items():
        T = text_sims(text_profiles(W))
        for part_name, idx in [('A1', A1), ('A1_q0', halves[0]), ('A1_q1', halves[1])]:
            ii = np.ix_(idx, idx)
            part = Partial([C[ii] for C in conf.values()], len(idx))
            for nets in (['r18', 'dino'], ['effb0', 'dinov2'], ['r18'], ['dino'], ['effb0'], ['dinov2']):
                r = mantel_table({f: V[f][ii] for f in nets}, {m: M[ii] for m, M in T.items()}, part, 400, rng)
                out['%s|%s|%s' % (tname, part_name, '+'.join(nets))] = dict(z=r['omni_z'], p=r['omni_p'], n=len(idx))
                print(tname, part_name, nets, round(r['omni_z'], 2), len(idx), flush=True)
    L.jsave('c3b_m18.json', out)
    return out


def m19():
    """v18 C+ pen-load coherence: stated kill = an equal effect in a known-language manuscript with short repetitive
    words. v18_cycle5.coh (edge-aware content-residualised dips, page-swap surrogates) run on each CREMMA Latin
    manuscript alone, and on the Latin pages restricted to lines whose words average <= 4.5 letters."""
    import collections
    from v18_dips import Corpus
    from v18_lib import glyphs
    from v18_cycle5 import coh, short
    from v57_lib import lglyphs
    pages = json.load(open(os.path.join(DER, 'v18_latin_words.json')))
    pages = [p for p in pages if len(p['words']) >= 60]
    out = {}
    groups = collections.defaultdict(list)
    for p in pages: groups[p['folio'].split(':')[0]].append(p)
    groups['ALL_SHORTLINES'] = []
    for p in pages:
        byl = collections.defaultdict(list)
        for w in p['words']: byl[w['li']].append(w)
        keep = [w for li, ws in byl.items() if np.mean([len(x['word']) for x in ws]) <= 4.5 for w in ws]
        if len(keep) >= 20: groups['ALL_SHORTLINES'].append(dict(p, words=keep))
    for ms, P in groups.items():
        if len(P) < 3: continue
        lines = collections.OrderedDict()
        for p in P:
            for w in sorted(p['words'], key=lambda w: (w['li'], w['k'])):
                lines.setdefault((p['folio'], w['li']), []).append(w['word'])
        try:
            C = Corpus(P, lglyphs, list(lines.values()))
            r = coh(C, 2)
            out[ms] = short(r)
        except Exception as e:
            out[ms] = repr(e)
        print('M19', ms, len(P), out[ms], flush=True)
        L.jsave('c3b_m19.json', out)
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'm19': m19()
    else: run()
