"""v46 cycle 1: rate (S1) and exact-repetition (S2) search for number words, family-wise and held-out.
Controls: Hyginus De astronomia III (39 chapters, Ptolemy star counts as the independent 'picture' counts;
exact and noisy); planted number words in the real Voynich pages; Voynich real (pooled and per section)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v46_lib as L

NP = int(os.environ.get('V46_NP', 500))
log = []
def out(s):
    print(s, flush=True); log.append(s)

def summarise(tag, units, nperm=NP, held=True, seed=0):
    t0 = time.time()
    res, (C, T, n, secs, st) = L.run_search(units, nperm=nperm, seed=seed)
    r1 = res['s1']; r2 = res['s2']
    line = '%s: units %d, patterns %d | S1 best %s | S2 best %s' % (
        tag, len(units), res['npat'], L.top_rows(res, 's1', 1)[0], L.top_rows(res, 's2', 1)[0])
    if held:
        real, null = L.heldout(C, T, n, secs, st, nperm=100, seed=seed)
        line += ' | held-out S1 top10 %.3f (null %.3f +- %.3f, p %.3f)' % (real, null.mean(), null.std(), (1 + (null >= real).sum()) / (1 + len(null)))
    out(line + ' [%.0fs]' % (time.time() - t0))
    out('   S1 top: ' + ' ; '.join(L.top_rows(res, 's1', 6)))
    out('   S2 top: ' + ' ; '.join(L.top_rows(res, 's2', 3)))
    return res

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    R = {}
    if which in ('all', 'hyg'):
        H = L.hyginus_units()
        R['hyg'] = summarise('HYGINUS (Ptolemy counts)', H)
        rng = np.random.default_rng(1)
        for sd in (0.25, 0.5):
            Hn = [dict(u, n=int(round(u['n'] * np.exp(rng.normal(0, sd))))) for u in H]
            summarise('HYGINUS counts x lognormal noise sd %.2f' % sd, Hn)
        Hs = [dict(u, n=v) for u, v in zip(H, rng.permutation([u['n'] for u in H]))]
        summarise('HYGINUS counts shuffled (negative)', Hs)
    if which in ('all', 'plant'):
        V = L.voy_units()
        for word in ('okchey', 'qokeedy'):
            for a in (0.05, 0.1, 0.2, 0.4):
                Vp = L.plant(V, word, a, seed=3)
                res, (C, T, n, secs, st) = L.run_search(Vp, nperm=200, seed=0, stats=('s1',))
                j = res['names'].index('w:' + word) if 'w:' + word in res['names'] else None
                rank = int(np.nonzero(res['s1']['order'] == j)[0][0]) + 1 if j is not None else None
                zo = res['s1']['z'][j]; pfw = (1 + (res['s1']['maxnull'] >= zo).sum()) / 201
                out('PLANT rate %s alpha %.2f/object: planted word rank %s of %d, S1 %.3f, z %.1f, pFW %.3f' % (
                    word, a, rank, res['npat'], res['s1']['obs'][j], zo, pfw))
            for a in (0.2, 0.4, 0.7):
                Vp = L.plant(V, word, a, seed=4, mode='exact')
                res, _ = L.run_search(Vp, nperm=200, seed=0, stats=('s2',))
                j = res['names'].index('w:' + word)
                zo = res['s2']['z'][j]; pfw = (1 + (res['s2']['maxnull'] >= zo).sum()) / 201
                rank = int(np.nonzero(res['s2']['order'] == j)[0][0]) + 1
                out('PLANT exact %s p %.1f: planted word rank %d, S2 %.3f, z %.1f, pFW %.3f' % (word, a, rank, res['s2']['obs'][j], zo, pfw))
    if which in ('all', 'voy'):
        V = L.voy_units()
        R['voy'] = summarise('VOYNICH all sections pooled', V)
        for s in ('bio', 'stars', 'zodiac', 'pharma', 'herbal'):
            summarise('VOYNICH %s' % s, [u for u in V if u['sec'] == s])
        summarise('VOYNICH no herbal (eye-counted only)', [u for u in V if u['sec'] != 'herbal'])
        Vs = L.voy_units(secs=('stars',))
        summarise('VOYNICH stars, n = 8-point stars', [dict(u, n=u['n8']) for u in Vs])
    open(os.path.join(L.CK, 'c1_%s.log' % which), 'w').write('\n'.join(log))
