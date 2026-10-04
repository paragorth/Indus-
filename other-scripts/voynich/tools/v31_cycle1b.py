"""v31 cycle 1b: is the MAGIC class an artefact of how the voces were extracted?

Control: pass texts through the SAME extraction (keep only maximal runs of >= 2 words that are in none of the
7 lexicons, drop near-lexical runs) and classify the output with the cycle-1 model. Languages outside the
lexicons (Swahili, Tagalog, Maori, Nahuatl, Turkish, Kaqchikel, Wolof, Yoruba, Pinyin) come through almost
whole: if they are then classed MAGIC, the class is a filter artefact. The Voynich (EVA words) and gibberish
are passed through the same filter, so all three are compared on equal terms.
"""
import os, sys, json, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_lib as L, v31_corpora as VC, v31_cycle1 as C1
from multiprocessing import Pool

N = 100
FN = 'v31_cycle1.txt'


def filt_docs(docs):
    txt = '\n\n'.join('\n'.join(' '.join(l) for l in d) for d in docs)
    return [VC.vox_runs(txt, VC.ALL_LANGS)]


def voy_raw():
    import vlib, v21_lib as V
    lines = vlib.load_voynich('ZL3b', drop_uncertain=True)
    out = []; cur = None; last = None
    for l in lines:
        ws = [w for w in l['words'] if '?' not in w]
        if not ws: continue
        out.append(ws)
    return [out]


def work(a):
    name, i, lines = a
    rng = random.Random(i)
    return name, L.features(lines, rng)


def main():
    import v21_lib as V
    C = json.load(open(os.path.join(L.CK, 'corpora.json')))
    src = {}
    for k in ['L_Swahili_Lite', 'L_Tagalog_Lite', 'L_Maori_Lite', 'L_Nahuatl_Lite', 'L_Turkish_Lite', 'L_MayanKaqch_Lite',
              'L_Wolof_Tech', 'L_Yoruba_Tech', 'L_ChinesePin_Lite', 'I_Klingon_Lite', 'I_Lojban_Lite', 'L_msC_Old', 'L_msG_Bav2']:
        src['F_' + k] = filt_docs([d[:3000] for d in C[k]['docs']])
    gib = [d for k, v in C.items() if v['cls'] == 'GIBB' for d in v['docs']]
    src['F_gibberish'] = filt_docs(gib)
    vr = filt_docs(voy_raw())
    src['F_Voynich_ZL'] = [[[V.U(w) for w in l] for l in vr[0]]]
    jobs = []
    for k, docs in src.items():
        S = L.samples(docs, N=N, maxs=40)
        n_in = sum(len(l) for d in docs for l in d)
        print(k, 'tokens kept', n_in, 'samples', len(S), flush=True)
        jobs += [(k, i, s) for i, s in enumerate(S)]
    with Pool(2) as p: R = p.map(work, jobs, chunksize=4)
    Rtr, keys, X, corp, cls = C1.load(N)
    tr = np.isin(cls, C1.TRAIN)
    P, cl = C1.fit(X[tr], cls[tr])
    out = {}
    for k in src:
        Z = np.array([[f[q] for q in keys] for n, f in R if n == k], float)
        if not len(Z): continue
        Z[~np.isfinite(Z)] = 0
        pr = P(Z)
        out[k] = {'n': len(Z), 'mean_p': dict(zip(cl, pr.mean(0).round(3))), 'votes': dict(Counter(np.array(cl)[pr.argmax(1)]))}
        print(k, out[k], flush=True)
    L.save('cycle1b.json', out)
    langs = [k for k in out if k.startswith('F_L_')]
    nm = sum(1 for k in langs if max(out[k]['mean_p'], key=out[k]['mean_p'].get) == 'MAGIC')
    L.row(FN, 'V-31.1f', 'Extraction-artefact control: unknown-to-lexicon languages, conlangs, gibberish and the Voynich passed through the SAME voces filter, classified by the cycle-1 model',
          '; '.join(f"{k[2:]}: {max(v['mean_p'], key=v['mean_p'].get)} ({max(v['mean_p'].values()):.2f})" for k, v in out.items()),
          f'{nm}/{len(langs)} filtered natural languages classed MAGIC: ' + ('the MAGIC class is largely a filter artefact' if nm >= len(langs) / 2 else 'the MAGIC class is not a filter artefact'))


if __name__ == '__main__':
    main()
