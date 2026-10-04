"""v48 cycle 3: massive random normalisation search.  Let the map choose the cleaning.

Each hypothesis = 0-6 unit merges (pairs of units with count >= 50) + 0-3 bigram fusions (a frequent within-word
unit pair written as one unit) + optional blind A/B undo.  Score = out-score (NN distance to the 60-language map
/ median language NN distance) on fold 1 (alternate pages), one 8,000-token window.  The 15 best are re-scored on
fold 2 (held-out pages, all windows) against the identity hypothesis on fold 2.

Targets (same search, same budget):
  VOY_ZL   ZL3b glyph units, raw (E0)            - does the search re-find k=t / ch=sh, and how far does it get?
  VOY_IT   IT2a glyph units, raw                 - transcription check
  LAT_PL   planted Latin (positional twins + bench variant + A/B rewrite) - positive control: must find its twins
  VOY_MK2  Markov-2 resynthesis of ZL3b          - null: how much 'language-likeness' a search can buy on a text
                                                   with Voynich unit statistics and no words
"""
import sys, os, json, random, itertools
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v48_lib as L

OUT = 'v48_cycle3.txt'
NH = 1000
_M = None
_T = None


def get_map():
    c1 = L.load('c1_results.json')
    lang = c1['lang']
    return L.Map([r['fp'] for r in lang], [r['name'] for r in lang], [r['fam'] for r in lang])


def targets():
    T = {}
    for src in ('ZL3b', 'IT2a'):
        T['VOY_' + src[:2]] = (L.voynich(src, 'E0', 'GLY'), {})
    import json as J
    lat = L.build('eb_lat', 'Latin', J.load(open(os.path.join(L.RAW, 'eb_lat.json'))))
    P, tr = L.plant(lat, seed=2)
    T['LAT_PL'] = (P, tr)
    T['VOY_MK2'] = (L.null_markov2(T['VOY_ZL'][0], seed=5), {})
    return T


def folds(C):
    a = dict(C, docs=C['docs'][0::2]); b = dict(C, docs=C['docs'][1::2])
    return a, b


def apply_h(C, h):
    mp = dict(h['merge'])
    C = L.merge_units(C, mp) if mp else C
    for x, y in h['fuse']:
        def f(w, d, li, wi, x=x, y=y):
            out, k = [], 0
            while k < len(w):
                if k + 1 < len(w) and w[k] == x and w[k + 1] == y:
                    out.append(x + y); k += 2
                else:
                    out.append(w[k]); k += 1
            return tuple(out)
        C = L.mapc(C, f)
    if h['ab']:
        C, _ = L.ab_blind(C)
    return C


def random_h(C, rng):
    cnt = Counter(u for w in L.tokens(C) for u in w)
    U = [u for u, c in cnt.items() if c >= 50]
    bg = Counter((w[i], w[i + 1]) for w in L.tokens(C) for i in range(len(w) - 1))
    B = [p for p, _ in bg.most_common(30)]
    h = dict(merge=[], fuse=[], ab=rng.random() < 0.5)
    used = set()
    for _ in range(rng.randint(0, 6)):
        x, y = rng.sample(U, 2)
        if x in used or y in used: continue
        hi, lo = (x, y) if cnt[x] >= cnt[y] else (y, x)
        h['merge'].append((lo, hi)); used |= {x, y}
    for _ in range(rng.randint(0, 3)):
        h['fuse'].append(rng.choice(B))
    return h


def init(tname):
    global _M, _T
    _M = get_map()
    _T = targets()


def score(args):
    tname, h, fold, full = args
    C = _T[tname][0]
    a, b = folds(C)
    X = apply_h(a if fold == 0 else b, h)
    toks = L.tokens(X)
    if full:
        fp = L.fp_corpus(X)
    else:
        fp = L.fingerprint(toks[:L.WIN])
        fp = {k: fp[k] for k in L.FEATS}
    p = _M.place(fp)
    return dict(out=p['out'], nn=[x[0] for x in p['nn'][:3]], fam=[x[1] for x in p['nn'][:3]])


def main():
    res = L.load('c3_results.json') or {}
    T = targets()
    with Pool(2, initializer=init, initargs=(None,)) as pool:
        for tname in ('LAT_PL', 'VOY_ZL', 'VOY_MK2', 'VOY_IT'):
            if tname in res: continue
            C, truth = T[tname]
            rng = random.Random(480 + len(tname))
            H = [dict(merge=[], fuse=[], ab=False)] + [random_h(C, rng) for _ in range(NH)]
            # seed the space with the explicit Voynich merges so the search can find them (not for controls' truth)
            sc = pool.map(score, [(tname, h, 0, False) for h in H], chunksize=8)
            order = np.argsort([s['out'] for s in sc])
            top = [int(i) for i in order[:15] if i != 0][:15]
            held = pool.map(score, [(tname, H[i], 1, True) for i in [0] + top], chunksize=1)
            res[tname] = dict(H=H, sc=sc, top=top, held=held, truth=truth)
            L.save('c3_results.json', res)
            print(tname, 'done', flush=True)
    report(res, T)


def report(res, T):
    lines = {}
    for tname, r in res.items():
        H, sc, top, held = r['H'], r['sc'], r['top'], r['held']
        base1 = sc[0]['out']; base2 = held[0]['out']
        gain1 = [base1 - sc[i]['out'] for i in top]
        gain2 = [base2 - h['out'] for h in held[1:]]
        allg = np.array([base1 - s['out'] for s in sc[1:]])
        # merge enrichment
        mc = Counter(); mt = Counter()
        for i, s in enumerate(sc[1:], 1):
            for m in H[i]['merge']:
                key = tuple(sorted(m)); mt[key] += 1
                if i in top: mc[key] += 1
        enr = sorted(((mc[k] / max(1, len(top))) / (mt[k] / NH), k) for k in mc if mt[k] >= 5)[::-1][:5]
        truth = r['truth'].get('twins', {}) if r['truth'] else {}
        tp = {tuple(sorted((k, v))) for k, v in truth.items()}
        hit_top = sum(1 for i in top for m in H[i]['merge'] if tuple(sorted(m)) in tp)
        hit_rand = np.mean([sum(1 for m in H[i]['merge'] if tuple(sorted(m)) in tp) for i in range(1, len(H))]) * len(top)
        fam = Counter(f for h in held[1:] for f in h['fam'][:1])
        ab_top = np.mean([H[i]['ab'] for i in top])
        best = top[int(np.argmin([h['out'] for h in held[1:]]))]
        lines[tname] = (f'{tname}: identity out fold1 {base1:.2f} / fold2 {base2:.2f}; best random fold1 {base1 - max(gain1):.2f}; '
                        f'top-15 held-out gain median {np.median(gain2):+.2f} (best {max(gain2):+.2f}; {np.mean(np.array(gain2) > 0):.2f} positive); '
                        f'share of all hypotheses improving fold1 {np.mean(allg > 0):.2f}; A/B undo in top {ab_top:.2f}; '
                        f'held-out NN family of survivors {dict(fam.most_common(3))}; '
                        f'most enriched merges {[(("".join(k[0]) + "=" + "".join(k[1])), round(e, 1)) for e, k in enr[:4]]}'
                        + (f'; planted twin merges in top-15 {hit_top} vs {hit_rand:.1f} expected' if tp else '')
                        + f'; best held-out hypothesis {H[best]["merge"]} fuse {H[best]["fuse"]} ab {H[best]["ab"]}')
        print(lines[tname])
    L.save('c3_summary.json', lines)
    for k, rid in (('LAT_PL', 'V-48.3a'), ('VOY_MK2', 'V-48.3b'), ('VOY_ZL', 'V-48.3c'), ('VOY_IT', 'V-48.3d')):
        if k in lines:
            L.row(OUT, rid, f'Random normalisation search on {k}: {NH} random hypotheses (0-6 unit merges among units with n >= 50, 0-3 bigram fusions, A/B undo on/off) scored by map out-score on alternate pages (one 8k window); top 15 re-scored on held-out pages',
                  lines[k], 'see cycle verdict')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report(L.load('c3_results.json'), None)
    else:
        main()
