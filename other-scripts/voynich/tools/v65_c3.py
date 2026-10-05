"""v65 cycle 3: does the text's carry-over agree with PHYSICAL adjacency evidence?

Claims (each names a directed page pair that physical evidence says was once adjacent):
  E1 f78v->f81r  one water-flow picture spans both halves of bifolio 78/81, so it was the centre of Q13
                 (Zandbergen, voynich.nu 'sp_origin'; L. F. Davis 2025 Yale talk, summarised at ciphermysteries.com 11 Oct 2025)
  E2 f33v->f40r  drawing crosses the centre of bifolio 33/40 and blue paint leached onto the pages now opposite
                 (voynich.nu sp_origin; Davis 2025)
  E3 f10v->f15r  red paint spatters suggest bifolio 10/15 was central in Q2 (Pelling, ciphermysteries 27 Mar 2009; disputed)
  E4 current openings with wet contact / paint transfers: f2v->f3r (stem ink + paint, strong), f3v->f4r, f5v->f6r,
     f19v->f20r, f14v->f15r (Pelling 2009; disputed)
  E5 quire order fixed by the quire signatures (voynich.nu): seams between consecutive quires vs 10,000 random quire orders.
Statistic: pct of W(a->b) among all directed pairs of text pages on different leaves of the same quire
(1 = the strongest seam possible there); config-level: share of the top-5% arrangements of the quire that satisfy
the claim (E1-E3), against its prior (1 / (2 x bifolios)).
Calibration: Isidore and Konrad (continuous, opaque + padding) poured into a 'true' order that satisfies E1-E3
(78/81 central in Q13, 33/40 central in Q5, 10/15 central in Q2; binding elsewhere), so every claim is truly adjacent;
and the same texts poured in binding order (E1-E3 false) as the negative.
"""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v65_lib as L

quires, leaves = L.structure()
QD = dict(quires)
CENTRE = {'M': 3, 'E': 0, 'B': 1}      # bifolio index (0 = outermost now) claimed central
CLAIMS = {'E1': ((78, 'v'), (81, 'r'), 'M'), 'E2': ((33, 'v'), (40, 'r'), 'E'), 'E3': ((10, 'v'), (15, 'r'), 'B'),
          'E4a': ((2, 'v'), (3, 'r'), 'A'), 'E4b': ((3, 'v'), (4, 'r'), 'A'), 'E4c': ((5, 'v'), (6, 'r'), 'A'),
          'E4d': ((19, 'v'), (20, 'r'), 'C'), 'E4e': ((14, 'v'), (15, 'r'), 'B')}


def claimed_order():
    seq = []
    for qn, bifs in quires:
        nb = len(bifs); order = list(range(nb))
        if qn in CENTRE:
            c = CENTRE[qn]; order.remove(c)
            # keep lost (empty) bifolios innermost
            order.append(c)
        seq += L.seq_for(bifs, tuple(order), (0,) * nb)
    return [s for s in seq if s != L.GAP]


def evaluate(pages, meta, feat, cls=None):
    keys = sorted(k for k in pages if sum(len(x) for x in pages[k]) >= L.MIN_WORDS and k in meta)
    kidx = {k: i for i, k in enumerate(keys)}
    groups = [str(meta[k]['sec']) + str(meta[k]['lang']) for k in keys]
    Js, _ = L.carry_matrices(pages, keys, cls=cls, feat=feat)
    W = L.centre(Js, groups)
    res = {}
    for cn, (a, b, qn) in CLAIMS.items():
        if a not in kidx or b not in kidx:
            res[cn] = None; continue
        qleaves = set(x for bf in QD[qn] for x in bf[1:] if x)
        cand = [kidx[k] for k in keys if k[0] in qleaves]
        vals = [W[x, y] for x in cand for y in cand if keys[x][0] != keys[y][0]]
        v = W[kidx[a], kidx[b]]
        pct = float((np.array(vals) < v).mean())
        r = {'pct': pct, 'W': float(v)}
        if cn in ('E1', 'E2', 'E3'):
            sp = L.QuireSpace(QD[qn], kidx)
            sc = sp.scores(W)
            top = np.argsort(-sc)[:max(1, len(sc) // 20)]
            c = CENTRE[qn]
            def ok(ci):
                o, f = sp.configs[ci]
                real = [i for i in o if not (QD[qn][i][1] is None and QD[qn][i][2] is None)]
                return real[-1] == c and f[c] == 0
            r['top5_share'] = float(np.mean([ok(i) for i in top]))
            r['prior'] = float(np.mean([ok(i) for i in range(len(sp.configs))]))
            r['best_ok'] = bool(ok(int(np.argmax(sc))))
        res[cn] = r
    # E5: quire order
    qfirst, qlast = [], []
    for qn, bifs in quires:
        seq = [k for k in L.seq_for(bifs, tuple(range(len(bifs))), (0,) * len(bifs)) if k != L.GAP and k in kidx]
        if seq:
            qfirst.append(kidx[seq[0]]); qlast.append(kidx[seq[-1]])
    nq = len(qfirst)
    def qscore(perm):
        return sum(W[qlast[perm[i]], qfirst[perm[i + 1]]] for i in range(nq - 1))
    obs = qscore(list(range(nq)))
    rng = random.Random(5)
    nulls = []
    for _ in range(10000):
        p = list(range(nq)); rng.shuffle(p); nulls.append(qscore(p))
    res['E5'] = {'obs': float(obs), 'pct': float((np.array(nulls) < obs).mean())}
    return res


def main():
    vp, vmeta = L.voynich_pages('ZL3b'); ip, imeta = L.voynich_pages('IT2a')
    isid = L.isidore_words()[5000:]; kon = [w for e in L.konrad_entries() for w in e]
    co = claimed_order()
    runs = {
        'VOY_ZL|word': (vp, vmeta, 'word'), 'VOY_ZL|skel': (vp, vmeta, 'skel'),
        'VOY_IT|word': (ip, imeta, 'word'), 'VOY_IT|skel': (ip, imeta, 'skel'),
    }
    for s in range(3):
        runs[f'CAL_isid_claimed_s{s}|skel'] = (L.pour(vp, isid[s * 9000:], 'flow', 10 + s, quires, order=co), vmeta, 'skel')
        runs[f'CAL_kon_claimed_s{s}|skel'] = (L.pour(vp, kon[s * 3000:] + kon[:s * 3000], 'flow', 20 + s, quires, order=co), vmeta, 'skel')
        runs[f'NEG_isid_binding_s{s}|skel'] = (L.pour(vp, isid[s * 9000:], 'flow', 30 + s, quires), vmeta, 'skel')
    runs['NULL_markov|word'] = (L.markov_pages(vp, vmeta, 7), vmeta, 'word')
    out = {}
    for name, (p, m, feat) in runs.items():
        r = evaluate(p, m, feat)
        out[name] = r
        line = ' '.join(f"{k}:{v['pct']:.2f}" + (f"/{v['top5_share']:.2f}({v['prior']:.2f})" if v and 'top5_share' in v else '')
                        for k, v in r.items() if v)
        print(name, line, flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
