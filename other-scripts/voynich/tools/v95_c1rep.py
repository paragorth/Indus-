"""v95 cycle 1, re-test: the top-20 hypotheses of a source corpus (selected on train folios by excess over twins) applied with
FROZEN weights (fitted on the source's train half) to the held-out half of another corpus / transcription, against 4 twins.
usage: python3 v95_c1rep.py SRC TGT [TGT ...]
"""
import sys, json, os, numpy as np
import v95_lib as L, v95_c1 as C, v95_sum1 as S


def frozen_eval(G, hyp, c):
    P = C.proj(hyp['groups'])
    D2, D1, n1, A1, B1 = G[(hyp['key'], 1)]
    if n1 < 30: return np.nan
    return float(C.robust_G(D2, D1, (P @ np.array(c)).astype(np.float32)))


def run(src, tgts):
    top, _ = S.table(src)
    H = C.hypotheses(json.load(open(os.path.join(L.CK, 'c1_%s.json' % src)))['nh'])
    out = {}
    for t in tgts:
        Cc = L.corpus(t)
        Gs = [C.grams(Cc)] + [C.grams(L.twin(Cc, 9600 + s)) for s in range(C.NTWIN)]
        res = []
        for x in top:
            h = H[x['i']]
            v = [frozen_eval(g, h, x['c']) for g in Gs]
            sd = np.nanstd(v[1:], ddof=1) + 0.02
            res.append(dict(i=x['i'], key=h['key'], G=v[0], tw=float(np.nanmean(v[1:])), ex=v[0] - np.nanmean(v[1:]),
                            z=(v[0] - np.nanmean(v[1:])) / sd))
        out[t] = res
        zs = [r['z'] for r in res]; ex = [r['ex'] for r in res]
        print('%s -> %-6s top20 frozen: z>=3 %d/20, ex>=0.3&z>=3 %d/20, median ex %.2f, median G %.2f' % (
            src, t, sum(z >= 3 for z in zs), sum((z >= 3) and (e >= 0.3) for z, e in zip(zs, ex)), np.nanmedian(ex),
            np.nanmedian([r['G'] for r in res])))
    json.dump(out, open(os.path.join(L.CK, 'c1rep_%s.json' % src), 'w'), default=float)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2:])
