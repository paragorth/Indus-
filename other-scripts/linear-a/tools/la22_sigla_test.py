#!/usr/bin/env python3
"""la22 out-of-corpus test: do frozen restorations match SigLA's independent re-reading?

Run only AFTER la22_freeze.py.  Checks the sha256 of the frozen predictions first.
Alignment: per document, lineara.xyz syllabic signs (reading order) vs SigLA syllabograms,
Needleman-Wunsch on sign codes.  Scored events:
  SUB  lineara reads X, SigLA reads Y != X at the aligned position -> rank of Y in the frozen
       int-mask prediction for that position (prediction made with X hidden)
  EDGE SigLA reads a sign where lineara's word touches a break '#' and has nothing -> rank of
       that sign in the frozen edge prediction
Nulls: (1) corpus frequency ranking; (2) Y permuted among events (prediction lists fixed, 10,000);
(3) the shuffled-tablet model's frozen predictions (words permuted across tablets within site).
"""
import os, sys, json, hashlib, re
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la22_data

OUT = os.path.join(la22_data.DATA, 'la22')
CK = la22_data.CK


def norm_code(c):
    m = re.match(r'(AB|A)0*(\d+)$', c or '')
    return m.group(1) + m.group(2) if m else c


def norm_name(n):
    return re.sub(r'[^A-Za-z0-9]', '', n).upper()


def nw(a, b, ms=2, mm=-1, gp=-1):
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1)); P = np.zeros((n + 1, m + 1), dtype=np.int8)
    S[:, 0] = gp * np.arange(n + 1); S[0, :] = gp * np.arange(m + 1); P[1:, 0] = 1; P[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (ms if a[i - 1] == b[j - 1] else mm)
            u = S[i - 1, j] + gp; l = S[i, j - 1] + gp
            if d >= u and d >= l: S[i, j] = d; P[i, j] = 0
            elif u >= l: S[i, j] = u; P[i, j] = 1
            else: S[i, j] = l; P[i, j] = 2
    i, j = n, m; out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and P[i, j] == 0: out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or P[i, j] == 1): out.append((i - 1, None)); i -= 1
        else: out.append((None, j - 1)); j -= 1
    return out[::-1]


def rank_of(top, code):
    for k, (c, p) in enumerate(top):
        if c == code: return k + 1, p
    return 99, 0.0


def main():
    txt = open(os.path.join(OUT, 'la22_predictions.json')).read()
    h = hashlib.sha256(txt.encode()).hexdigest()
    want = open(os.path.join(OUT, 'la22_predictions.sha256')).read().split()[0]
    assert h == want, 'frozen predictions changed!'
    F = json.loads(txt)
    S = json.load(open(os.path.join(CK, 'sigla.json')))
    D = la22_data.load()
    sname = {norm_name(k): k for k in S}
    pos = {(p['doc'], p['tok'], p['j'], p['mode']): p for p in F['pos']}
    posS = {(p['doc'], p['tok'], p['j'], p['mode']): p for p in F.get('pos_shuf', [])}
    edge = {(e['doc'], e['tok'], e['mode']): e for e in F['edge']}
    edgeS = {(e['doc'], e['tok'], e['mode']): e for e in F.get('edge_shuf', [])}
    freq = Counter(c for d in D for t in d['toks'] if t['t'] == 'w' for c in t['c'])
    frank = [c for c, _ in freq.most_common()]
    subs = []; edges = []; n_al = 0; n_match = 0; n_docs = 0
    for d in D:
        k = sname.get(norm_name(d['id']))
        if not k: continue
        n_docs += 1
        a = []  # lineara signs: (code, tok, j)
        for ti, t in enumerate(d['toks']):
            if t['t'] == 'w':
                for j, c in enumerate(t['c']): a.append((c, ti, j))
        b = [norm_code(o['code']) for o in S[k]['occ'] if o['role'] == 'syllabogram']
        bsure = [o['sure'] for o in S[k]['occ'] if o['role'] == 'syllabogram']
        if not a or not b: continue
        al = nw([x[0] for x in a], b)
        T = d['toks']
        for idx, (i, j) in enumerate(al):
            if i is not None and j is not None:
                n_al += 1
                if a[i][0] == b[j]: n_match += 1; continue
                key = (d['id'], a[i][1], a[i][2], 'int')
                if key in pos:
                    subs.append({'doc': d['id'], 'tok': a[i][1], 'j': a[i][2], 'X': a[i][0], 'Y': b[j],
                                 'sure': bsure[j], 'pred': pos[key]['top10'],
                                 'pred_shuf': posS.get(key, {}).get('top10')})
            elif i is None and j is not None:
                # SigLA sign with no lineara counterpart: is it at a break edge of a lineara word?
                prev = next((al[q][0] for q in range(idx - 1, -1, -1) if al[q][0] is not None), None)
                nxt = next((al[q][0] for q in range(idx + 1, len(al)) if al[q][0] is not None), None)
                cand = None
                if prev is not None:
                    ti, jj = a[prev][1], a[prev][2]
                    if jj == len(T[ti]['c']) - 1 and ti + 1 < len(T) and T[ti + 1]['t'] == 'gap':
                        # only the sign directly after the fragment (first inserted sign)
                        if idx - 1 >= 0 and al[idx - 1][0] == prev:
                            cand = (d['id'], ti, 'R')
                if cand is None and nxt is not None:
                    ti, jj = a[nxt][1], a[nxt][2]
                    if jj == 0 and ti > 0 and T[ti - 1]['t'] == 'gap':
                        if idx + 1 < len(al) and al[idx + 1][0] == nxt:
                            cand = (d['id'], ti, 'L')
                if cand and cand in edge:
                    edges.append({'doc': cand[0], 'tok': cand[1], 'mode': cand[2], 'Y': b[j], 'sure': bsure[j],
                                  'pred': edge[cand]['top10'], 'pred_shuf': edgeS.get(cand, {}).get('top10'),
                                  'fragment': edge[cand]['fragment_tr']})
    res = {'docs_aligned': n_docs, 'aligned_pairs': n_al, 'identical': n_match, 'n_sub': len(subs), 'n_edge': len(edges)}
    pairc = Counter((s['X'], s['Y']) for s in subs)
    res['top_sub_pairs'] = [[x, y, n] for (x, y), n in pairc.most_common(15)]
    rng = np.random.default_rng(22)

    def score(ev, label, field='pred'):
        ev = [e for e in ev if e.get(field)]
        if not ev: return {'n': 0}
        r = np.array([rank_of(e[field], e['Y'])[0] for e in ev])
        fr = np.array([(frank.index(e['Y']) + 1) if e['Y'] in frank else 99 for e in ev])
        out = {'n': len(ev), 'top1': int((r == 1).sum()), 'top5': int((r <= 5).sum()), 'top10': int((r <= 10).sum()),
               'freq_top1': int((fr == 1).sum()), 'freq_top5': int((fr <= 5).sum()), 'freq_top10': int((fr <= 10).sum())}
        Ys = [e['Y'] for e in ev]; hits5 = []; hits1 = []
        for it in range(10000):
            perm = rng.permutation(len(Ys))
            rr = np.array([rank_of(ev[q][field], Ys[perm[q]])[0] for q in range(len(ev))])
            hits1.append((rr == 1).sum()); hits5.append((rr <= 5).sum())
        hits1 = np.array(hits1); hits5 = np.array(hits5)
        out['perm_top1_mean'] = round(float(hits1.mean()), 2); out['perm_top1_p'] = round(float((hits1 >= out['top1']).mean()), 4)
        out['perm_top5_mean'] = round(float(hits5.mean()), 2); out['perm_top5_p'] = round(float((hits5 >= out['top5']).mean()), 4)
        return out

    nonsys = [s for s in subs if pairc[(s['X'], s['Y'])] <= 2]
    res['SUB_all'] = score(subs, 'sub')
    res['SUB_nonsystematic'] = score(nonsys, 'sub')
    res['SUB_all_shufmodel'] = score(subs, 'sub', 'pred_shuf')
    res['SUB_nonsys_shufmodel'] = score(nonsys, 'sub', 'pred_shuf')
    # does the model prefer SigLA's Y over lineara's own X (the reading it was trained around)?
    pref = [rank_of(s['pred'], s['Y'])[1] > rank_of(s['pred'], s['X'])[1] for s in subs]
    res['SUB_prefers_Y_over_X'] = [int(sum(pref)), len(pref)]
    res['EDGE'] = score(edges, 'edge')
    res['EDGE_shufmodel'] = score(edges, 'edge', 'pred_shuf')
    res['edge_events'] = [[e['doc'], e['mode'], '-'.join(e['fragment']), e['Y'], rank_of(e['pred'], e['Y'])[0],
                           e['pred'][0][0], e['pred'][0][1]] for e in edges]
    res['sub_events_nonsys'] = [[s['doc'], s['X'], s['Y'], rank_of(s['pred'], s['Y'])[0], rank_of(s['pred'], s['X'])[0],
                                 s['pred'][0][0]] for s in nonsys]
    json.dump(res, open(os.path.join(OUT, 'la22_sigla_test.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('edge_events', 'sub_events_nonsys')}, indent=1))


if __name__ == '__main__':
    main()
