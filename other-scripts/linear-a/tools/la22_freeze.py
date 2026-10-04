#!/usr/bin/env python3
"""la22 freeze: predict every real Linear A lacuna edge and every sign position, then hash.

Run BEFORE la22_sigla_test.py.  Writes data/la22/la22_predictions.json and prints its sha256.
  (a) edge: each word touching a break '#': the sign just beyond the break (R: after the
      fragment, L: before it), top-10 with probabilities, plus a lexicon estimate of P(word complete)
  (b) pos: every sign position in every word, three masks (int, R, L), top-10, out-of-fold
      (selection of the random ensemble on the other half of the documents, LOO-doc counts)
  (c) numbers: sections closed by KU-RO / PO-TO-KU-RO that contain a break: missing amount
"""
import sys, os, json, time, hashlib, datetime
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la22_data
from la22_engine import Corpus, EXPERTS, feats_for_nn
from la22_c1 import random_ensemble, combine, doc_fold, metrics
from la22_c2 import sections

torch.set_num_threads(1)
OUT = os.path.join(la22_data.DATA, 'la22')
EPS = 1e-6


def nn_full(C, train, pred, use_doc, seed):
    nV = C.nV
    sites = sorted({d['site'] for d in C.docs}); sidx = {s: k for k, s in enumerate(sites)}
    def build(insts, with_y):
        X = torch.tensor([feats_for_nn(C, it) for it in insts])
        S = torch.tensor([sidx[C.docs[it[0]]['site']] for it in insts])
        B = np.zeros((len(insts), nV), dtype=np.float32)
        if use_doc:
            for k, (di, ti, j, m) in enumerate(insts):
                v = C.docU[di].copy()
                for c in C.docs[di]['toks'][ti]['c']:
                    if c in C.ix: v[C.ix[c]] -= 1
                v = np.maximum(v, 0)
                if v.sum() > 0: B[k] = v / v.sum()
        y = torch.tensor([C.ix[C.docs[it[0]]['toks'][it[1]]['c'][it[2]]] for it in insts]) if with_y else None
        return X, S, torch.tensor(B), y
    Xt, St, Bt, yt = build(train, True); Xp, Sp, Bp, _ = build(pred, False)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(nV + 3, 24); semb = torch.nn.Embedding(len(sites), 6)
    memb = torch.nn.Embedding(3, 4); lemb = torch.nn.Embedding(7, 4)
    din = 24 * 4 + 6 + 4 + 4 + (nV if use_doc else 0)
    mlp = torch.nn.Sequential(torch.nn.Linear(din, 96), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(96, nV))
    params = [p for m in (emb, semb, memb, lemb, mlp) for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-4)
    def fwd(X, S, B, train_):
        mlp.train(train_)
        h = [emb(X[:, k]) for k in range(4)] + [semb(S), memb(X[:, 4]), lemb(X[:, 5])]
        if use_doc: h.append(B)
        return mlp(torch.cat(h, 1))
    idx = np.arange(len(train)); rs = np.random.default_rng(seed)
    for ep in range(25):
        rs.shuffle(idx)
        for b in range(0, len(idx), 128):
            ii = torch.tensor(idx[b:b + 128])
            loss = torch.nn.functional.cross_entropy(fwd(Xt[ii], St[ii], Bt[ii], True), yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.softmax(fwd(Xp, Sp, Bp, False), 1).numpy()


def topk(C, p, k=10):
    o = np.argsort(-p)[:k]
    return [[C.V[i], round(float(p[i]), 4)] for i in o]


def shuffle_tablets(D, seed=7):
    """words, numbers and logograms permuted across tablets within site; words keep 'orig'."""
    D = json.loads(json.dumps(D))
    for di, d in enumerate(D):
        for ti, t in enumerate(d['toks']):
            if t['t'] == 'w': t['orig'] = [d['id'], ti]
    rng = np.random.default_rng(seed); by = {}
    for di, d in enumerate(D):
        for ti, t in enumerate(d['toks']):
            if t['t'] in ('w', 'N', 'L'): by.setdefault((d['site'], t['t']), []).append((di, ti))
    for key, slots in sorted(by.items()):
        toks = [D[di]['toks'][ti] for di, ti in slots]; perm = rng.permutation(len(toks))
        for (di, ti), k in zip(slots, perm): D[di]['toks'][ti] = toks[k]
    return D


def run(D, edge_keys=None, seed=2022):
    """edge_keys: for the shuffled corpus, list of (orig doc, orig tok, mode) to predict."""
    t0 = time.time()
    C = Corpus(D)
    names = [e for e in EXPERTS if e not in ('NNL', 'NND')] + ['NNL', 'NND']
    sm = {'add': 0.1, 'lam': 2.0, 'lam3': 1.0, 'lamlex': 0.5, 'lamdoc': 0.5, 'lamdocs': 5.0}
    cal = C.instances()
    edge = []
    if edge_keys is None:
        for di, d in enumerate(D):
            T = d['toks']
            for ti, t in enumerate(T):
                if t['t'] != 'w': continue
                n = len(t['c'])
                if ti + 1 < len(T) and T[ti + 1]['t'] == 'gap': edge.append((di, ti, n, 'R'))
                if ti > 0 and T[ti - 1]['t'] == 'gap': edge.append((di, ti, -1, 'L'))
    else:
        where = {}
        for di, d in enumerate(D):
            for ti, t in enumerate(d['toks']):
                if t['t'] == 'w': where[tuple(t['orig'])] = (di, ti)
        for od, ot, mode in edge_keys:
            di, ti = where[(od, ot)]; n = len(D[di]['toks'][ti]['c'])
            edge.append((di, ti, n if mode == 'R' else -1, mode))
    allI = cal + edge
    LP = np.zeros((len(allI), len(names), C.nV), dtype=np.float32)
    for k, it in enumerate(allI):
        E = C.experts(it, sm)
        for e, nm in enumerate(names[:-2]): LP[k, e] = np.log(E[nm] + EPS)
    print('experts', len(cal), len(edge), round(time.time() - t0), flush=True)
    from la22_c1 import train_nn
    for e, use_doc in ((len(names) - 2, False), (len(names) - 1, True)):
        Pc = np.mean([train_nn(C, cal, None, use_doc, s) for s in (1, 2)], 0)
        Pe = np.mean([nn_full(C, cal, edge, use_doc, s) for s in (1, 2)], 0)
        LP[:len(cal), e] = np.log(Pc + EPS); LP[len(cal):, e] = np.log(Pe + EPS)
    print('nn', round(time.time() - t0), flush=True)
    y = np.array([C.ix[D[it[0]]['toks'][it[1]]['c'][it[2]]] for it in cal])
    fold = np.array([doc_fold(D[it[0]]['id']) for it in cal])
    rng = np.random.default_rng(seed)
    LPc = LP[:len(cal)]
    Pcal = np.zeros((len(cal), C.nV), dtype=np.float32); sel = []
    for fa in (0, 1):
        A = fold == fa; B = ~A
        PB, wbar, lA, order = random_ensemble(LPc, y, A, B, names, 4000, rng)
        Pcal[B] = PB; sel.append(list(random_ensemble.last))
    Pedge = np.mean([combine(LP[len(cal):], *c) for s_ in sel for c in s_], 0)
    modes = np.array([it[3] for it in cal])
    cal_metrics = {md: metrics(Pcal[modes == md], y[modes == md]) for md in ('int', 'R', 'L')}
    return C, cal, edge, Pcal, Pedge, y, cal_metrics


def main():
    t0 = time.time()
    D = la22_data.load()
    C, cal, edge, Pcal, Pedge, y, cal_metrics = run(D)
    lex = C.wordlist
    preds = {'made': datetime.datetime.utcnow().isoformat() + 'Z', 'corpus': 'lineara.xyz LinearAInscriptions.js',
             'calibration_out_of_fold': cal_metrics, 'edge': [], 'pos': [], 'numbers': []}
    for k, it in enumerate(edge):
        di, ti, j, mode = it; s = D[di]['toks'][ti]['c']; s_t = tuple(s)
        own = C.dc[di]['words']
        full = sum(max(c - own.get(w, 0), 0) for w, c in lex if w == s_t)
        if mode == 'R': ext = sum(max(c - own.get(w, 0), 0) for w, c in lex if len(w) > len(s) and w[:len(s)] == s_t)
        else: ext = sum(max(c - own.get(w, 0), 0) for w, c in lex if len(w) > len(s) and w[-len(s):] == s_t)
        preds['edge'].append({'doc': D[di]['id'], 'tok': ti, 'mode': mode, 'fragment': s,
                              'fragment_tr': D[di]['toks'][ti]['tr'], 'lex_complete': full, 'lex_extended': ext,
                              'top10': topk(C, Pedge[k])})
    for k, it in enumerate(cal):
        di, ti, j, mode = it
        preds['pos'].append({'doc': D[di]['id'], 'tok': ti, 'j': j, 'mode': mode,
                             'read': D[di]['toks'][ti]['c'][j], 'top10': topk(C, Pcal[k]),
                             'p_read': round(float(Pcal[k][y[k]]), 4)})
    # shuffled-tablet model (null): same pipeline on a corpus with tablets scrambled within site
    DS = shuffle_tablets(D)
    ekeys = [(D[di]['id'], ti, mode) for di, ti, j, mode in edge]
    CS, calS, edgeS, PcalS, PedgeS, yS, cmS = run(DS, edge_keys=ekeys)
    preds['calibration_shuffled_tablets'] = cmS
    preds['pos_shuf'] = []; preds['edge_shuf'] = []
    for k, it in enumerate(calS):
        di, ti, j, mode = it; o = DS[di]['toks'][ti]['orig']
        preds['pos_shuf'].append({'doc': o[0], 'tok': o[1], 'j': j, 'mode': mode, 'top10': topk(CS, PcalS[k])})
    for k, (od, ot, mode) in enumerate(ekeys):
        preds['edge_shuf'].append({'doc': od, 'tok': ot, 'mode': mode, 'top10': topk(CS, PedgeS[k])})
    # numbers: sections with a break inside
    for di, d in enumerate(D):
        T = d['toks']
        for tix, ents in sections(T):
            start = ents[0] if ents else tix
            lo = max([k for k in range(tix) if k < start and T[k]['t'] == 'N'] + [-1])
            has_gap = any(T[k]['t'] == 'gap' for k in range(lo + 1, tix))
            if not has_gap: continue
            vis = sum(T[k]['v'] for k in ents); tot = T[tix]['v']
            preds['numbers'].append({'doc': d['id'], 'total_tok': tix, 'total': tot, 'visible_sum': vis,
                                     'missing_pred': tot - vis, 'n_visible_entries': len(ents)})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'la22_predictions.json')
    txt = json.dumps(preds, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    open(p, 'w').write(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    open(os.path.join(OUT, 'la22_predictions.sha256'), 'w').write(h + '  la22_predictions.json\n')
    print('frozen', p, h, 'edges', len(edge), 'pos', len(cal), 'numbers', len(preds['numbers']),
          json.dumps(cal_metrics), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
