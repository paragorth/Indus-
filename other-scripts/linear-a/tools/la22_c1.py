#!/usr/bin/env python3
"""la22 cycle 1: calibrate a random-ensemble sign restorer by masking known signs.

usage: python3 la22_c1.py CORPUS [seed]
  CORPUS = LA | LAshuf (words permuted across tablets within site) | LB (DAMOS, LA-size sample)
Outputs (git-ignored checkpoint dir): la22_ckpt/c1_<CORPUS>.json, c1_<CORPUS>_P.npz
"""
import sys, os, json, time, hashlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la22_data, la22_lb
from la22_engine import Corpus, EXPERTS, GROUPS, feats_for_nn

torch.set_num_threads(1)
CK = la22_data.CK
EPS = 1e-6


def load_corpus(name, seed):
    rng = np.random.default_rng(seed)
    if name.startswith('LA'):
        D = la22_data.load()
        if name == 'LAshuf':
            D = json.loads(json.dumps(D))
            by_site = {}
            for di, d in enumerate(D):
                for ti, t in enumerate(d['toks']):
                    if t['t'] in ('w', 'N', 'L'): by_site.setdefault((d['site'], t['t']), []).append((di, ti))
            for site, slots in sorted(by_site.items()):
                words = [D[di]['toks'][ti] for di, ti in slots]
                perm = rng.permutation(len(words))
                for (di, ti), k in zip(slots, perm): D[di]['toks'][ti] = json.loads(json.dumps(words[k]))
        return D
    if name.startswith('LB'):
        D = la22_lb.load()
        target = 5460  # LA syllabic sign tokens
        order = rng.permutation(len(D)); out = []; n = 0
        for k in order:
            d = D[k]; out.append(d)
            n += sum(len(t['c']) for t in d['toks'] if t['t'] == 'w' and not t.get('dot'))
            if n >= target: break
        return out
    raise ValueError(name)


def doc_fold(doc_id):
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 2


def train_nn(C, insts, folds, use_doc, seed, nfold=5):
    """returns prob matrix (n, nV) from 5-fold-by-document MLPs."""
    n = len(insts); nV = C.nV
    X = np.array([feats_for_nn(C, it) for it in insts])
    sites = sorted({d['site'] for d in C.docs}); sidx = {s: k for k, s in enumerate(sites)}
    S = np.array([sidx[C.docs[it[0]]['site']] for it in insts])
    y = np.array([C.ix[C.docs[it[0]]['toks'][it[1]]['c'][it[2]]] for it in insts])
    if use_doc:
        B = np.zeros((n, nV), dtype=np.float32)
        for k, (di, ti, j, m) in enumerate(insts):
            v = C.docU[di].copy()
            for c in C.docs[di]['toks'][ti]['c']:
                if c in C.ix: v[C.ix[c]] -= 1
            v = np.maximum(v, 0)
            if v.sum() > 0: B[k] = v / v.sum()
    dk = np.array([int(hashlib.md5((C.docs[it[0]]['id'] + str(seed)).encode()).hexdigest(), 16) % nfold for it in insts])
    P = np.zeros((n, nV), dtype=np.float32)
    for f in range(nfold):
        tr = dk != f; te = dk == f
        torch.manual_seed(seed * 100 + f)
        emb = torch.nn.Embedding(nV + 3, 24); semb = torch.nn.Embedding(len(sites), 6)
        memb = torch.nn.Embedding(3, 4); lemb = torch.nn.Embedding(7, 4)
        din = 24 * 4 + 6 + 4 + 4 + (nV if use_doc else 0)
        mlp = torch.nn.Sequential(torch.nn.Linear(din, 96), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(96, nV))
        params = list(emb.parameters()) + list(semb.parameters()) + list(memb.parameters()) + list(lemb.parameters()) + list(mlp.parameters())
        opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-4)
        Xt = torch.tensor(X); St = torch.tensor(S); yt = torch.tensor(y)
        Bt = torch.tensor(B) if use_doc else None

        def fwd(idx, train):
            mlp.train(train)
            h = [emb(Xt[idx, k]) for k in range(4)] + [semb(St[idx]), memb(Xt[idx, 4]), lemb(Xt[idx, 5])]
            if use_doc: h.append(Bt[idx])
            return mlp(torch.cat(h, 1))
        tri = np.where(tr)[0]
        rs = np.random.default_rng(seed + f)
        for ep in range(25):
            rs.shuffle(tri)
            for b in range(0, len(tri), 128):
                idx = torch.tensor(tri[b:b + 128])
                loss = torch.nn.functional.cross_entropy(fwd(idx, True), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tei = torch.tensor(np.where(te)[0])
            P[te] = torch.softmax(fwd(tei, False), 1).numpy()
    return P


def metrics(P, y, nb=10):
    n = len(y)
    if n == 0: return {}
    top = np.argsort(-P, 1)[:, :5]
    t1 = (top[:, 0] == y).mean(); t5 = (top == y[:, None]).any(1).mean()
    ll = -np.log(P[np.arange(n), y] + 1e-12).mean()
    conf = P.max(1); acc = (top[:, 0] == y)
    bins = np.minimum((conf * nb).astype(int), nb - 1); ece = 0.0
    for b in range(nb):
        m = bins == b
        if m.any(): ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return {'n': int(n), 'top1': round(float(t1), 4), 'top5': round(float(t5), 4), 'logloss': round(float(ll), 4),
            'ece': round(float(ece), 4), 'mean_conf': round(float(conf.mean()), 4)}


def combine(LP, w, eps):
    """LP: (n, E, V) log probs; w: (E,) weights -> probs (n, V)."""
    z = np.einsum('nev,e->nv', LP, w)
    z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return (1 - eps) * p + eps / p.shape[1]


def random_ensemble(LP, y, foldA, foldB, names, nrand, rng, allowed=None, topk=25):
    E = len(names)
    allow = np.ones(E, bool) if allowed is None else np.array([nm in allowed for nm in names])
    base = names.index('FREQ') if 'FREQ' in names else names.index('GLOB')
    cands = []
    lA = []
    for r in range(nrand):
        m = (rng.random(E) < 0.5) & allow; m[base] = True
        w = rng.exponential(1.0, E) * m
        w = w / max(w.sum(), 1e-9) * rng.uniform(0.6, 2.5)
        eps = 10 ** rng.uniform(-4, -1)
        P = combine(LP[foldA], w, eps)
        ll = -np.log(P[np.arange(len(P)), y[foldA]] + 1e-12).mean()
        cands.append((w, eps)); lA.append(ll)
    order = np.argsort(lA)[:topk]
    PB = np.mean([combine(LP[foldB], *cands[k]) for k in order], 0)
    wbar = np.mean([cands[k][0] for k in order], 0)
    return PB, wbar, np.array(lA), order


def main():
    name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nrand = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    t0 = time.time()
    D = load_corpus(name, seed)
    C = Corpus(D)
    insts = C.instances()
    if len(insts) > 16000:
        rs = np.random.default_rng(seed); insts = [insts[k] for k in sorted(rs.choice(len(insts), 16000, replace=False))]
    y = np.array([C.ix[C.docs[it[0]]['toks'][it[1]]['c'][it[2]]] for it in insts])
    names = [e for e in EXPERTS if e not in ('NNL', 'NND')] + ['NNL', 'NND']
    LP = np.zeros((len(insts), len(names), C.nV), dtype=np.float32)
    sm = {'add': 0.1, 'lam': 2.0, 'lam3': 1.0, 'lamlex': 0.5, 'lamdoc': 0.5, 'lamdocs': 5.0}
    for k, it in enumerate(insts):
        E = C.experts(it, sm)
        for e, nm in enumerate(names[:-2]):
            LP[k, e] = np.log(E[nm] + EPS)
    print(name, 'experts done', len(insts), round(time.time() - t0), flush=True)
    PNL = np.mean([train_nn(C, insts, None, False, s) for s in (seed, seed + 1)], 0)
    PND = np.mean([train_nn(C, insts, None, True, s) for s in (seed, seed + 1)], 0)
    LP[:, -2] = np.log(PNL + EPS); LP[:, -1] = np.log(PND + EPS)
    print(name, 'nn done', round(time.time() - t0), flush=True)
    modes = np.array([it[3] for it in insts])
    fold = np.array([doc_fold(C.docs[it[0]]['id']) for it in insts])
    rng = np.random.default_rng(seed + 7)
    res = {'corpus': name, 'n_inst': len(insts), 'nV': C.nV, 'n_docs': len(D), 'experts': names, 'by_mode': {}}
    # full ensemble, 2-fold (select on one half of the documents, score the other half)
    Pens = np.zeros((len(insts), C.nV), dtype=np.float32); W = []
    for fa in (0, 1):
        A = fold == fa; B = fold != fa
        PB, wbar, lA, order = random_ensemble(LP, y, A, B, names, nrand, rng)
        Pens[B] = PB; W.append(wbar)
    res['mean_weights_selected'] = dict(zip(names, np.round(np.mean(W, 0), 3).tolist()))
    # learned weights by gradient descent (comparison)
    def learned(allowed=None):
        Pl = np.zeros_like(Pens)
        for fa in (0, 1):
            A = fold == fa; B = fold != fa
            lp = torch.tensor(LP[A]); yy = torch.tensor(y[A])
            msk = torch.tensor([1.0 if (allowed is None or nm in allowed) else 0.0 for nm in names])
            w = torch.zeros(len(names), requires_grad=True)
            opt = torch.optim.Adam([w], lr=0.05)
            for it in range(300):
                z = torch.einsum('nev,e->nv', lp, torch.nn.functional.softplus(w) * msk)
                loss = torch.nn.functional.cross_entropy(z, yy)
                opt.zero_grad(); loss.backward(); opt.step()
            ww = (torch.nn.functional.softplus(w) * msk).detach().numpy()
            Pl[B] = combine(LP[B], ww, 1e-4)
        return Pl
    Plearn = learned()
    iF = names.index('FREQ'); iT = names.index('TRI')
    Pfreq = np.exp(LP[:, iF]); Pfreq /= Pfreq.sum(1, keepdims=True)
    Pmk = np.exp(LP[:, iT]); Pmk /= Pmk.sum(1, keepdims=True)
    for md in ('int', 'R', 'L', 'all'):
        m = (modes == md) if md != 'all' else np.ones(len(insts), bool)
        res['by_mode'][md] = {'ensemble': metrics(Pens[m], y[m]), 'learned': metrics(Plearn[m], y[m]),
                              'freq': metrics(Pfreq[m], y[m]), 'markov': metrics(Pmk[m], y[m])}
        for e, nm in enumerate(names):
            if nm in ('FREQ', 'TRI'): continue
            Pe = np.exp(LP[m, e]); Pe /= Pe.sum(1, keepdims=True)
            res['by_mode'][md]['single_' + nm] = metrics(Pe, y[m])['top1']
    # group ablation: learned weights without each group; and each group alone (+FREQ)
    res['ablation'] = {}
    for g, mem in GROUPS.items():
        if g == 'base': continue
        Pw = learned([nm for nm in names if nm not in mem])
        Po = learned(mem + ['FREQ'])
        res['ablation'][g] = {}
        for md in ('int', 'R', 'L'):
            m = modes == md
            res['ablation'][g][md] = {'without': metrics(Pw[m], y[m])['top1'], 'only': metrics(Po[m], y[m])['top1'],
                                      'full': res['by_mode'][md]['learned']['top1']}
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(CK, 'c1_%s.json' % name), 'w'), indent=1)
    np.savez_compressed(os.path.join(CK, 'c1_%s_P.npz' % name), Pens=Pens.astype(np.float16), y=y, modes=modes)
    print(json.dumps({md: res['by_mode'][md]['ensemble'] for md in res['by_mode']}), flush=True)


if __name__ == '__main__':
    main()
