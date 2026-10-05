#!/usr/bin/env python3
"""LA-49 cycle 3: CONVERGENT FEATURES. Do independently trained models of Linear A grow the same internal
features? For every saved cycle-2 transformer: final-layer states at every token of every document;
FastICA (K components) per model; a component is UNIVERSAL if, over the same token positions, it has a
partner with |r| >= RMIN in at least FRAC of the other seeds. For universal features: the word types
and token kinds that switch them on, and a causal test (project the feature direction out of the final
state; held-out loss change where the feature fires vs elsewhere).
Controls: shuffled Linear A (no universal feature beyond token-kind ones should appear), planted corpus
(a feature must fire on the planted totals), Linear B / Ur III at 4x (features must line up with
commodities, totals, persons). usage: LA49_MAXLEN=128 la49_c3.py SRC_TAG CORPUS [K]"""
import sys, os, json, glob, collections, random
os.environ.setdefault('LA49_MAXLEN', '128')
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
import la45_common as C45
from sklearn.decomposition import FastICA

torch.set_num_threads(1)
SRC, CORP = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
RMIN, FRAC = float(os.environ.get('LA49_RMIN', '0.5')), float(os.environ.get('LA49_FRAC', '0.5'))


def load_models():
    out = []
    for f in sorted(glob.glob(os.path.join(L.CK, SRC, CORP + '_tf_*.json'))):
        r = json.load(open(f))
        cfg = r['cfg']
        out.append((cfg, f.replace('.json', '.pt'), set(r['train_docs'])))
    return out


def main():
    docs = L.corpus(CORP)
    voc = L.Vocab(docs)
    ch = L.chunks(docs, voc)
    V = len(voc.itos)
    # position table
    pos = [(ci, j) for ci, c in enumerate(ch) for j in range(len(c[1]))]
    kinds = np.array([voc.kind(ch[ci][1][j]) for ci, j in pos])
    toks = [ch[ci][2][j] for ci, j in pos]
    models = load_models()
    H, comps, mix = [], [], []
    for cfg, pt, trd in models:
        m = L.TF(V, cfg['d'], cfg['h'], cfg['nl'], 0.0)
        m.load_state_dict(torch.load(pt))
        m.eval()
        Z = []
        with torch.no_grad():
            for s in range(0, len(ch), 128):
                part = ch[s:s + 128]
                _, z = m(L.pad([c[1] for c in part]), hidden=True)
                for k, c in enumerate(part):
                    Z.append(z[k, :len(c[1])].numpy())
        Z = np.concatenate(Z)
        ica = FastICA(n_components=min(K, Z.shape[1]), random_state=0, max_iter=800, whiten='unit-variance')
        S = ica.fit_transform(Z)
        H.append((m, cfg, trd, ica, Z))
        comps.append(S)
    n = len(comps)
    # universality
    uni = []
    best_all = []
    for a in range(n):
        Sa = (comps[a] - comps[a].mean(0)) / (comps[a].std(0) + 1e-9)
        hits = np.zeros(Sa.shape[1])
        for b in range(n):
            if a == b:
                continue
            Sb = (comps[b] - comps[b].mean(0)) / (comps[b].std(0) + 1e-9)
            R = np.abs(Sa.T @ Sb) / len(Sa)
            hits += (R.max(1) >= RMIN)
            best_all.append(R.max(1))
        for c in range(Sa.shape[1]):
            uni.append((a, c, hits[c] / (n - 1)))
    universal = [(a, c, f) for a, c, f in uni if f >= FRAC]
    B = np.concatenate(best_all) if best_all else np.zeros(1)
    res = {'best_match_mean': float(B.mean()), 'best_match_q90': float(np.quantile(B, 0.9)),
           'frac_best_ge_0.7': float(np.mean(B >= 0.7)), 'frac_best_ge_0.5': float(np.mean(B >= 0.5)),
           'RMIN': RMIN, 'FRAC': FRAC}
    res.update({'corpus': CORP, 'n_models': n, 'K': K, 'n_comp': len(uni), 'n_universal': len(universal),
           'frac_universal': len(universal) / max(1, len(uni))})
    # describe universal features: sign so that the top 2 % tail is positive
    cnt = collections.Counter(t[1] if t[0] == 'T' else 'NUM' for t in toks)
    feats = []
    for a, c, f in universal:
        s = comps[a][:, c].copy()
        if np.abs(np.quantile(s, 0.02)) > np.abs(np.quantile(s, 0.98)):
            s = -s
        thr = np.quantile(s, 0.95)
        on = s >= thr
        kind_share = {k: float(np.mean(kinds[on] == k)) for k in (0, 1, 2)}
        tw = collections.Counter(toks[i][1] if toks[i][0] == 'T' else 'NUM' for i in np.where(on)[0])
        enr = sorted(((tw[w] / cnt[w], w, tw[w]) for w in tw if cnt[w] >= 3 and w != 'NUM'), reverse=True)[:8]
        # context: word before the firing position (for number tokens)
        prevw = collections.Counter()
        for i in np.where(on & (kinds == 2))[0]:
            ci, j = pos[i]
            for k in range(j - 1, -1, -1):
                if ch[ci][2][k][0] == 'T':
                    prevw[ch[ci][2][k][1]] += 1; break
        feats.append({'model': a, 'comp': c, 'univ': f, 'kind_share': kind_share, 'top_words': enr,
                      'num_prev': prevw.most_common(6), 'on': on})
    # cluster universal features across models into feature families by activation correlation
    fam = []
    for i, ft in enumerate(feats):
        s = comps[ft['model']][:, ft['comp']]
        placed = False
        for F_ in fam:
            s0 = comps[feats[F_[0]]['model']][:, feats[F_[0]]['comp']]
            if abs(np.corrcoef(s, s0)[0, 1]) >= RMIN:
                F_.append(i); placed = True; break
        if not placed:
            fam.append([i])
    res['families'] = []
    for F_ in sorted(fam, key=len, reverse=True):
        nmod = len(set(feats[i]['model'] for i in F_))
        ft = feats[F_[0]]
        res['families'].append({'n_models': nmod, 'kind_share': ft['kind_share'], 'top_words': ft['top_words'],
                                'num_prev': ft['num_prev'], 'rep': (ft['model'], ft['comp'])})
    # planted check: does any family fire on planted total numbers?
    if CORP == 'PLANT':
        P = L.planted()[1]['TOTAL']
        lab = np.zeros(len(pos), bool)
        for i, (ci, j) in enumerate(pos):
            if kinds[i] == 2 and j > 0 and ch[ci][2][j - 1][0] == 'T' and ch[ci][2][j - 1][1] in P:
                lab[i] = True
        aucs = []
        for F_ in fam:
            ft = feats[F_[0]]
            s = comps[ft['model']][:, ft['comp']]
            if np.abs(np.quantile(s, 0.02)) > np.abs(np.quantile(s, 0.98)):
                s = -s
            msk = kinds == 2
            from sklearn.metrics import roc_auc_score
            aucs.append(round(float(roc_auc_score(lab[msk], s[msk])), 3))
        res['planted_auc_by_family'] = aucs
    # causal test: project feature direction out of final states of its model, held-out loss in/out of firing set
    caus = []
    for fi, F_ in enumerate(fam[:12]):
        ft = feats[F_[0]]
        m, cfg, trd, ica, Z = H[ft['model']]
        w = ica.mixing_[:, ft['comp']]
        u = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32)
        mean_proj = float((Z @ (w / np.linalg.norm(w))).mean())
        on = ft['on']
        te = [i for i, (ci, j) in enumerate(pos) if ch[ci][0] not in trd]
        d_in, d_out = [], []
        with torch.no_grad():
            for s in range(0, len(te), 256):
                part = te[s:s + 256]
                xs = []
                for i in part:
                    ci, j = pos[i]; ids = list(ch[ci][1]); ids[j] = 1; xs.append(ids)
                x = L.pad(xs)
                idx = torch.arange(len(part)); cols = torch.tensor([pos[i][1] for i in part])
                y = torch.tensor([ch[pos[i][0]][1][pos[i][1]] for i in part])
                _, z = m(x, hidden=True)
                zq = z[idx, cols]
                base = F.cross_entropy(m.out(zq), y, reduction='none')
                zq2 = zq - (zq @ u)[:, None] * u[None] + mean_proj * u[None]
                abl = F.cross_entropy(m.out(zq2), y, reduction='none')
                for k, i in enumerate(part):
                    (d_in if on[i] else d_out).append(float(abl[k] - base[k]))
        caus.append((fi, round(float(np.mean(d_in)) if d_in else 0, 3), round(float(np.mean(d_out)), 3), len(d_in)))
    res['causal'] = caus
    # truth enrichment for controls
    if CORP.startswith('LB') or CORP.startswith('UR'):
        tr = C45.lb_truth(docs) if CORP.startswith('LB') else C45.ur_truth(docs)
        tw = {'LB': ['to-so', 'to-sa'], 'UR': ['szunigin']}[CORP[:2]]
        ann = []
        for F_ in fam[:12]:
            ft = feats[F_[0]]
            on = ft['on']
            roles = collections.Counter(tr.get(toks[i][1], '-') for i in np.where(on)[0] if toks[i][0] == 'T')
            base = collections.Counter(tr.get(t[1], '-') for t in toks if t[0] == 'T')
            tot = sum(roles.values()) or 1; btot = sum(base.values())
            enr = {r: round((roles[r] / tot) / (base[r] / btot), 2) for r in roles if base[r] >= 20}
            # totals: firing on numbers after/before the total word
            near = 0; nn_ = 0
            for i in np.where(kinds == 2)[0]:
                ci, j = pos[i]; raw = ch[ci][2]
                adj = [raw[k][1] for k in (j - 1, j - 2, j + 1) if 0 <= k < len(raw) and raw[k][0] == 'T']
                if any(a in tw for a in adj):
                    nn_ += 1; near += on[i]
            ann.append({'enrich': enr, 'total_num_fire': (int(near), nn_)})
        res['truth'] = ann
    for ft in feats:
        del ft['on']
    json.dump(res, open(os.path.join(L.CK, 'c3', CORP + '.json'), 'w'), default=str)
    print(json.dumps({k: v for k, v in res.items() if k != 'families'}, default=str))
    for i, F_ in enumerate(res['families'][:12]):
        print(i, F_['n_models'], {k: round(v, 2) for k, v in F_['kind_share'].items()}, F_['top_words'][:6], F_['num_prev'][:4])


if __name__ == '__main__':
    os.makedirs(os.path.join(L.CK, 'c3'), exist_ok=True)
    main()
