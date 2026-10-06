"""v70 cycle 3: let the ink choose the alphabet size with no language prior (minimum description length),
on self-supervised patch codes.

Embedding: small denoising conv autoencoder (PyTorch, CPU, 2 threads) trained per corpus on its own
unit patches (random shifts + pixel dropout), 16-dim code, page-centred.
For each theta and K = 6..80, k-means on the codes; description length in bits
   DL = sum over units of Gaussian residual cost (d/2 log2 of per-cluster variance, quantised at 1/16 SD)
      + cross-entropy of the label sequence under a held-out-trained bigram model (words in, '^'/'$' edges)
      + model cost (K*d/2*log2 N + K^2/2*log2 N for the bigram table).
The MDL alphabet (argmin DL over theta, K) is then scored by the cycle-1 language classifier and its size is
compared with the true unit inventory for synthetic corpora (SV: EVA units; SL: letters + 2 ligatures +
split halves) and the Latin control (letters + abbreviation signs).
Usage: python3 v70_c3.py corpus
"""
import sys, os, json, collections, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import *
from v70_c1 import words_from, bpe, nearest, THETAS
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
import torch
import torch.nn as nn
torch.set_num_threads(1)
ARR = os.path.join(SCR, 'v70', 'arr')
KS3 = [6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30, 34, 38, 43, 48, 54, 60, 70, 80]


class AE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(32 * 6 * 8, d))
        self.dec = nn.Sequential(nn.Linear(d, 32 * 6 * 8), nn.ReLU(), nn.Unflatten(1, (32, 6, 8)),
                                 nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, x):
        z = self.enc(x); return self.dec(z), z


def ae_codes(name, th, epochs=4):
    fn = os.path.join(ARR, f'{name}_t{th}_AE.npy')
    D = json.load(open(os.path.join(ARR, name + '.json')))
    U = D['units'][str(th)]
    if os.path.exists(fn):
        return np.load(fn), D, U
    P = torch.tensor(np.load(os.path.join(ARR, f'{name}_t{th}_P.npy')).astype(np.float32) / 255.0)[:, None]
    torch.manual_seed(0)
    net = AE(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    n = len(P); g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)[:min(n, 40000)]
        tot = 0.0
        for i in range(0, len(perm), 256):
            x = P[perm[i:i + 256]]
            sx, sy = np.random.randint(-2, 3, 2)
            xa = torch.roll(x, (int(sy), int(sx)), (2, 3)) * (torch.rand_like(x) > 0.1)
            rec, _ = net(xa)
            loss = ((rec - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(x)
        print(name, th, 'ae epoch', ep, tot / len(perm), flush=True)
    with torch.no_grad():
        Z = torch.cat([net(P[i:i + 2048])[1] for i in range(0, n, 2048)]).numpy()
    wd = np.array([u[1] for u in U], np.float32)[:, None]
    Z = np.concatenate([Z, wd * Z.std() * 3], 1)
    pg = np.array([D['recs'][u[0]]['pg'] for u in U])
    for p in np.unique(pg):
        Z[pg == p] -= Z[pg == p].mean(0)
    Z = (Z / (Z.std(0) + 1e-9)).astype(np.float32)
    np.save(fn, Z)
    return Z, D, U


def bigram_xent(train, test, K):
    c = collections.Counter(); cc = collections.Counter()
    for w in train:
        t = ['^'] + list(w) + ['$']
        for a, b in zip(t, t[1:]):
            c[(a, b)] += 1; cc[a] += 1
    V = K + 1; bits = 0.0; n = 0
    for w in test:
        t = ['^'] + list(w) + ['$']
        for a, b in zip(t, t[1:]):
            bits -= math.log2((c[(a, b)] + 0.5) / (cc[a] + 0.5 * V)); n += 1
    return bits, n


def dl(Z, lab, K, W, D):
    N, d = Z.shape
    res = 0.0
    for k in range(K):
        x = Z[lab == k]
        if len(x) < 2:
            continue
        var = x.var(0) + 1e-3
        res += len(x) * 0.5 * np.log2(2 * np.pi * np.e * var * 256).sum()   # quantisation 1/16 SD
    pg = np.array([r['pg'] for r in D['recs']])
    A = [w for w, p in zip(W, pg) if p % 2 == 0 and w]; B = [w for w, p in zip(W, pg) if p % 2 == 1 and w]
    b1, _ = bigram_xent(A, B, K); b2, _ = bigram_xent(B, A, K)
    model = K * d / 2 * np.log2(N) + K * K / 2 * np.log2(N)
    return float(res), float(b1 + b2), float(model)


def main(name):
    out = {'rows': []}
    for th in THETAS:
        Z, D, U = ae_codes(name, th)
        truth = [u[3] for u in U]
        for K in KS3:
            lab = MiniBatchKMeans(K, random_state=0, n_init=1, batch_size=4096).fit_predict(Z)
            W = words_from(lab, U, len(D['recs']))
            r, t, m = dl(Z, lab, K, W, D)
            row = {'theta': th, 'K': K, 'res': r, 'text': t, 'model': m, 'DL': r + t + m}
            pgs = np.array([D['recs'][u[0]]['pg'] for u in U])
            A = [w for w, rr in zip(W, D['recs']) if rr['pg'] % 2 == 0 and w]
            B = [w for w, rr in zip(W, D['recs']) if rr['pg'] % 2 == 1 and w]
            row['A'] = battery(A); row['B'] = battery(B)
            if any(truth):
                ok = [i for i, t in enumerate(truth) if t]
                row['nmi'] = float(NMI([truth[i] for i in ok], lab[ok]))
            out['rows'].append(row)
        print(name, th, min(out['rows'], key=lambda r: r['DL'])['K'], flush=True)
    # per-unit-count normalisation: DL per unit differs by theta (different N), so pick per theta
    # and across thetas by DL per word (all thetas encode the same words)
    nW = len(D['recs'])
    for r in out['rows']:
        r['DLw'] = r['DL'] / nW
    best = min(out['rows'], key=lambda r: r['DLw'])
    out['mdl_best'] = {k: best[k] for k in ('theta', 'K', 'DLw')}
    out['mdl_by_theta'] = {str(th): min([r for r in out['rows'] if r['theta'] == th], key=lambda r: r['DL'])['K'] for th in THETAS}
    if 'nmi' in best:
        out['mdl_best']['nmi'] = best['nmi']
        out['nmi_best_any'] = max(r['nmi'] for r in out['rows'])
        out['K_at_nmi_best'] = max(out['rows'], key=lambda r: r['nmi'])['K']
    json.dump(out, open(os.path.join(CK, f'c3_{name}.json'), 'w'))
    print(name, json.dumps({k: v for k, v in out.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
