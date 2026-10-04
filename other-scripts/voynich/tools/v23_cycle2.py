"""v23 cycle 2: small neural forward vs backward predictors (glyph-level GRU) with held-out pages.

For each corpus: pages split into 2 folds; a GRU is trained on the training fold read forwards and
another on the same fold read backwards (identical size, steps, seed); both score the held-out pages
(forward model on pages as written, backward model on reversed pages).  a_p = bits_bwd - bits_fwd.
Sign-flip null over held-out pages; reversible-chain surrogates calibrate.
"""
import sys, os, json, math, random, time
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
torch.set_num_threads(1)
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L
from v23_cycle1 import build

STEPS, BATCH, SEQ, H = 800, 32, 64, 64
SEEDS = (0, 1)


class Net(torch.nn.Module):
    def __init__(self, V):
        super().__init__()
        self.e = torch.nn.Embedding(V, 32); self.g = torch.nn.GRU(32, H, batch_first=True)
        self.o = torch.nn.Linear(H, V)

    def forward(self, x, h=None):
        y, h = self.g(self.e(x), h); return self.o(y), h


def train(text, V, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Net(V); opt = torch.optim.Adam(m.parameters(), 3e-3)
    t = torch.tensor(text, dtype=torch.long); n = len(text) - SEQ - 1
    for s in range(STEPS):
        ix = rng.integers(0, n, BATCH)
        x = torch.stack([t[i:i + SEQ] for i in ix]); y = torch.stack([t[i + 1:i + SEQ + 1] for i in ix])
        lo, _ = m(x)
        loss = torch.nn.functional.cross_entropy(lo.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s == int(STEPS * 0.7):
            for g in opt.param_groups: g['lr'] = 1e-3
    return m


@torch.no_grad()
def score(m, seq, V):
    x = torch.tensor([seq[:-1]], dtype=torch.long); y = torch.tensor(seq[1:], dtype=torch.long)
    lo, _ = m(x)
    return float(torch.nn.functional.cross_entropy(lo[0], y, reduction='sum')) / math.log(2), len(seq) - 1


def run(name, seed=0):
    fn = os.path.join(L.CK, f'c2s{seed}_{name}.json')
    if os.path.exists(fn): return name
    t0 = time.time()
    C = build(name)
    S = [L.stream(p) for p in C]
    alpha = sorted(set(''.join(S)) | {'^'})
    ix = {c: i for i, c in enumerate(alpha)}; V = len(alpha)
    enc = lambda s: [ix[c] for c in s]
    idx = list(range(len(C))); random.Random(seed).shuffle(idx)
    fold = {p: i % 2 for i, p in enumerate(idx)}
    a, ns, hf = [None] * len(C), 0, 0
    for f in (0, 1):
        tr = [p for p in range(len(C)) if fold[p] != f]
        txt_f = enc('^' + '^'.join(S[p] for p in tr) + '^')
        txt_b = enc('^' + '^'.join(S[p][::-1] for p in tr[::-1]) + '^')
        mf = train(txt_f, V, seed); mb = train(txt_b, V, seed)
        for p in range(len(C)):
            if fold[p] == f:
                bf, n = score(mf, enc('^' + S[p] + '^'), V); bb, _ = score(mb, enc('^' + S[p][::-1] + '^'), V)
                a[p] = bb - bf; ns += n; hf += bf
    z, pv = L.signflip(a)
    res = {'name': name, 'a': a, 'eff': sum(a) / ns, 'H_fwd': hf / ns, 'rel': sum(a) / hf, 'z': float(z), 'p': float(pv),
           'sec': time.time() - t0}
    json.dump(res, open(fn, 'w'))
    print(name, seed, round(res['eff'], 4), round(z, 2), round(res['sec']), flush=True)
    return name


if __name__ == '__main__':
    names = sys.argv[1:] or ['ZL', 'IT', 'LA', 'ITA', 'DE', 'CS', 'HE', 'PL_REV', 'MkG_1']
    jobs = [(n, s) for s in SEEDS for n in names]
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.starmap(run, jobs): pass
