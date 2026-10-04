"""v21 neural pieces: a small GRU line forger and learned (CNN) line / paragraph discriminators.

Forger: character (unit) GRU language model. Input = [section][line type] previous line '|' current line
'\n'; loss only on the current line. Sampling is constrained to the real skeleton: exactly the real number
of words per line (an end-of-line before that is forbidden; the space after the last word becomes EOL).
"""
import os, sys, math, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_lib import *
import torch, torch.nn as nn
torch.set_num_threads(1)

PAD, BOS, SEP, EOL, SPC = 0, 1, 2, 3, 4


class Vocab:
    def __init__(self, C):
        units = sorted({c for p in C for w in tokens(p) for c in w})
        secs = sorted({p['sec'] for p in C})
        self.itos = ['<pad>', '<bos>', '|', '\n', ' '] + ['S:' + s for s in secs] + ['T:pf', 'T:li'] + units
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def enc_line(self, ws):
        return [SPC if c == ' ' else self.stoi[c] for c in ' '.join(ws)]

    def ctx(self, sec, li, prev):
        x = [BOS, self.stoi['S:' + sec], self.stoi['T:pf' if li == 0 else 'T:li']]
        if prev: x += self.enc_line(prev)
        return x + [SEP]


class LM(nn.Module):
    def __init__(self, V, d=32, h=192):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.gru = nn.GRU(d, h, batch_first=True); self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(h, V)

    def forward(self, x, hid=None):
        y, hid = self.gru(self.emb(x), hid)
        return self.out(self.drop(y)), hid


def lm_examples(C, voc):
    ex = []
    for p in C:
        for pa in p['paras']:
            for li, ws in enumerate(pa):
                prev = pa[li - 1] if li > 0 else None
                c = voc.ctx(p['sec'], li, prev); t = voc.enc_line(ws) + [EOL]
                ex.append((c, t))
    return ex


def train_lm(C, epochs=25, seed=0, log=None):
    torch.manual_seed(seed); random.seed(seed)
    voc = Vocab(C); ex = lm_examples(C, voc)
    m = LM(len(voc.itos)); opt = torch.optim.Adam(m.parameters(), 3e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(epochs):
        random.shuffle(ex); tot = 0; nt = 0; m.train()
        for b in range(0, len(ex), 64):
            B = ex[b:b + 64]
            L = max(len(c) + len(t) for c, t in B)
            X = torch.full((len(B), L), PAD, dtype=torch.long); Y = torch.full((len(B), L), -100, dtype=torch.long)
            for i, (c, t) in enumerate(B):
                s = c + t
                X[i, :len(s) - 1] = torch.tensor(s[:-1])
                Y[i, len(c) - 1:len(s) - 1] = torch.tensor(t)
            o, _ = m(X)
            loss = lossf(o.reshape(-1, o.shape[-1]), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            k = (Y != -100).sum().item(); tot += loss.item() * k; nt += k
        if log: log(f'  lm epoch {ep} loss {tot / nt:.3f} bits/unit {tot / nt / math.log(2):.3f}')
    m.eval()
    return m, voc


@torch.no_grad()
def sample_line(m, voc, sec, li, prev, nwords, rng, temp=1.0, maxlen=120):
    x = torch.tensor([voc.ctx(sec, li, prev)])
    o, hid = m(x)
    logits = o[0, -1]
    out = []; words = 0; cur = 0
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    while len(out) < maxlen:
        l = logits / temp
        l[PAD] = l[BOS] = l[SEP] = -1e9
        for i, s in enumerate(voc.itos):
            if s.startswith('S:') or s.startswith('T:'): l[i] = -1e9
        if cur == 0: l[SPC] = -1e9; l[EOL] = -1e9          # no empty word
        if words < nwords - 1: l[EOL] = -1e9
        if words == nwords - 1 and cur > 0:                 # last word: a space means end
            l[EOL] = torch.logsumexp(torch.stack([l[EOL], l[SPC]]), 0); l[SPC] = -1e9
        pr = torch.softmax(l, 0)
        t = int(torch.multinomial(pr, 1, generator=g))
        if t == EOL: break
        out.append(t)
        if t == SPC: words += 1; cur = 0
        else: cur += 1
        o, hid = m(torch.tensor([[t]]), hid); logits = o[0, -1]
    s = ''.join(' ' if t == SPC else voc.itos[t] for t in out)
    ws = [w for w in s.split(' ') if w]
    while len(ws) < nwords: ws.append('y')
    return ws[:nwords]


class NeuralForger:
    def __init__(self, C, epochs=25, seed=0, log=None, name='F8'):
        self.name = name
        self.m, self.voc = train_lm(C, epochs, seed, log)

    def forge(self, C, rng, temp=1.0):
        out = []
        for p in C:
            paras = []
            for pa in p['paras']:
                o = []
                for li, ws in enumerate(pa):
                    o.append(sample_line(self.m, self.voc, p['sec'], li, o[-1] if li else None, len(ws), rng, temp))
                paras.append(o)
            q = dict(p); q['paras'] = paras; out.append(q)
        return out


# ------------------------------------------------------------------ learned discriminator
class CNN(nn.Module):
    def __init__(self, V, d=24, f=64):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.c3 = nn.Conv1d(d, f, 3, padding=1); self.c5 = nn.Conv1d(d, f, 5, padding=2); self.c9 = nn.Conv1d(d, f, 9, padding=4)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * f, 1))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2); mask = (x != 0).unsqueeze(1)
        hs = [torch.relu(c(e)).masked_fill(~mask, -1e4).max(2).values for c in (self.c3, self.c5, self.c9)]
        return self.out(torch.cat(hs, 1)).squeeze(1)


def seq_units(C, level, voc_map):
    """level 'line': one sample per line; 'para': one per paragraph (lines joined by EOL). Returns (seqs, page idx)."""
    S, G = [], []
    for pi, p in enumerate(C):
        for pa in p['paras']:
            if level == 'line':
                for ws in pa:
                    S.append([voc_map.get(c, 5) for c in ' '.join(ws)]); G.append(pi)
            else:
                s = []
                for ws in pa: s += [voc_map.get(c, 5) for c in ' '.join(ws)] + [voc_map['\n']]
                S.append(s[:600]); G.append(pi)
    return S, np.array(G)


def cnn_auc(Creal, Cforg, level='line', seed=0, nfold=5, epochs=6, log=None):
    torch.manual_seed(seed)
    chars = sorted({c for C in (Creal, Cforg) for p in C for w in tokens(p) for c in w})
    vm = {c: i + 6 for i, c in enumerate(chars)}; vm[' '] = 2; vm['\n'] = 3
    Sr, Gr = seq_units(Creal, level, vm); Sf, Gf = seq_units(Cforg, level, vm)
    S = Sr + Sf; G = np.r_[Gr, Gf]; y = np.r_[np.ones(len(Sr)), np.zeros(len(Sf))]
    fold = folds(len(Creal), nfold, seed)[G]
    sc = np.zeros(len(S))
    for f in range(nfold):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
        m = CNN(len(vm) + 6); opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-4)
        rng = np.random.RandomState(seed * 100 + f)
        bs = 64 if level == 'line' else 16
        for ep in range(epochs):
            m.train(); rng.shuffle(tr)
            for b in range(0, len(tr), bs):
                I = tr[b:b + bs]; L = max(len(S[i]) for i in I)
                X = torch.zeros((len(I), L), dtype=torch.long)
                for j, i in enumerate(I): X[j, :len(S[i])] = torch.tensor(S[i])
                loss = nn.functional.binary_cross_entropy_with_logits(m(X), torch.tensor(y[I], dtype=torch.float32))
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            for b in range(0, len(te), 256):
                I = te[b:b + 256]; L = max(len(S[i]) for i in I)
                X = torch.zeros((len(I), L), dtype=torch.long)
                for j, i in enumerate(I): X[j, :len(S[i])] = torch.tensor(S[i])
                sc[I] = m(X).numpy()
    unit_auc = _auc(sc[y == 1], sc[y == 0])
    # page-level aggregate: mean score of a page's units
    npg = len(Creal)
    pr = np.array([sc[:len(Sr)][Gr == i].mean() if (Gr == i).any() else 0 for i in range(npg)])
    pf = np.array([sc[len(Sr):][Gf == i].mean() if (Gf == i).any() else 0 for i in range(npg)])
    return unit_auc, _auc(pr, pf)
