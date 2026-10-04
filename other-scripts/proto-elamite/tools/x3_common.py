"""X-3 common: pretrain a tiny sequence model on corpus X, fine-tune on a slice of corpus Y,
score held-out Y. Conditions for X:
  scratch  no pretraining
  real     X as it is
  relab    X with its sign identities randomly permuted (structure kept, identities gone)
  band     X relabelled only within frequency-rank bands of 8 (keeps approximate frequency identity)
  shuf     X tokens shuffled across the corpus, line lengths kept (same unigrams, no order)
  markov   random first-order Markov text, Zipfian, X's vocabulary size and line lengths
Index mapping from Y signs to model rows:
  'label'  a Y sign with the same label as an X sign uses that sign's row; other Y signs fill free
           rows by Y frequency (identity transfer through a shared transliteration convention)
  'rank'   the r-th commonest Y sign uses the row of the r-th commonest X sign
           (no shared labels needed; 'identity' then means frequency-rank correspondence)
Model: neural 4-gram (sign embeddings d 64 -> MLP 128 -> tied output), BOS/EOS, full fine-tuning (Adam).
Held-out Y: test = blocks of 20 consecutive lines (<= 2,000 tokens), val (<= 1,000 tokens) for
choosing the fine-tune step; loss in bits per token (EOS included).
"""
import os, json, random, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
SCR = os.environ.get('X3_SCR', '/tmp/claude-0/x3')
NV = 420            # model rows: 0 PAD, 1 BOS/EOS, 2 UNK, 3.. signs
MAXLEN = 40
PRE_TOK = 8000
D = 64

_C = None


def corpora():
    global _C
    if _C is None:
        _C = json.load(open(os.path.join(SCR, 'x3_corpora.json')))
        _C.update(planted())
        lb = _C['LB']
        blocks = [lb[i:i + 50] for i in range(0, len(lb), 50)]
        _C['LBa'] = [t for k, b in enumerate(blocks) if k % 2 == 0 for t in b]
        _C['LBb'] = [t for k, b in enumerate(blocks) if k % 2 == 1 for t in b]
    return _C


def planted(seed=11):
    """Two corpora from one hidden grammar (HMM, 30 states, 80 signs, sparse emissions and
    transitions, line lengths 2-12). PLa / PLb share sign labels; PLc = a third sample with its
    signs renamed (same grammar, no shared identities)."""
    rng = np.random.default_rng(seed)
    S, V = 30, 80
    T = rng.dirichlet(np.full(S, 0.08), size=S)
    E = rng.dirichlet(np.full(V, 0.05), size=S)
    st0 = rng.dirichlet(np.full(S, 0.3))
    out = {}
    names = [f'p{i}' for i in range(V)]
    perm = rng.permutation(V)
    for k, n in (('PLa', 9000), ('PLb', 9000), ('PLc', 9000)):
        lines, tot = [], 0
        while tot < n:
            L = int(rng.integers(2, 13)); s = rng.choice(S, p=st0); t = []
            for _ in range(L):
                v = rng.choice(V, p=E[s]); t.append(names[v] if k != 'PLc' else f'q{perm[v]}')
                s = rng.choice(S, p=T[s])
            lines.append(t); tot += L
        out[k] = lines
    return out


def split(lines, seed=0):
    """Fixed test / val / train split by blocks of 20 lines."""
    rng = random.Random(1234 + seed)
    blocks = [lines[i:i + 20] for i in range(0, len(lines), 20)]
    idx = list(range(len(blocks))); rng.shuffle(idx)
    ntok = sum(map(len, lines))
    tcap, vcap = min(2000, 0.18 * ntok), min(1000, 0.12 * ntok)
    test, val, train = [], [], []
    for i in idx:
        b = blocks[i]; n = sum(map(len, b))
        if sum(map(len, test)) < tcap:
            test += b
        elif sum(map(len, val)) < vcap:
            val += b
        else:
            train.append(b)
    return test, val, train


def sample_blocks(blocks, ntok, rng):
    idx = list(range(len(blocks))); rng.shuffle(idx)
    out, n = [], 0
    for i in idx:
        for t in blocks[i]:
            out.append(t); n += len(t)
        if n >= ntok:
            break
    return out


def take_tokens(lines, ntok, rng):
    blocks = [lines[i:i + 20] for i in range(0, len(lines), 20)]
    return sample_blocks(blocks, ntok, rng)


def rank_order(lines):
    from collections import Counter
    c = Counter(x for t in lines for x in t)
    return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]


# ------------------------------------------------------------------ X conditions
def make_x(xname, cond, seed):
    """Return (lines as label lists, sign->row map) for pretraining condition cond."""
    C = corpora()
    rng = random.Random(seed * 7919 + sum(map(ord, xname)) * 13)
    X = take_tokens(C[xname], PRE_TOK, rng)
    order = rank_order(X)
    nrow = min(len(order), NV - 3)
    rowmap = {s: 3 + i for i, s in enumerate(order[:nrow])}
    if cond == 'relab':
        rows = list(rowmap.values()); rng.shuffle(rows)
        rowmap = dict(zip(rowmap.keys(), rows))
    elif cond == 'band':
        keys = list(rowmap.keys()); new = {}
        for b in range(0, len(keys), 8):
            ks = keys[b:b + 8]; rows = [rowmap[k] for k in ks]; rng.shuffle(rows)
            new.update(zip(ks, rows))
        rowmap = new
    elif cond == 'shuf':
        toks = [x for t in X for x in t]; rng.shuffle(toks)
        out, i = [], 0
        for t in X:
            out.append(toks[i:i + len(t)]); i += len(t)
        X = out
    elif cond == 'markov':
        g = np.random.default_rng(seed * 31 + 5)
        V = nrow
        base = 1.0 / np.arange(1, V + 1)
        base /= base.sum()
        T = np.stack([g.dirichlet(base * V * 0.3 + 1e-3) for _ in range(V)])
        out = []
        for t in X:
            s = g.choice(V, p=base); u = []
            for _ in range(len(t)):
                u.append(order[s]); s = g.choice(V, p=T[s])
            out.append(u)
        X = out
    return X, rowmap


def y_map(xrowmap, xorder_rows, ytrain, mode):
    """Map Y signs to rows. xorder_rows: X's rows in X frequency order (as pretrained)."""
    yord = rank_order(ytrain)
    m = {}
    if mode == 'label':
        used = set()
        for s in yord:
            if s in xrowmap:
                m[s] = xrowmap[s]; used.add(xrowmap[s])
        free = [r for r in range(3, NV) if r not in used]
        k = 0
        for s in yord:
            if s not in m:
                if k < len(free):
                    m[s] = free[k]; k += 1
    else:
        free = list(xorder_rows) + [r for r in range(3, NV) if r not in set(xorder_rows)]
        for i, s in enumerate(yord):
            if i < len(free):
                m[s] = free[i]
    return m


def encode(lines, m):
    return [[1] + [m.get(x, 2) for x in t[:MAXLEN]] + [1] for t in lines if t]


# ------------------------------------------------------------------ model
K = 4               # context window (previous signs; BOS-padded)
H = 128


class LM(nn.Module):
    """Neural K-gram model (Bengio-style): K previous sign embeddings -> MLP -> tied output."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NV, D, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(K, D))
        self.l1 = nn.Linear(K * D, H)
        self.l2 = nn.Linear(H, D)
        self.drop = nn.Dropout(0.1)

    def forward(self, ctx):
        e = self.emb(ctx) + self.pos
        h = torch.tanh(self.l1(self.drop(e.reshape(len(ctx), -1))))
        return self.l2(self.drop(h)) @ self.emb.weight.T


def positions(seqs):
    """seqs (encoded, BOS ... EOS) -> (contexts [n,K], targets [n])."""
    ctx, tgt = [], []
    for s in seqs:
        for i in range(1, len(s)):
            c = s[max(0, i - K):i]
            ctx.append([0] * (K - len(c)) + c); tgt.append(s[i])
    return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    ctx, tgt = data
    l = nn.functional.cross_entropy(model(ctx), tgt).item()
    model.train()
    return l / math.log(2)


def train(model, data, steps, lr, bs, rng, evals=None, every=25):
    ctx, tgt = data
    g = torch.Generator().manual_seed(rng.randrange(10 ** 9))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    model.train()
    for st in range(1, steps + 1):
        i = torch.randint(0, len(tgt), (bs,), generator=g)
        loss = nn.functional.cross_entropy(model(ctx[i]), tgt[i])
        opt.zero_grad(); loss.backward()
        opt.step()
        if evals and st % every == 0:
            curve.append([st] + [evaluate(model, e) for e in evals])
    return curve


def pretrain(xname, cond, seed, steps=3000):
    path = os.path.join(SCR, 'pre', f'{xname}_{cond}_{seed}_{PRE_TOK}.pt')
    X, rowmap = make_x(xname, cond, seed)
    xorder_rows = [rowmap[s] for s in rank_order(make_x(xname, 'real', seed)[0]) if s in rowmap]
    if os.path.exists(path):
        sd = torch.load(path)
    else:
        torch.manual_seed(seed)
        model = LM()
        train(model, positions(encode(X, rowmap)), steps, 3e-3, 128, random.Random(seed))
        sd = model.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(sd, path)
    return sd, rowmap, xorder_rows


def finetune(sd, ymap, ytr, yval, yte, seed, steps=300):
    torch.manual_seed(seed)
    model = LM()
    if sd is not None:
        model.load_state_dict(sd)
    rng = random.Random(seed)
    tr, va, te = (positions(encode(z, ymap)) for z in (ytr, yval, yte))
    curve = train(model, tr, steps, 2e-3, 64, rng, evals=[va, te], every=20)
    best = min(curve, key=lambda r: r[1])
    aulc = float(np.mean([r[2] for r in curve]))
    return {'best_step': best[0], 'test': best[2], 'aulc': aulc,
            'curve': [round(r[2], 4) for r in curve]}
