#!/usr/bin/env python3
"""LA-49 shared code: DISSECT A MACHINE THAT LEARNED LINEAR A.

We do not decipher the script; we decipher models of it. Hundreds of tiny masked-token models
(transformers with inspectable heads, bidirectional GRUs) are trained with random sizes / seeds /
train splits to fill masked words, logograms and numbers on whole documents. Each trained model is
then dissected:
  F1  attention read-off: for every head, how strongly a masked NUMBER query attends to each word
      type (log ratio to uniform attention), giving a per-word "attractor" vector per head.
  F2  head ablation: held-out loss increase on number / word / logogram positions when one head is
      zeroed. The 'number head' is the head with the largest number-specific damage.
  F3  running-sum circuit by causal intervention: mask a number, raise every EARLIER number in the
      document by two size buckets, measure the shift of the predicted log value. Positions that
      react like totals are attributed to the nearest word before (PRE) and after (POST).
      The head whose ablation removes most of that reaction is the sum head; its ablation damage on
      held-out top-sensitivity positions vs other number positions is recorded.
  F4  geometry: mean final-layer state per word type (types seen >= 3 times) for cross-seed
      representational-similarity analysis and consensus clustering.
Word identities are opaque strings; no sound values are used anywhere.
"""
import json, os, sys, math, random, hashlib, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la49_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
import la45_common as C45  # corpus loaders and control truth tables (truth used for scoring only)

MAXLEN = 64
BUCKETS = [0, 1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 300, 1000]  # lower edges
CENT = [math.log(0.5), 0, math.log(2), math.log(3), math.log(4), math.log(5), math.log(7.5),
        math.log(14), math.log(32), math.log(71), math.log(170), math.log(550), math.log(2000)]


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def bucket(v):
    b = 0
    for i, lo in enumerate(BUCKETS):
        if v >= lo:
            b = i
    if v < 1:
        b = 0
    return b


# ------------------------------------------------------------------ corpora
def corpus(name, rng=None):
    """Return list of docs (dict id, site, toks) for a named corpus variant."""
    if name == 'LA':
        return C45.la_docs()
    if name.startswith('LASHUF'):
        k = int(name[6:] or 0)
        docs = C45.la_docs()
        r = random.Random(seed('la49-shuf-%d' % k))
        slots = [(i, j) for i, d in enumerate(docs) for j, t in enumerate(d['toks']) if t[0] == 'T']
        vals = [docs[i]['toks'][j] for i, j in slots]
        r.shuffle(vals)
        out = [dict(d, toks=list(d['toks'])) for d in docs]
        for (i, j), v in zip(slots, vals):
            out[i]['toks'][j] = v
        return out
    if name.startswith('LB') or name.startswith('UR'):
        base, mult = name[:2], (int(name[2:]) if len(name) > 2 else 1)
        docs = C45.lb_docs_all() if base == 'LB' else C45.ur3_docs_all()
        r = random.Random(seed('la49-size-' + name))
        return C45.sample_size(docs, 5245 * mult, r)
    if name == 'PLANT':
        return planted()[0]
    raise ValueError(name)


def planted():
    """Linear A with two planted roles on REAL word types (chosen by a fixed seed):
    TOTAL: 5 word types; on 50% of documents with >= 2 numbers, append [word, sum of numbers].
    BIND : 5 other word types; every number right after them is forced to a fixed value per word.
    Returns (docs, {'TOTAL': [...], 'BIND': [...]})."""
    docs = [dict(d, toks=list(d['toks'])) for d in C45.la_docs()]
    cnt = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'T' and not t[1].startswith('L:'))
    nxtnum = collections.Counter()
    for d in docs:
        for j in range(len(d['toks']) - 1):
            if d['toks'][j][0] == 'T' and d['toks'][j + 1][0] == 'N':
                nxtnum[d['toks'][j][1]] += 1
    r = random.Random(seed('la49-plant'))
    pool_t = sorted(w for w, c in cnt.items() if 3 <= c <= 8 and w not in ('KU-RO', 'PO-TO-KU-RO', 'KI-RO'))
    tot = r.sample(pool_t, 5)
    pool_b = sorted(w for w, c in nxtnum.items() if c >= 4 and w not in tot and w not in ('KU-RO', 'PO-TO-KU-RO', 'KI-RO'))
    bind = r.sample(pool_b, 5)
    bval = {w: v for w, v in zip(bind, [2, 7, 30, 150, 600])}
    for d in docs:
        ts = d['toks']
        for j in range(len(ts) - 1):
            if ts[j][0] == 'T' and ts[j][1] in bval and ts[j + 1][0] == 'N':
                ts[j + 1] = ('N', float(bval[ts[j][1]]), False)
        nums = [t[1] for t in ts if t[0] == 'N']
        if len(nums) >= 2 and r.random() < 0.5:
            ts += [('T', r.choice(tot)), ('N', float(sum(nums)), False)]
    return docs, {'TOTAL': tot, 'BIND': bind}


# ------------------------------------------------------------------ encoding
class Vocab:
    def __init__(self, docs, minc=2):
        c = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'T')
        self.itos = ['<pad>', '<mask>', '<unkW>', '<unkL>'] + ['N%d%s' % (b, f) for b in range(len(BUCKETS)) for f in ('', 'f')]
        self.num0 = 4
        self.nnum = 2 * len(BUCKETS)
        self.itos += sorted(w for w, k in c.items() if k >= minc)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.count = c

    def enc_tok(self, t):
        if t[0] == 'N':
            return self.num0 + 2 * bucket(t[1]) + (1 if t[2] else 0)
        w = t[1]
        if w in self.stoi:
            return self.stoi[w]
        return 3 if w.startswith('L:') else 2

    def kind(self, i):  # 0 word, 1 logogram, 2 number, -1 special
        if i < 4:
            return -1 if i < 2 else (0 if i == 2 else 1)
        if i < self.num0 + self.nnum:
            return 2
        return 1 if self.itos[i].startswith('L:') else 0


def chunks(docs, voc):
    """Encode docs, split into windows of MAXLEN (non-overlapping). Returns list of (docidx, ids, strs)."""
    out = []
    for di, d in enumerate(docs):
        ids = [voc.enc_tok(t) for t in d['toks']]
        raw = [t for t in d['toks']]
        for s in range(0, len(ids), MAXLEN):
            out.append((di, ids[s:s + MAXLEN], raw[s:s + MAXLEN]))
    return out


def pad(seqs):
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x


# ------------------------------------------------------------------ models
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, padm, hmask=None, keep=False):
        B, L, Dm = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        a = a.masked_fill(padm[:, None, None, :], -1e9)
        a = a.softmax(-1)
        y = a @ v  # B h L dk
        if hmask is not None:
            y = y * hmask[None, :, None, None]
        self.last = a if keep else None
        return self.o(y.transpose(1, 2).reshape(B, L, Dm))


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.l1, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attn(d, h)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.dr = nn.Dropout(drop)

    def forward(self, x, padm, hmask=None, keep=False):
        x = x + self.dr(self.att(self.l1(x), padm, hmask, keep))
        return x + self.dr(self.ff(self.l2(x)))


class TF(nn.Module):
    kind = 'tf'

    def __init__(self, V, d, h, nl, drop):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        self.nl, self.h = nl, h

    def forward(self, x, hmask=None, keep=False, hidden=False):
        padm = x == 0
        z = self.emb(x) + self.pos(torch.arange(x.shape[1]))[None]
        for i, b in enumerate(self.blocks):
            z = b(z, padm, None if hmask is None else hmask[i], keep)
        z = self.ln(z)
        return (self.out(z), z) if hidden else self.out(z)


class RNN(nn.Module):
    kind = 'gru'

    def __init__(self, V, d, nl, drop):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.GRU(d, d, num_layers=nl, batch_first=True, bidirectional=True, dropout=drop if nl > 1 else 0)
        self.out = nn.Linear(2 * d, V)

    def forward(self, x, hmask=None, keep=False, hidden=False):
        z, _ = self.rnn(self.emb(x))
        return (self.out(z), z) if hidden else self.out(z)


# ------------------------------------------------------------------ training
def train_model(cfg, ch_tr, V, epochs):
    torch.manual_seed(cfg['seed'])
    if cfg['arch'] == 'tf':
        m = TF(V, cfg['d'], cfg['h'], cfg['nl'], cfg['drop'])
    else:
        m = RNN(V, cfg['d'], cfg['nl'], cfg['drop'])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=0.01)
    r = random.Random(cfg['seed'])
    seqs = [c[1] for c in ch_tr]
    for ep in range(epochs):
        m.train()
        order = list(range(len(seqs)))
        r.shuffle(order)
        for s in range(0, len(order), 32):
            batch = [seqs[i] for i in order[s:s + 32]]
            x = pad(batch)
            y = x.clone()
            msk = (torch.rand(x.shape) < 0.2) & (x != 0)
            # ensure at least one mask per row
            for i, sq in enumerate(batch):
                if not msk[i].any():
                    msk[i, r.randrange(len(sq))] = True
            xin = x.masked_fill(msk, 1)
            logits = m(xin)
            loss = F.cross_entropy(logits[msk], y[msk])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def one_mask_batches(ch, voc, only_num=False):
    """Yield (x_masked, rows, cols, chunk_index) with one masked position per row."""
    rows = []
    for ci, (di, ids, raw) in enumerate(ch):
        for j, t in enumerate(ids):
            if only_num and voc.kind(t) != 2:
                continue
            rows.append((ci, j))
    for s in range(0, len(rows), 256):
        part = rows[s:s + 256]
        x = pad([ch[ci][1] for ci, _ in part])
        y = x[torch.arange(len(part)), torch.tensor([j for _, j in part])].clone()
        x[torch.arange(len(part)), torch.tensor([j for _, j in part])] = 1
        yield x, part, y


@torch.no_grad()
def heldout_loss(m, ch, voc, hmask=None, positions=None):
    """Mean NLL by kind (word, logo, num) at one-masked positions; positions: optional set of (ci, j)."""
    tot = collections.defaultdict(float); n = collections.Counter()
    for x, part, y in one_mask_batches(ch, voc):
        lg = m(x, hmask=hmask)
        idx = torch.arange(len(part))
        cols = torch.tensor([j for _, j in part])
        nll = F.cross_entropy(lg[idx, cols], y, reduction='none')
        for k, (pp, v) in enumerate(zip(part, nll.tolist())):
            kd = voc.kind(int(y[k]))
            key = ['w', 'l', 'n'][kd] if kd >= 0 else 'w'
            if positions is not None:
                key = ('in' if pp in positions else 'out') + key
            tot[key] += v; n[key] += 1
    return {k: tot[k] / n[k] for k in tot}, dict(n)


def numexp(lg, voc):
    p = lg[..., voc.num0:voc.num0 + voc.nnum].softmax(-1)
    cent = torch.tensor([CENT[i // 2] for i in range(voc.nnum)])
    return (p * cent).sum(-1)


@torch.no_grad()
def sum_sens(m, ch, voc, hmask=None, positions=None):
    """F3: for every number position (ci, j): shift of predicted log value when all earlier numbers are
    raised two buckets. Returns dict (ci, j) -> (sens, attn per head at query j or None)."""
    rows = [(ci, j) for ci, (di, ids, raw) in enumerate(ch) for j, t in enumerate(ids)
            if voc.kind(t) == 2 and any(voc.kind(u) == 2 for u in ids[:j])]
    if positions is not None:
        rows = [r for r in rows if r in positions]
    out = {}
    for s in range(0, len(rows), 256):
        part = rows[s:s + 256]
        xs, xm = [], []
        for ci, j in part:
            ids = list(ch[ci][1])
            mod = list(ids)
            for k in range(j):
                if voc.kind(ids[k]) == 2:
                    o = ids[k] - voc.num0
                    b, f = o // 2, o % 2
                    mod[k] = voc.num0 + 2 * min(b + 2, len(BUCKETS) - 1) + f
            ids[j] = 1; mod[j] = 1
            xs.append(ids); xm.append(mod)
        cols = torch.tensor([j for _, j in part]); idx = torch.arange(len(part))
        a = numexp(m(pad(xs), hmask=hmask)[idx, cols], voc)
        b = numexp(m(pad(xm), hmask=hmask)[idx, cols], voc)
        for k, pp in enumerate(part):
            out[pp] = float(b[k] - a[k])
    return out


@torch.no_grad()
def num_attention(m, ch, voc):
    """F1: for each head, per word-type mean log(attn / uniform) from masked number queries.
    Returns {head_id: {word: [sum, n]}}."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for x, part, y in one_mask_batches(ch, voc, only_num=True):
        m(x, keep=True)
        lens = (x != 0).sum(1)
        for li, b in enumerate(m.blocks):
            A = b.att.last  # B h L L
            for k, (ci, j) in enumerate(part):
                L = int(lens[k])
                raw = ch[ci][2]
                row = A[k, :, j, :L].numpy()  # h x L
                for pos in range(L):
                    if pos == j or raw[pos][0] != 'T':
                        continue
                    w = raw[pos][1]
                    if voc.count[w] < 3:
                        continue
                    lr = np.log(row[:, pos] * L + 1e-6)
                    for hh in range(m.h):
                        e = acc['%d.%d' % (li, hh)][w]
                        e[0] += float(lr[hh]); e[1] += 1
    return {h: {w: v for w, v in d.items()} for h, d in acc.items()}


@torch.no_grad()
def type_vectors(m, ch, voc):
    """F4: mean final hidden state per word type (count >= 3) at unmasked occurrences."""
    acc = {}
    for s in range(0, len(ch), 128):
        part = ch[s:s + 128]
        x = pad([c[1] for c in part])
        _, z = m(x, hidden=True)
        for k, c in enumerate(part):
            for j, t in enumerate(c[2]):
                if t[0] == 'T' and voc.count[t[1]] >= 3:
                    v = z[k, j].numpy()
                    if t[1] in acc:
                        acc[t[1]][0] += v; acc[t[1]][1] += 1
                    else:
                        acc[t[1]] = [v.copy(), 1]
    return {w: (v / n) for w, (v, n) in acc.items()}


def attribute(ch, sens):
    """Map position-level sensitivities to nearest non-logogram word before (PRE) and after (POST)."""
    pre = collections.defaultdict(list); post = collections.defaultdict(list)
    for (ci, j), s in sens.items():
        raw = ch[ci][2]
        for k in range(j - 1, -1, -1):
            if raw[k][0] == 'T' and not raw[k][1].startswith('L:'):
                pre[raw[k][1]].append(s); break
        for k in range(j + 1, len(raw)):
            if raw[k][0] == 'T' and not raw[k][1].startswith('L:'):
                post[raw[k][1]].append(s); break
    return pre, post


def run_model(cfg):
    """Train one model on corpus cfg['corpus'] and dissect it. Returns a JSON-able dict."""
    torch.set_num_threads(1)
    docs = corpus(cfg['corpus'])
    voc = Vocab(docs)
    r = random.Random(cfg['seed'] * 7 + 1)
    idx = list(range(len(docs))); r.shuffle(idx)
    ntr = int(0.8 * len(idx))
    tr_docs, te_docs = set(idx[:ntr]), set(idx[ntr:])
    ch = chunks(docs, voc)
    ch_tr = [c for c in ch if c[0] in tr_docs]
    ch_te = [c for c in ch if c[0] in te_docs]
    m = train_model(cfg, ch_tr, len(voc.itos), cfg['epochs'])
    res = {'cfg': cfg}
    base, nb = heldout_loss(m, ch_te, voc)
    res['loss'] = base; res['n'] = nb
    # F3 on all chunks
    sens = sum_sens(m, ch, voc)
    pre, post = attribute(ch, sens)
    allv = np.array(list(sens.values())) if sens else np.zeros(1)
    res['sens_mean'] = float(allv.mean()); res['sens_sd'] = float(allv.std())
    res['pre'] = {w: [float(np.sum(v)), len(v)] for w, v in pre.items()}
    res['post'] = {w: [float(np.sum(v)), len(v)] for w, v in post.items()}
    tv = type_vectors(m, ch, voc)
    res['vec'] = {w: [round(float(a), 4) for a in v] for w, v in tv.items()}
    if cfg['arch'] == 'tf':
        res['attn'] = num_attention(m, ch, voc)
        # F2 head ablation on held-out
        # top-sensitivity positions (top 10%) define 'total-like' positions inside this model
        thr = np.quantile(allv, 0.9) if len(allv) > 10 else 1e9
        top = {p for p, s in sens.items() if s >= thr}
        top_te = {p for p in top if ch[p[0]][0] in te_docs}
        # map held-out positions to ch_te indexing
        ch_te_idx = [k for k, c in enumerate(ch) if c[0] in te_docs]
        pos_te = {(ch_te_idx.index(ci), j) for ci, j in top_te} if top_te else set()
        b2, n2 = heldout_loss(m, ch_te, voc, positions=pos_te)
        res['n_top_te'] = n2
        heads = {}
        top_list = set(list(top))
        for li in range(m.nl):
            for hh in range(m.h):
                hm = [torch.ones(m.h) for _ in range(m.nl)]
                hm[li] = hm[li].clone(); hm[li][hh] = 0.0
                l2, _ = heldout_loss(m, ch_te, voc, hmask=hm, positions=pos_te)
                s2 = sum_sens(m, ch, voc, hmask=hm, positions=top_list)
                drop = float(np.mean([sens[p] - s2[p] for p in s2])) if s2 else 0.0
                heads['%d.%d' % (li, hh)] = {'d': {k: l2[k] - b2[k] for k in l2 if k in b2}, 'sensdrop': drop}
        res['heads'] = heads
        res['top_mean'] = float(np.mean([sens[p] for p in top])) if top else 0.0
    return res


# ------------------------------------------------------------------ cycle 2: function vectors by transplant
@torch.no_grad()
def function_vectors(m, ch, voc, nhost=120, seedv=0):
    """Transplant every word type w (count >= 3) into fixed host slots and read what the model expects
    of the adjacent number. Hosts: NEXT = word followed by a number; PREV = word preceded by a number.
    fv(w) = [log mean bucket probs of the masked neighbour number (13), sum sensitivity of it] x 2 dirs."""
    r = random.Random(seedv)
    nxt, prv = [], []
    for ci, (di, ids, raw) in enumerate(ch):
        for j in range(len(ids)):
            if raw[j][0] == 'T' and not raw[j][1].startswith('L:'):
                if j + 1 < len(ids) and raw[j + 1][0] == 'N':
                    nxt.append((ci, j, j + 1))
                if j > 0 and raw[j - 1][0] == 'N':
                    prv.append((ci, j, j - 1))
    r.shuffle(nxt); r.shuffle(prv)
    hosts = {'next': nxt[:nhost], 'prev': prv[:nhost]}
    words = sorted(w for w, c in voc.count.items() if c >= 3 and w in voc.stoi)
    nb = len(BUCKETS)
    out = {}
    for w in words:
        wid = voc.stoi[w]
        vec = []
        for dname in ('next', 'prev'):
            H = hosts[dname]
            if not H:
                vec += [0.0] * (nb + 1); continue
            xs, xm, cols = [], [], []
            for ci, j, t in H:
                ids = list(ch[ci][1]); ids[j] = wid; ids[t] = 1
                mod = list(ids)
                for k in range(t):
                    if voc.kind(ids[k]) == 2:
                        o = ids[k] - voc.num0
                        mod[k] = voc.num0 + 2 * min(o // 2 + 2, nb - 1) + o % 2
                xs.append(ids); xm.append(mod); cols.append(t)
            idx = torch.arange(len(xs)); cols = torch.tensor(cols)
            la = m(pad(xs))[idx, cols]
            lb = m(pad(xm))[idx, cols]
            p = la[:, voc.num0:voc.num0 + voc.nnum].softmax(-1).view(len(xs), nb, 2).sum(-1).mean(0)
            sens = (numexp(lb, voc) - numexp(la, voc)).mean()
            vec += [float(x) for x in torch.log(p + 1e-6)] + [float(sens)]
        out[w] = vec
    return out


def run_model2(cfg):
    torch.set_num_threads(1)
    docs = corpus(cfg['corpus'])
    voc = Vocab(docs)
    r = random.Random(cfg['seed'] * 7 + 1)
    idx = list(range(len(docs))); r.shuffle(idx)
    ntr = int(0.8 * len(idx))
    tr_docs = set(idx[:ntr])
    ch = chunks(docs, voc)
    ch_tr = [c for c in ch if c[0] in tr_docs]
    ch_te = [c for c in ch if c[0] not in tr_docs]
    m = train_model(cfg, ch_tr, len(voc.itos), cfg['epochs'])
    res = {'cfg': cfg}
    res['loss'], res['n'] = heldout_loss(m, ch_te, voc)
    res['fv'] = function_vectors(m, ch, voc)
    res['train_docs'] = sorted(tr_docs)
    return res, m
