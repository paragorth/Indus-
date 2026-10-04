"""pe39: unsupervised machine translation between Proto-Elamite (PE) and
proto-cuneiform (PC) administrative lines, with numerals as the only shared
tokens.  Shared embedding + shared encoder, language-tagged shared decoder,
denoising autoencoding and online iterative back-translation.

Corpora (one 'sentence' = one line with >=1 sign and >=1 numeral):
  pe : data/pe_corpus.json          (signs base-normalised, 'x' dropped)
  pc : data/pe2_pc_corpus.json      (same format, CDLI Uruk IV/III admin)
  ur : data/pe38_ckpt/ur3_docs.json (Ur III admin lines, sys 1 count, 2 capacity)
Numeral bridges:
  ncode : one token per numeral code and count, e.g. 'N14:2' (PE and PC share codes)
  abs   : 'SYS:cnt|cap|oth' + quantile bin of the value inside corpus x system
"""
import json, os, sys, random, math, re, collections, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe39_ckpt')
os.makedirs(CK, exist_ok=True)

W = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000,
     'N02': .2, 'N08': .1, 'N08A': .1, 'N39B': .2, 'N24': .2, 'N30C': .1, 'N30D': .05,
     'N39C': .1, 'N39A': .3, 'N28': .05, 'N29B': .1, 'N51': 1, 'N54': 10, 'N46': 3}


def _cls_pe(numerals):
    s = common.system_of(numerals)
    if s == 'SDB':
        return 'cnt'
    if s in ('C', 'C*', 'S-frac', 'B'):
        return 'cap'
    return 'oth'


def _val(numerals):
    return sum(n * W.get(c.split('@')[0], 1.0) for n, c in numerals)


def load_script(name):
    fn = {'pe': 'pe_corpus.json', 'pc': 'pe2_pc_corpus.json'}[name]
    T = json.load(open(os.path.join(DATA, fn)))
    out = []
    for t in T:
        for l in t['lines']:
            num = [[n if isinstance(n, int) else 1, common.norm_code(c)] for n, c in l['numerals']]
            if not num:
                continue
            if name == 'pe':
                sg = [common.base(s) for s in l['signs'] if common.is_sign(s)]
            else:
                sg = [common.base(s) for s in l['signs'] if s not in ('x', 'X', '...')]
                sg = [s for s in sg if s and (not re.match(r'^N\d', s) or s in ('N57', 'N58'))]
            if not sg:
                continue
            out.append({'tab': t['id'], 'signs': sg[:8], 'num': num, 'cls': _cls_pe(num), 'val': _val(num)})
    return out


def _fval(v):
    try:
        if '/' in v:
            a, b = v.split('/')
            return float(a) / float(b)
        return float(v)
    except Exception:
        return None


def load_ur3(max_lines=None, seed=0):
    d = json.load(open(os.path.join(CK if os.path.exists(os.path.join(CK, 'ur3_docs.json'))
                                    else os.path.join(DATA, 'pe38_ckpt'), 'ur3_docs.json')))
    rng = random.Random(seed)
    rng.shuffle(d)
    out = []
    for doc in d:
        for l in doc['lines']:
            if l['sys'] not in (1, 2) or l['val'] is None:
                continue
            v = _fval(l['val'])
            w = [x for x in l['toks'] if x]
            if v is None or not w:
                continue
            out.append({'tab': doc['id'], 'signs': w[:8], 'num': None,
                        'cls': 'cnt' if l['sys'] == 1 else 'cap', 'val': v})
        if max_lines and len(out) >= max_lines:
            break
    return out


def numeral_tokens(lines, mode):
    """Attach 'ntoks' to each line."""
    if mode == 'abs':
        by = collections.defaultdict(list)
        for l in lines:
            by[l['cls']].append(l['val'])
        qs = {k: np.quantile(v, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]) for k, v in by.items()}
        for l in lines:
            b = int(np.searchsorted(qs[l['cls']], l['val'], side='right'))
            l['ntoks'] = ['SYS:' + l['cls'], 'Q:%d' % b]
    else:
        for l in lines:
            l['ntoks'] = ['%s:%d' % (c, min(n, 9)) for n, c in l['num']]
    return lines


def tablet_split(lines, frac=0.15, seed=0):
    tabs = sorted({l['tab'] for l in lines})
    rng = random.Random(seed)
    rng.shuffle(tabs)
    ho = set(tabs[:int(len(tabs) * frac)])
    return [l for l in lines if l['tab'] not in ho], [l for l in lines if l['tab'] in ho]


def subsample_tablets(lines, n_lines, seed):
    tabs = sorted({l['tab'] for l in lines})
    rng = random.Random(seed)
    rng.shuffle(tabs)
    by = collections.defaultdict(list)
    for l in lines:
        by[l['tab']].append(l)
    out = []
    for t in tabs:
        out += by[t]
        if len(out) >= n_lines:
            break
    return out


def shuffle_numerals(lines, seed):
    """Control: numeral groups permuted across all lines (signs kept)."""
    rng = random.Random(seed)
    nt = [l['ntoks'] for l in lines]
    rng.shuffle(nt)
    return [dict(l, ntoks=n) for l, n in zip(lines, nt)]


def rename(lines, prefix):
    return [dict(l, signs=[prefix + s for s in l['signs']]) for l in lines]


# ---------------------------------------------------------------- vocab
class Vocab:
    def __init__(self, langs, min_count=3):
        """langs: {lang: list of lines (training only)}"""
        self.itos = ['<pad>', '<eos>', '<blank>'] + ['<2%s>' % L for L in langs]
        self.lang_tok = {L: 3 + i for i, L in enumerate(langs)}
        self.lang_of = {}
        cnt = {L: collections.Counter(s for l in ls for s in l['signs']) for L, ls in langs.items()}
        ncnt = collections.Counter(t for ls in langs.values() for l in ls for t in l['ntoks'])
        for L in langs:
            self.itos.append('%s|UNK' % L)
            for s, c in sorted(cnt[L].items()):
                if c >= min_count:
                    self.itos.append('%s|%s' % (L, s))
        self.num_start = len(self.itos)
        for t, c in sorted(ncnt.items()):
            self.itos.append('#' + t)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.langs = list(langs)
        self.mask = {}
        for L in langs:
            m = np.zeros(len(self.itos), bool)
            for i, s in enumerate(self.itos):
                if s.startswith(L + '|') or s.startswith('#'):
                    m[i] = True
            m[1] = True
            self.mask[L] = m

    def enc(self, L, line):
        ids = [self.stoi.get('%s|%s' % (L, s), self.stoi['%s|UNK' % L]) for s in line['signs']]
        ids += [self.stoi['#' + t] for t in line['ntoks'] if '#' + t in self.stoi]
        return ids

    def is_sign(self, i):
        s = self.itos[i]
        return '|' in s and not s.endswith('|UNK')


# ---------------------------------------------------------------- model
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)


class GRUAttn(nn.Module):
    def __init__(self, V, emb=64, hid=128, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)
        self.dec = nn.GRU(emb + 2 * hid, hid, batch_first=True)
        self.att = nn.Linear(hid, 2 * hid, bias=False)
        self.out = nn.Linear(hid + 2 * hid, V)
        self.drop = nn.Dropout(drop)

    def encode(self, x):
        e = self.drop(self.emb(x))
        h, _ = self.enc(e)
        m = (x != 0)
        hm = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return h, m, torch.tanh(self.bridge(hm)).unsqueeze(0)

    def step(self, y_prev, s, H, M, ctx):
        e = self.drop(self.emb(y_prev))
        o, s = self.dec(torch.cat([e, ctx], -1).unsqueeze(1), s)
        o = o.squeeze(1)
        sc = torch.bmm(H, self.att(o).unsqueeze(-1)).squeeze(-1).masked_fill(~M, -1e9)
        a = F.softmax(sc, -1)
        ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)
        return self.out(torch.cat([o, ctx], -1)), s, ctx

    def forward(self, x, y_in):
        H, M, s = self.encode(x)
        ctx = torch.zeros(x.size(0), H.size(-1))
        outs = []
        for t in range(y_in.size(1)):
            lo, s, ctx = self.step(y_in[:, t], s, H, M, ctx)
            outs.append(lo)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def greedy(self, x, ltok, mask, maxlen=12, ban=None):
        H, M, s = self.encode(x)
        B = x.size(0)
        ctx = torch.zeros(B, H.size(-1))
        y = torch.full((B,), ltok, dtype=torch.long)
        mk = torch.tensor(~mask)
        if ban is not None:
            mk = mk.clone()
            mk[ban] = True
        out = []
        done = torch.zeros(B, dtype=torch.bool)
        for t in range(maxlen):
            lo, s, ctx = self.step(y, s, H, M, ctx)
            lo = lo.masked_fill(mk, -1e9)
            if t == 0:
                lo[:, 1] = -1e9
            y = lo.argmax(-1)
            y = y.masked_fill(done, 0)
            out.append(y)
            done = done | (y == 1)
            if done.all():
                break
        return torch.stack(out, 1)


class TinyTF(nn.Module):
    """Small transformer encoder-decoder with the same interface."""

    def __init__(self, V, emb=96, hid=None, drop=0.1, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.pos = nn.Embedding(32, emb)
        self.tf = nn.Transformer(emb, heads, layers, layers, 2 * emb, drop, batch_first=True)
        self.out = nn.Linear(emb, V)

    def _e(self, x):
        p = torch.arange(x.size(1)).unsqueeze(0)
        return self.emb(x) + self.pos(p)

    def forward(self, x, y_in):
        T = y_in.size(1)
        cm = torch.triu(torch.full((T, T), float('-inf')), 1)
        h = self.tf(self._e(x), self._e(y_in), tgt_mask=cm, src_key_padding_mask=(x == 0),
                    tgt_key_padding_mask=(y_in == 0), memory_key_padding_mask=(x == 0))
        return self.out(h)

    @torch.no_grad()
    def greedy(self, x, ltok, mask, maxlen=12, ban=None):
        B = x.size(0)
        mem = self.tf.encoder(self._e(x), src_key_padding_mask=(x == 0))
        y = torch.full((B, 1), ltok, dtype=torch.long)
        mk = torch.tensor(~mask)
        if ban is not None:
            mk = mk.clone()
            mk[ban] = True
        done = torch.zeros(B, dtype=torch.bool)
        for t in range(maxlen):
            T = y.size(1)
            cm = torch.triu(torch.full((T, T), float('-inf')), 1)
            h = self.tf.decoder(self._e(y), mem, tgt_mask=cm, memory_key_padding_mask=(x == 0))
            lo = self.out(h[:, -1]).masked_fill(mk, -1e9)
            if t == 0:
                lo[:, 1] = -1e9
            nx = lo.argmax(-1).masked_fill(done, 0)
            y = torch.cat([y, nx.unsqueeze(1)], 1)
            done = done | (nx == 1)
            if done.all():
                break
        return y[:, 1:]


# ---------------------------------------------------------------- training
def pad(seqs, maxlen=14):
    L = min(maxlen, max(len(s) for s in seqs))
    a = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs):
        s = s[:L]
        a[i, :len(s)] = s
    return torch.tensor(a)


def strip(ids):
    out = []
    for i in ids:
        if i in (0, 1):
            break
        out.append(int(i))
    return out


def noise(s, rng, pdrop=0.1, pblank=0.1, k=3):
    s2 = [t for t in s if rng.random() > pdrop] or s[:1]
    s2 = [2 if rng.random() < pblank else t for t in s2]
    keys = [i + rng.uniform(0, k) for i in range(len(s2))]
    return [t for _, t in sorted(zip(keys, s2))]


def seq_loss(model, src, tgt, ltok):
    x = pad(src)
    y = pad([[ltok] + t + [1] for t in tgt])
    lo = model(x, y[:, :-1])
    return F.cross_entropy(lo.reshape(-1, lo.size(-1)), y[:, 1:].reshape(-1), ignore_index=0)


def translate(model, V, seqs, L, bs=256, ban=None):
    model.eval()
    out = []
    for i in range(0, len(seqs), bs):
        g = model.greedy(pad(seqs[i:i + bs]), V.lang_tok[L], V.mask[L], ban=ban)
        out += [strip(r.tolist()) or [V.stoi['%s|UNK' % L]] for r in g]
    return out


def train(cfg, data, V, log=None):
    """data: {'A': (langA, train seqs), 'B': (langB, train seqs)}"""
    seed = cfg['seed']
    torch.manual_seed(seed)
    rng = random.Random(seed)
    Arch = {'gru': GRUAttn, 'tf': TinyTF}[cfg.get('arch', 'gru')]
    model = Arch(len(V.itos), **cfg.get('akw', {}))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 2e-3))
    (La, Sa), (Lb, Sb) = data['A'], data['B']
    bs = cfg.get('bs', 64)
    steps = cfg.get('steps', 1500)
    ae_only = cfg.get('ae_steps', 400)
    t0 = time.time()
    for st in range(steps):
        model.train()
        ba = rng.sample(Sa, bs)
        bb = rng.sample(Sb, bs)
        loss = seq_loss(model, [noise(s, rng) for s in ba], ba, V.lang_tok[La]) + \
            seq_loss(model, [noise(s, rng) for s in bb], bb, V.lang_tok[Lb])
        if st >= ae_only:
            ya = translate(model, V, ba, Lb)
            yb = translate(model, V, bb, La)
            model.train()
            loss = loss + cfg.get('bt_w', 1.0) * (seq_loss(model, ya, ba, V.lang_tok[La]) +
                                                  seq_loss(model, yb, bb, V.lang_tok[Lb]))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and st % 500 == 0:
            log('  step %d loss %.3f (%.0fs)' % (st, loss.item(), time.time() - t0))
    model.eval()
    return model


@torch.no_grad()
def roundtrip_nll(model, V, seqs, L1, L2, ban=None, bs=256):
    """mean per-token NLL of reconstructing L1 seqs from their L2 translation."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(seqs), bs):
        s = seqs[i:i + bs]
        y = translate(model, V, s, L2, ban=ban)
        x = pad(y)
        t = pad([[V.lang_tok[L1]] + q + [1] for q in s])
        lo = model(x, t[:, :-1])
        l = F.cross_entropy(lo.reshape(-1, lo.size(-1)), t[:, 1:].reshape(-1), ignore_index=0, reduction='sum')
        tot += l.item()
        n += int((t[:, 1:] != 0).sum())
    return tot / max(n, 1)


def lexicon(V, src, out, min_sup=5):
    """For each source sign s: target sign t maximising P(t|s) - P(t) over the
    translated corpus. Returns {s: (t, P(t|s), P(t), n_s)}."""
    cs = collections.Counter()
    ct = collections.Counter()
    cst = collections.Counter()
    N = len(src)
    for a, b in zip(src, out):
        A = {i for i in a if V.is_sign(i)}
        B = {i for i in b if V.is_sign(i)}
        for i in A:
            cs[i] += 1
        for j in B:
            ct[j] += 1
        for i in A:
            for j in B:
                cst[i, j] += 1
    best = {}
    for (i, j), c in cst.items():
        if cs[i] < min_sup:
            continue
        sc = c / cs[i] - ct[j] / N
        if i not in best or sc > best[i][1]:
            best[i] = (j, sc, c / cs[i], ct[j] / N)
    return {V.itos[i]: (V.itos[j], round(sc, 4), round(p, 3), round(q, 3), cs[i]) for i, (j, sc, p, q) in best.items()}


def pair_gain(model, V, ho, L1, L2, src_tok, tgt_tok, freq_tgt, rng, nrand=3):
    """Held-out gain of a pair: round-trip NLL rise on held-out L1 lines containing
    src_tok when tgt_tok is banned from the L2 output, minus the mean rise when a
    frequency-matched random L2 sign is banned."""
    si = V.stoi[src_tok]
    sub = [s for s in ho if si in s]
    if len(sub) < 3:
        return None
    base = roundtrip_nll(model, V, sub, L1, L2)
    d = roundtrip_nll(model, V, sub, L1, L2, ban=[V.stoi[tgt_tok]]) - base
    cands = sorted(freq_tgt, key=lambda t: abs(math.log(freq_tgt[t] + 1) - math.log(freq_tgt.get(tgt_tok, 1) + 1)))
    cands = [t for t in cands if t != tgt_tok][:12]
    rs = [roundtrip_nll(model, V, sub, L1, L2, ban=[V.stoi[t]]) - base for t in rng.sample(cands, min(nrand, len(cands)))]
    return round(d - float(np.mean(rs)), 4), round(d, 4), len(sub)


def jdump(o, fn):
    with open(fn, 'w') as f:
        json.dump(o, f)
