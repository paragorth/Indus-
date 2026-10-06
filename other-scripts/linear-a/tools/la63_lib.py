"""la63 WHAT WOULD THE MINOAN CLERK MISS? The pe51 deletion game ported to Linear A.

Every corpus is reduced to one value-free shape (opaque tokens only):
  doc = {'id', 'ents': [{'kind': 'ctx'|'num', 'toks': [str], 'mag': int|None, 'frac': str|None,
                         'tot': 0|1|None}]}
  An ENTRY is a run of word/logogram tokens closed by one number (Linear A, Linear B: the number
  follows its words) or one written line (Ur III: the number comes first). Entries without a
  number are 'ctx'. Logogram tokens start with 'L:'. mag = log2 bucket of the integer part;
  frac = fraction class (opaque string; Linear A fraction letters are never valued); tot = 1 if the
  entry's number equals the running sum of >= 2 entries above it (arithmetic only, never the sign).

Small random models (transformer or bi-GRU) fill in a partly masked document:
  NUM  : every numeral hidden -> MAG, FRAC and TOT of every numeral entry
  LOGO : every logogram hidden -> its commodity base (class before '+')
  ENT  : the non-logogram tokens of alternate numeral entries hidden -> their identity (2 patterns)
  HEAD : alternate ctx tokens hidden -> their identity (2 patterns)
Deletion game: on held-out documents a word type is replaced by <del> everywhere (its own target
slots are never scored); the change of every other loss is its loss profile; the null is as many
random OTHER tokens erased in the same documents (R draws). No Linear B sound value is used for
Linear A anywhere; Linear B and Ur III role keys are used only to score the controls.
"""
import collections, json, math, os, random, re, sys, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la63_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

NB = 14


def hseed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def magb(v):
    if v is None:
        return None
    v = int(v)
    if v <= 0:
        return 0
    return int(min(NB - 1, math.floor(math.log2(v)) + 1))


def lbase(x):
    b = re.sub(r"[\[\]'?*]", '', x[2:].split('+')[0]).strip() if x.startswith('L:') else x
    return b or x


# ------------------------------------------------------------------ segmentation
def _close(total, ent):
    """Linear A / Linear B running-sum test on integer parts; fractions allow slack."""
    s = sum(e[0] for e in ent)
    nfr = sum(1 for e in ent if e[1])
    if nfr == 0:
        return total == s
    return s <= total <= s + nfr // 2 + 1


def seg_after(toks):
    """toks: la60 format [kind, val, frac] with kinds W, L, N, NL. Number closes an entry."""
    E, cur = [], []
    for k, v, f in toks:
        if k == 'W':
            cur.append(v)
        elif k == 'L':
            cur.append('L:' + v)
        elif k == 'N':
            E.append(dict(kind='num', toks=cur[:10], val=int(v), frac=(f or ''), tot=0))
            cur = []
        elif k == 'NL':
            if cur:
                E.append(dict(kind='ctx', toks=cur[:10], val=None, frac=None, tot=None))
            cur = []
    if cur:
        E.append(dict(kind='ctx', toks=cur[:10], val=None, frac=None, tot=None))
    return E


def mark_totals(E):
    acc = []
    for e in E:
        if e['kind'] != 'num':
            continue
        if len(acc) >= 2 and e['val'] > max(a[0] for a in acc) and _close(e['val'], acc):
            e['tot'] = 1
            acc = []
        else:
            acc.append((e['val'], e['frac']))


def finish(E):
    mark_totals(E)
    for e in E:
        e['mag'] = magb(e['val']) if e['kind'] == 'num' else None
        e.pop('val', None)
    return E


# ------------------------------------------------------------------ corpora
def build_la():
    fn = os.path.join(CK, 'la.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import la60_common as C
    docs = C.admin_docs(C.load_la())
    out = []
    for d in docs:
        E = finish(seg_after(d['toks']))
        if sum(e['kind'] == 'num' for e in E) >= 1 and sum(len(e['toks']) for e in E) >= 1:
            out.append(dict(id=d['id'], site=d['site'], ents=E[:40]))
    json.dump(out, open(fn, 'w'))
    return out


def build_lb(n=700):
    fn = os.path.join(CK, 'lb.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import la60_common as C
    docs = C.load_lb()
    out = []
    for d in docs:
        E = finish(seg_after(d['toks']))
        if sum(e['kind'] == 'num' for e in E) >= 1:
            out.append(dict(id=d['id'], site=d['site'], ents=E[:40]))
    random.Random(63).shuffle(out)
    out = out[:n]
    json.dump(out, open(fn, 'w'))
    return out


def build_ur3(n=500):
    """Ur III admin (la57 cache): one entry per line; COM-key words become logograms (the control's
    'commodity' slot, since Ur III writes goods as words)."""
    fn = os.path.join(CK, 'ur3.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    sys.path.insert(0, HERE)
    import la57_common as K
    docs = K.ur3_docs()
    tr = K.meso_truth(docs)
    out = []
    for d in docs:
        E, cur, num = [], [], None
        for x in list(d['toks']) + [('L',)]:
            if x[0] == 'L':
                if cur or num is not None:
                    if num is not None and num[1] is not None:
                        E.append(dict(kind='num', toks=cur[:8], val=int(num[1]), frac='f' if num[2] else '', tot=0))
                    elif num is None:
                        E.append(dict(kind='ctx', toks=cur[:8], val=None, frac=None, tot=None))
                cur, num = [], None
            elif x[0] == 'N':
                if num is None:
                    num = x
            else:
                w = x[1]
                cur.append(('L:' + w) if tr.get(w) == 'COM' else w)
        E = finish(E)
        if 2 <= sum(e['kind'] == 'num' for e in E) and len(E) <= 40:
            out.append(dict(id=d['id'], site=d.get('site', ''), ents=E))
    random.Random(63).shuffle(out)
    out = out[:n]
    json.dump(dict(docs=out, truth=tr), open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    return json.load(open(fn))


# answer keys (controls only)
LB_KEY_W = {
    'TOT': 'TO-SO TO-SA TO-SO-DE TO-SA-DE TO-SO-PA',
    'TRA': 'A-PU-DO-SI DE-KA-SA-TO DO-SO-MO E-KO-SI E-KE A-KE-RE DI-DO-SI O-U-DI-DO-SI A-PE-DO-KE '
           'A-PE-E-SI PA-RO O-DA-A2 E-E-SI O-PE-RO E-KE-QE',
    'PLA': 'PA-I-TO KU-DO-NI-JA A-MI-NI-SO KO-NO-SO TU-RI-SO RU-KI-TO DA-WO E-RA SU-RI-MO PU-RO PA-KI-JA-NE '
           'RO-U-SO KA-RA-DO-RO RI-JO TI-MI-TO-A-KO A-PU2-WE E-RA-TO PE-TO-NO ME-TA-PA ZA-MA-E-WI-JA RI-JO-NO '
           'KU-TA-TO QA-MO SI-JA-DU-WE RA-TO E-KO-ME-NO KA-RU-NO U-TA-NO A-KA-WO-NE TI-RI-TO KO-TU-WE '
           'PA-RA-JA DA-*22-TO DO-TI-JA RA-SU-TO TU-NI-JA E-RA-JO KA-TO-RO QA-RA',
    # commodity-bound qualifiers and measured nouns (children with women, old/new, purple, land, seed, metals)
    'QUAL': 'KO-WO KO-WA PA-RA-JO NE-WA PE-RU-SI-NU-WO PO-NI-KI-JA PA-WE-A TU-NA-NO E-RA3-WO KU-PA-RO '
            'KO-RI-JA-DO-NO KI-TA-NO E-RI-KA KA-KO KU-RU-SO A-RA-RU-JA ME-ZO ME-U-JO ME-ZO-E ME-WI-JO '
            'ME-WI-JO-E PO-ME RE-U-KO KI-TO A-NI-JA-PI PE-MO KO-TO-NA KE-KE-ME-NA KI-TI-ME-NA O-NA-TO',
}


def lb_key(C):
    K = {w: r for r, ws in LB_KEY_W.items() for w in ws.split()}
    first, tot = collections.Counter(), collections.Counter()
    for d in C:
        m = re.match(r'^KN\s+(D[a-z])', d['id'])
        ws = [x for e in d['ents'] for x in e['toks']]
        for i, w in enumerate(ws):
            tot[w] += 1
            if m and i == 0 and not w.startswith('L:'):
                first[w] += 1
    out = {}
    for w, c in tot.items():
        if w.startswith('L:'):
            out[w] = 'COM'
        elif w in K:
            out[w] = K[w]
        elif first[w] >= max(1, 0.5 * c):
            out[w] = 'PER'
    return out


def ur3_key(truth, C):
    out = {}
    for d in C:
        for e in d['ents']:
            for w in e['toks']:
                if w.startswith('L:'):
                    out[w] = 'COM'
                elif w in truth:
                    out[w] = truth[w]
    return out


# ------------------------------------------------------------------ plant (into real Linear A)
def build_plant(seed=0):
    """Linear A admin docs plus planted tokens with known roles:
      P_COM, P_COM4 : inserted into entries of documents that hold VIN (commodity-bound) -> LOGO
      P_UNIT, P_UNIT4 : inserted into entries whose fraction is then forced to class 'E' -> FRAC/NUM
      P_N0..P_N5    : inserted into random entries (names) -> SILENT
    Counts ~20 (P_COM, P_UNIT, names) and 4 (the *4 tokens, la57 candidate frequency)."""
    C = json.loads(json.dumps(build_la()))
    r = random.Random(seed)
    vin = [d for d in C if any(t.startswith('L:VIN') for e in d['ents'] for t in e['toks'])]
    allnum = [(d, e) for d in C for e in d['ents'] if e['kind'] == 'num']

    def put(e, tok):
        e['toks'] = list(e['toks'])
        e['toks'].insert(r.randrange(len(e['toks']) + 1), tok)

    for tok, n in (('P_COM', 20), ('P_COM4', 4)):
        cand = [(d, e) for d in vin for e in d['ents'] if e['kind'] == 'num']
        for d, e in r.sample(cand, min(n, len(cand))):
            put(e, tok)
    for tok, n in (('P_UNIT', 20), ('P_UNIT4', 4)):
        for d, e in r.sample(allnum, n):
            put(e, tok)
            e['frac'] = 'E'
    for i in range(6):
        for d, e in r.sample(allnum, 20 if i < 4 else 4):
            put(e, 'P_N%d' % i)
    return C


def build_plant_dose(seed=5):
    """Dose-response plant: unit markers (each forces its own fraction class) and commodity-bound
    markers (each only in documents of one commodity) at 10, 20 and 40 tokens; names at 10/20/40."""
    C = json.loads(json.dumps(build_la()))
    r = random.Random(seed)
    allnum = [(d, e) for d in C for e in d['ents'] if e['kind'] == 'num']

    def put(e, tok):
        e['toks'] = list(e['toks'])
        e['toks'].insert(r.randrange(len(e['toks']) + 1), tok)

    for (n, com, fr) in ((10, 'VIN', 'K'), (20, 'CYP', 'B'), (40, 'GRA', 'E')):
        docs = [d for d in C if any(t.startswith('L:' + com) for e in d['ents'] for t in e['toks'])]
        cand = [(d, e) for d in docs for e in d['ents'] if e['kind'] == 'num']
        for d, e in r.sample(cand, min(n, len(cand))):
            put(e, 'D_COM%d' % n)
        for d, e in r.sample(allnum, n):
            put(e, 'D_UNIT%d' % n)
            e['frac'] = fr
        for i in range(2):
            for d, e in r.sample(allnum, n):
                put(e, 'D_N%d_%d' % (n, i))
    return C


DOSE_KEY = {**{'D_COM%d' % n: 'LOGO' for n in (10, 20, 40)}, **{'D_UNIT%d' % n: 'NUM' for n in (10, 20, 40)},
            **{'D_N%d_%d' % (n, i): 'SILENT' for n in (10, 20, 40) for i in range(2)}}

PLANT_KEY = {'P_COM': 'LOGO', 'P_COM4': 'LOGO', 'P_UNIT': 'NUM', 'P_UNIT4': 'NUM',
             **{'P_N%d' % i: 'SILENT' for i in range(6)}}


# ------------------------------------------------------------------ nulls
def null_within_doc(C, seed):
    """Non-logogram tokens re-dealt across the entries of the same document (entry lengths and
    logogram positions kept): what survives is document-level, not entry-level."""
    r = random.Random(seed)
    out = []
    for d in C:
        pool = [x for e in d['ents'] for x in e['toks'] if not x.startswith('L:')]
        r.shuffle(pool)
        k, E = 0, []
        for e in d['ents']:
            e2 = dict(e)
            t2 = []
            for x in e['toks']:
                if x.startswith('L:'):
                    t2.append(x)
                else:
                    t2.append(pool[k]); k += 1
            e2['toks'] = t2
            E.append(e2)
        out.append(dict(d, ents=E))
    return out


# ------------------------------------------------------------------ tensors
SPECIAL = ['<pad>', '<unk>', '<mask>', '<del>', '<num>']
NSPEC = len(SPECIAL)


class Vocab:
    def __init__(self, C, min_n=2):
        c = collections.Counter(x for d in C for e in d['ents'] for x in e['toks'])
        self.itos = SPECIAL + sorted(w for w, k in c.items() if k >= min_n)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.count = c
        fr = collections.Counter(e['frac'] for d in C for e in d['ents'] if e['kind'] == 'num')
        top = [f for f, _ in fr.most_common() if f][:7]
        self.frac = {'': 0, **{f: i + 1 for i, f in enumerate(top)}}
        self.nfrac = len(top) + 2  # + 'other'
        lb = collections.Counter(lbase(x) for d in C for e in d['ents'] for x in e['toks'] if x.startswith('L:'))
        topl = [b for b, k in lb.most_common() if k >= 3][:24]
        self.logo = {b: i for i, b in enumerate(topl)}
        self.nlogo = len(topl) + 1

    def id(self, w):
        return self.stoi.get(w, 1)

    def fid(self, f):
        if f is None:
            return -1
        return self.frac.get(f, self.nfrac - 1)

    def lid(self, w):
        return self.logo.get(lbase(w), self.nlogo - 1)


MAXLEN = 200


def encode(d, V):
    tok, kind, li, pos, mag, frac, tot, isnum, logo, word = [], [], [], [], [], [], [], [], [], []
    for i, e in enumerate(d['ents']):
        for j, w in enumerate(e['toks']):
            tok.append(V.id(w)); kind.append(0 if e['kind'] == 'ctx' else 1); li.append(min(i, 39)); pos.append(min(j, 10))
            mag.append(-1); frac.append(-1); tot.append(-1); isnum.append(0)
            logo.append(V.lid(w) if w.startswith('L:') else -1); word.append(w)
        if e['kind'] == 'num':
            tok.append(4); kind.append(2); li.append(min(i, 39)); pos.append(11)
            mag.append(e['mag'] if e['mag'] is not None else -1); frac.append(V.fid(e['frac']))
            tot.append(e['tot'] if e['tot'] is not None else -1); isnum.append(1); logo.append(-1); word.append(None)
    n = min(len(tok), MAXLEN)
    return dict(tok=tok[:n], kind=kind[:n], li=li[:n], pos=pos[:n], mag=mag[:n], frac=frac[:n], tot=tot[:n],
                isnum=isnum[:n], logo=logo[:n], word=word[:n])


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self, nv, nfrac, nlogo, arch):
        super().__init__()
        d = arch['d']
        self.arch = arch
        self.emb = nn.Embedding(nv, d)
        self.kind = nn.Embedding(3, d)
        self.li = nn.Embedding(40, d)
        self.pos = nn.Embedding(12, d)
        self.fracE = nn.Embedding(nfrac + 1, d)
        self.magE = nn.Embedding(NB + 1, d)
        if arch['type'] == 'tf':
            layer = nn.TransformerEncoderLayer(d, arch['heads'], 2 * d, arch['drop'], batch_first=True)
            self.enc = nn.TransformerEncoder(layer, arch['layers'], enable_nested_tensor=False)
        else:
            self.enc = nn.GRU(d, d // 2, arch['layers'], batch_first=True, bidirectional=True,
                              dropout=arch['drop'] if arch['layers'] > 1 else 0)
        self.out_tok = nn.Linear(d, nv)
        self.out_mag = nn.Linear(d, NB)
        self.out_frac = nn.Linear(d, nfrac)
        self.out_tot = nn.Linear(d, 2)
        self.out_logo = nn.Linear(d, nlogo)

    def forward(self, b):
        x = self.emb(b['tok']) + self.kind(b['kind']) + self.li(b['li']) + self.pos(b['pos'])
        fv = b['frac_in'].clamp(min=-1) + 1
        mv = b['mag_in'].clamp(min=-1) + 1
        x = x + (self.fracE(fv) + self.magE(mv)) * b['isnum'].unsqueeze(-1)
        pad = b['tok'] == 0
        if self.arch['type'] == 'tf':
            return self.enc(x, src_key_padding_mask=pad)
        h, _ = self.enc(x)
        return h


KEYS_T = ('tok', 'kind', 'li', 'pos', 'isnum', 'mag_in', 'frac_in', 'mag', 'frac', 'tot', 'tgt', 'lgt')


def collate(items):
    n = max(len(it['tok']) for it in items)
    out = {}
    for k in KEYS_T:
        fill = 0 if k in ('tok', 'kind', 'li', 'pos', 'isnum') else -1
        out[k] = torch.tensor([it[k] + [fill] * (n - len(it[k])) for it in items], dtype=torch.long)
    return out


VIEWS = [('num', 0), ('logo', 0), ('ent', 0), ('ent', 1), ('head', 0), ('head', 1)]


def make_view(e, mode, rng=None, pattern=0, delmask=None, pdel=0.0):
    tok = list(e['tok']); n = len(tok)
    mag_in = list(e['mag']); frac_in = list(e['frac'])
    tgt = [-1] * n; lgt = [-1] * n
    tm = [-1] * n; tf = [-1] * n; tt = [-1] * n
    if mode == 'num':
        for i in range(n):
            if e['isnum'][i]:
                mag_in[i] = -1; frac_in[i] = -1
                tm[i] = e['mag'][i]; tf[i] = e['frac'][i]; tt[i] = e['tot'][i]
    elif mode == 'logo':
        for i in range(n):
            if e['logo'][i] >= 0:
                lgt[i] = e['logo'][i]; tok[i] = 2
    elif mode == 'ent':
        lines = sorted({e['li'][i] for i in range(n) if e['isnum'][i]})
        for k, L in enumerate(lines):
            hide = (rng.random() < 0.5) if rng else (k % 2 == pattern)
            if hide:
                for i in range(n):
                    if e['li'][i] == L and e['kind'][i] == 1 and e['logo'][i] < 0:
                        tgt[i] = tok[i]; tok[i] = 2
    elif mode == 'head':
        ctxp = [i for i in range(n) if e['kind'][i] == 0 and e['logo'][i] < 0]
        for k, i in enumerate(ctxp):
            hide = (rng.random() < 0.5) if rng else (k % 2 == pattern)
            if hide:
                tgt[i] = tok[i]; tok[i] = 2
    if pdel and rng:
        for i in range(n):
            if tgt[i] < 0 and lgt[i] < 0 and tok[i] >= NSPEC and rng.random() < pdel:
                tok[i] = 3
    if delmask:
        for i in delmask:
            if tgt[i] < 0 and lgt[i] < 0 and tok[i] != 4:
                tok[i] = 3
    tgt = [x if x >= NSPEC else -1 for x in tgt]
    return dict(tok=tok, kind=e['kind'], li=e['li'], pos=e['pos'], isnum=e['isnum'], mag_in=mag_in,
                frac_in=frac_in, mag=tm, frac=tf, tot=tt, tgt=tgt, lgt=lgt)


def losses(model, b):
    h = model(b)
    res = {}
    for k, head, key in (('MAG', model.out_mag, 'mag'), ('FRAC', model.out_frac, 'frac'), ('TOT', model.out_tot, 'tot'),
                         ('TOK', model.out_tok, 'tgt'), ('LOGO', model.out_logo, 'lgt')):
        m = b[key] >= 0
        if m.any():
            res[k] = (m.nonzero(), F.cross_entropy(head(h[m]), b[key][m], reduction='none'))
    return res


def random_arch(r):
    t = r.choice(['tf', 'tf', 'gru'])
    return dict(type=t, d=r.choice([32, 48, 64]), layers=r.choice([1, 2, 3]) if t == 'tf' else r.choice([1, 2]),
                heads=r.choice([2, 4]), drop=r.choice([0.0, 0.1, 0.2]), lr=r.choice([1e-3, 2e-3, 3e-3]),
                epochs=r.choice([30, 40, 50]), pdel=r.choice([0.05, 0.1, 0.15]))


def train(E_tr, V, arch, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Net(len(V.itos), V.nfrac, V.nlogo, arch)
    opt = torch.optim.Adam(model.parameters(), lr=arch['lr'])
    bs = 32
    for ep in range(arch['epochs']):
        model.train()
        idx = list(range(len(E_tr)))
        rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            items = [make_view(E_tr[i], rng.choice(['num', 'num', 'logo', 'ent', 'head']), rng=rng, pdel=arch['pdel'])
                     for i in idx[s:s + bs]]
            R = losses(model, collate(items))
            if not R:
                continue
            L = sum(v[1].mean() * (0.5 if k == 'TOK' else 1.0) for k, v in R.items())
            opt.zero_grad(); L.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_views(model, E, jobs, bs=256):
    out = [None] * len(jobs)
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        items = [make_view(E[ti], VIEWS[vi][0], pattern=VIEWS[vi][1], delmask=dm) for ti, vi, dm in chunk]
        R = losses(model, collate(items))
        acc = [dict() for _ in chunk]
        for k, (ix, lv) in R.items():
            for r_, c_, v in zip(ix[:, 0].tolist(), ix[:, 1].tolist(), lv.tolist()):
                kk = k if k != 'TOK' else ('HEAD' if VIEWS[chunk[r_][1]][0] == 'head' else 'ENT')
                acc[r_][(kk, c_)] = v
        out[s:s + len(chunk)] = acc
    return out
