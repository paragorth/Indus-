"""pe51 WHAT WOULD THE CLERK MISS? Deletion game on small predictive models.

Every corpus is reduced to one shape (opaque tokens only):
  tablet = {'id', 'lines': [{'kind': 'ctx'|'num', 'toks': [str], 'sys': str|None,
                              'mag': int|None, 'tot': 0|1|None}]}
  'ctx' = a line with no numeral (PE/PC header and other non-numeric lines, Ur III
  non-numeric lines incl. formula/footer lines); 'num' = a line carrying a numeral.
  sys = number-system class of the numeral, mag = log2 bucket of its value,
  tot = 1 if the line is the tablet total by a definition that does NOT look at
  the line's signs (layout or arithmetic, see builders).

Tasks (all predicted by one small model from a partly masked tablet):
  NUM forward : every numeral masked -> SYS, MAG and TOT of every numeral line
  HEAD forward: half of the ctx tokens masked (two complementary patterns) -> each
  ENT forward : alternate numeral lines' tokens masked (two patterns) -> each
Deletion game: a sign's tokens are replaced by DEL everywhere except a target slot;
the loss change on the other targets is that sign's loss profile.
"""
import collections, csv, json, math, os, random, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe51_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)

NB = 16  # magnitude buckets


def magb(v):
    if v is None or v <= 0:
        return None
    return int(min(NB - 1, max(0, math.floor(math.log2(v)) + 2)))


# ------------------------------------------------------------------ PE / PC
def _num_lines_pe(T, keep_variant=False):
    from common import base, is_sign, system_of
    from pe47_common import pe_value
    out = []
    for t in T:
        L = []
        for l in t['lines']:
            sg = [s if keep_variant else base(s) for s in l['signs'] if s != 'x' and (is_sign(s) or not s.startswith('N'))]
            sg = [s for s in sg if s and s != 'x']
            if l['numerals']:
                codes = [[n, c] for n, c in l['numerals']]
                sy = system_of(codes) if all(c for _, c in codes) else None
                pv = None
                tail = l['raw'].split(',')[-1] if ',' in l['raw'] else l['raw']
                if not ('...' in tail or '[' in tail or '?' in tail):
                    try:
                        pv = pe_value(codes)
                    except Exception:
                        pv = None
                L.append(dict(kind='num', toks=sg[:10], sys=sy, mag=magb(float(pv[0])) if pv else None,
                              tot=None, surf=l['surface']))
            elif sg:
                L.append(dict(kind='ctx', toks=sg[:10], sys=None, mag=None, tot=None, surf=l['surface']))
        # total by layout (same convention as earlier loops): exactly one numeral line off the
        # obverse, >= 2 obverse numeral lines, and it is the last numeral line
        nl = [l for l in L if l['kind'] == 'num']
        off = [l for l in nl if l['surf'] != 'obverse']
        ob = [l for l in nl if l['surf'] == 'obverse']
        for l in nl:
            l['tot'] = 0
        if len(off) == 1 and len(ob) >= 2 and off[0] is nl[-1]:
            off[0]['tot'] = 1
        if len(nl) < 2 or len(nl) > 40:
            continue
        for l in L:
            l.pop('surf', None)
        out.append(dict(id=t['id'], lines=L[:48]))
    return out


def build_pe():
    fn = os.path.join(CK, 'pe.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from common import load
    C = _num_lines_pe(load())
    json.dump(C, open(fn, 'w'))
    return C


def build_pc():
    fn = os.path.join(CK, 'pc.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    for t in T:
        for l in t['lines']:
            l['numerals'] = [[n, ('N01' if c == 'N1' else c)] for n, c in l['numerals']]
            l['signs'] = [re.sub(r'~[a-z0-9]+$', '', s) if not s.startswith('|') else s for s in l['signs']]
            l['signs'] = ['P_' + s for s in l['signs'] if s != 'x' and not re.match(r'^N\d', s)]
            if 'surface' not in l:
                l['surface'] = 'obverse'
    import common
    old = common.is_sign
    common.is_sign = lambda s: True
    try:
        C = _num_lines_pe(T)
    finally:
        common.is_sign = old
    json.dump(C, open(fn, 'w'))
    return C


# ------------------------------------------------------------------ Ur III
LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")
NUMSTART = re.compile(r"^(\d+(?:/\d+)?)\(")


def build_ur3(n=1500, seed=0):
    fn = os.path.join(CK, 'ur3.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from pe24_ur3 import parse_ur_line
    csv.field_size_limit(10 ** 9)
    ok = set()
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and ('Puzri' in row['provenience'] or 'Umma' in row['provenience']) \
                and row['genre'].startswith('Admin') and row['id_text'].isdigit():
            ok.add('P%06d' % int(row['id_text']))
    txt = open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace').read()
    res = []
    for d in re.split(r'\n(?=&P)', txt):
        pid = d[1:8]
        if pid not in ok:
            continue
        L, surf, bad = [], 'obverse', False
        for raw in d.split('\n'):
            if raw.startswith('@'):
                w = raw[1:].split()[0] if raw[1:].split() else ''
                if w in ('obverse', 'reverse', 'seal', 'envelope', 'left', 'edge', 'bottom', 'top'):
                    surf = w
                continue
            if surf in ('seal', 'envelope'):
                continue
            m = LINE.match(raw.strip())
            if not m:
                continue
            t = m.group(2)
            if '[' in t or '...' in t:
                bad = True
                continue
            t = re.sub(r'[#?!<>*]', '', t).strip()
            t = re.sub(r'^szu-?nigin2?\b', 'szu-nigin2', t)
            toks = t.split()
            if not toks:
                continue
            tt = re.sub(r'^szu-nigin2?\s+', '', t)
            if NUMSTART.match(tt):
                p = parse_ur_line(t)
                words = [w for w in toks if not NUMSTART.match(w) and w != 'la2']
                words = ['szu-nigin2' if w.startswith('szu-nigin') else w for w in words]
                if p is None:
                    L.append(dict(kind='num', toks=words[:8], sys=None, mag=None, val=None))
                else:
                    L.append(dict(kind='num', toks=words[:8], sys=p[3], mag=magb(float(p[0])), val=float(p[0])))
            else:
                L.append(dict(kind='ctx', toks=toks[:8], sys=None, mag=None, val=None))
        nl = [l for l in L if l['kind'] == 'num']
        if bad or len(nl) < 2 or len(nl) > 40:
            continue
        # total by arithmetic only: value = sum of the numeral lines since the previous total (>= 2)
        acc = []
        for l in nl:
            l['tot'] = 0
            vals = [a['val'] for a in acc]
            if len(acc) >= 2 and l['val'] is not None and None not in vals and \
                    all(a['sys'] == l['sys'] for a in acc) and abs(sum(vals) - l['val']) < 1e-6:
                l['tot'] = 1
                acc = []
            else:
                acc.append(l)
        for l in L:
            l.pop('val', None)
        res.append(dict(id=pid, lines=L[:48]))
    random.Random(seed).shuffle(res)
    res = res[:n]
    json.dump(res, open(fn, 'w'))
    return res


# answer key for Ur III (opaque in the models; used only to score the control)
UR3_KEY = {
    'commodity': ['udu', 'masz2', 'gu4', 'sze', 'kasz', 'ninda', 'i3', 'siki', 'u8', 'sila4', 'ab2', 'ansze',
                  'masz2-gal', 'ud5', 'kir11', 'dabin', 'zi3', 'gu4-niga', 'udu-niga', 'tug2', 'gesz', 'sa',
                  'ku6', 'munu4', 'zu2-lum', 'masz', 'amar', 'ga', 'dug', 'esir2', 'gi', 'niga', 'kusz', 'u2',
                  'gukkal', 'sila3-ga', 'gurusz', 'geme2', 'dumu', 'szim', 'kasz-saga', 'durah', 'szah2',
                  'u8-niga', 'sze-ba', 'mu-du-lum', 'dur3', 'eme6', 'mastu?'],
    'unit': ['sila3', 'gur', 'ma-na', 'gin2', 'sar', 'iku', 'gun2', 'bur3', 'ban2', 'gu2', 'kusz3',
             'gin2-ta', 'sila3-ta'],
    'total': ['szu-nigin2', 'sag-nig2-gur11-ra-kam', 'sza3-bi-ta', 'la2-ia3', 'zi-ga',
              'nig2-ka9-ak', 'gu2-an-sze3'],
    'doctype': ['mu-kux(DU)', 'ba-zi', 'i3-dab5', 'kiszib3', 'szu', 'ba-ti', 'sa2-du11', 'ba-usz2',
                'dab5-ba', 'mu-kux', 'gaba-ri', 'sa10', 'e2-kiszib3-ba', 'nig2-dab5', 'szu-ba-ti', 'giri3',
                'ki', 'iti', 'mu', 'u4', 'a2', 'ugula', 'kiszib3-ta'],
}


def ur3_names(C, min_n=4):
    """Name answer key by frame only: word after 'ki'/'kiszib3'/'giri3'/'ugula' or before
    'i3-dab5'/'szu ba-ti' (with -ta/-sze3 suffixes stripped), excluding known function words."""
    fn = collections.Counter()
    known = set(w for v in UR3_KEY.values() for w in v)
    for t in C:
        for l in t['lines']:
            w = l['toks']
            for i, x in enumerate(w):
                if x in ('ki', 'kiszib3', 'giri3', 'ugula') and i + 1 < len(w):
                    fn[w[i + 1]] += 1
                if x in ('i3-dab5',) and i > 0:
                    fn[w[i - 1]] += 1
    return {w for w, c in fn.items() if c >= min_n and w not in known and not w.startswith('iti')}


# ------------------------------------------------------------------ PLANT
def build_plant(n=1500, seed=0):
    """Known roles: COM c0..c11 (fix number system + magnitude), UNT u0..u3 (scale x8),
    TOTM t0, t1 (on total line only), DOC d0..d3 (header; choose commodity subset and
    system), NAME n0..n59 (1-3 per entry, header officials, per-tablet pool), FILL f0..f5
    (random everywhere). Total = arithmetic (last line = sum)."""
    r = random.Random(seed)
    sysof = {c: ['A', 'B', 'C'][c % 3] for c in range(12)}
    mu = {c: r.uniform(1, 7) for c in range(12)}
    doc_com = {d: r.sample(range(12), 5) for d in range(4)}
    out = []
    for i in range(n):
        d = r.randrange(4)
        pool = r.sample(range(60), 8)
        L = [dict(kind='ctx', toks=['d%d' % d] + (['h%d_%d' % (d, r.randrange(3))] if r.random() < .7 else []) +
                  ['n%d' % r.choice(pool)] + (['f%d' % r.randrange(6)] if r.random() < .3 else []),
                  sys=None, mag=None, tot=0)]
        k = r.randint(2, 8)
        vals = []
        for j in range(k):
            c = r.choice(doc_com[d])
            u = r.randrange(4) if r.random() < .4 else None
            v = max(1, int(2 ** r.gauss(mu[c], 0.5)))
            if u is not None:
                v *= 8
            vals.append(v)
            toks = ['n%d' % r.choice(pool) for _ in range(r.randint(1, 3))] + ['c%d' % c] + (['u%d' % u] if u is not None else [])
            if r.random() < .2:
                toks.insert(r.randrange(len(toks) + 1), 'f%d' % r.randrange(6))
            L.append(dict(kind='num', toks=toks, sys=sysof[c], mag=magb(v), tot=0))
        if r.random() < .6:
            L.append(dict(kind='num', toks=['t%d' % r.randrange(2)] + (['c%d' % r.choice(doc_com[d])] if r.random() < .5 else []) +
                          (['n%d' % r.choice(pool)] if r.random() < .3 else []),
                          sys='T', mag=magb(sum(vals)), tot=1))
        out.append(dict(id='PL%d' % i, lines=L))
    return out


PC_KEY = {'commodity': ['P_' + x for x in ['SZE', 'GAR', 'KASZ', 'KU6', 'UDU', 'U8', 'SZAH2', 'GU4', 'AB2',
                                            'DUG', 'TUG2', 'SILA4', 'MASZ', 'KISZ', 'NINDA2', 'ZIZ2', 'SZE3', 'ZATU']],
          'title': ['P_' + x for x in ['EN', 'SANGA', 'NAM2', 'GAL', 'SUKKAL', 'UMBISAG', 'NUN']]}

PLANT_KEY = {'commodity': ['c%d' % i for i in range(12)], 'unit': ['u%d' % i for i in range(4)],
             'total': ['t0', 't1'], 'doctype': ['d%d' % i for i in range(4)],
             'doc_companion': ['h%d_%d' % (i, j) for i in range(4) for j in range(3)],
             'name': ['n%d' % i for i in range(60)], 'filler': ['f%d' % i for i in range(6)]}


# ------------------------------------------------------------------ nulls
def null_within_entry(C, seed):
    r = random.Random(seed)
    out = []
    for t in C:
        L = []
        for l in t['lines']:
            l2 = dict(l)
            tk = list(l['toks'])
            r.shuffle(tk)
            l2['toks'] = tk
            L.append(l2)
        out.append(dict(id=t['id'], lines=L))
    return out


def null_across_lines(C, seed):
    """Signs re-dealt across the lines of the same tablet (line lengths kept; numerals stay)."""
    r = random.Random(seed)
    out = []
    for t in C:
        pool = [x for l in t['lines'] for x in l['toks']]
        r.shuffle(pool)
        L, k = [], 0
        for l in t['lines']:
            l2 = dict(l)
            l2['toks'] = pool[k:k + len(l['toks'])]
            k += len(l['toks'])
            L.append(l2)
        out.append(dict(id=t['id'], lines=L))
    return out


# ------------------------------------------------------------------ tensors
SPECIAL = ['<pad>', '<unk>', '<mask>', '<del>', '<num>', '<nummask>', '<sep>']


class Vocab:
    def __init__(self, C, min_n=3):
        c = collections.Counter(x for t in C for l in t['lines'] for x in l['toks'])
        self.itos = SPECIAL + sorted(w for w, k in c.items() if k >= min_n)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.count = c
        sy = sorted({l['sys'] for t in C for l in t['lines'] if l['sys']})
        self.sys = {s: i for i, s in enumerate(sy)}

    def id(self, w):
        return self.stoi.get(w, 1)


MAXLEN = 160


def encode(t, V):
    """Flatten a tablet. Returns dict of int lists (one per position) + per-line info."""
    tok, kind, li, pos, sysv, magv, totv, isnum = [], [], [], [], [], [], [], []
    for i, l in enumerate(t['lines']):
        for j, w in enumerate(l['toks']):
            tok.append(V.id(w)); kind.append(0 if l['kind'] == 'ctx' else 1); li.append(min(i, 47)); pos.append(min(j, 11))
            sysv.append(-1); magv.append(-1); totv.append(-1); isnum.append(0)
        if l['kind'] == 'num':
            tok.append(4); kind.append(2); li.append(min(i, 47)); pos.append(11)
            sysv.append(V.sys.get(l['sys'], -1) if l['sys'] else -1)
            magv.append(l['mag'] if l['mag'] is not None else -1)
            totv.append(l['tot'] if l['tot'] is not None else -1)
            isnum.append(1)
    n = min(len(tok), MAXLEN)
    return dict(tok=tok[:n], kind=kind[:n], li=li[:n], pos=pos[:n], sys=sysv[:n], mag=magv[:n],
                tot=totv[:n], isnum=isnum[:n])


# ------------------------------------------------------------------ model
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self, nv, nsys, arch):
        super().__init__()
        d = arch['d']
        self.arch = arch
        self.emb = nn.Embedding(nv, d)
        self.kind = nn.Embedding(3, d)
        self.li = nn.Embedding(48, d)
        self.pos = nn.Embedding(12, d)
        self.sysE = nn.Embedding(nsys + 1, d)
        self.magE = nn.Embedding(NB + 1, d)
        if arch['type'] == 'tf':
            layer = nn.TransformerEncoderLayer(d, arch['heads'], 2 * d, arch['drop'], batch_first=True)
            self.enc = nn.TransformerEncoder(layer, arch['layers'])
        else:
            self.enc = nn.GRU(d, d // 2, arch['layers'], batch_first=True, bidirectional=True, dropout=arch['drop'] if arch['layers'] > 1 else 0)
        self.out_tok = nn.Linear(d, nv)
        self.out_sys = nn.Linear(d, nsys)
        self.out_mag = nn.Linear(d, NB)
        self.out_tot = nn.Linear(d, 2)
        self.nsys = nsys

    def forward(self, b):
        x = self.emb(b['tok']) + self.kind(b['kind']) + self.li(b['li']) + self.pos(b['pos'])
        # visible numeral content
        sv = b['sys_in'].clamp(min=-1) + 1  # 0 = hidden
        mv = b['mag_in'].clamp(min=-1) + 1
        x = x + (self.sysE(sv) + self.magE(mv)) * b['isnum'].unsqueeze(-1)
        pad = b['tok'] == 0
        if self.arch['type'] == 'tf':
            h = self.enc(x, src_key_padding_mask=pad)
        else:
            h, _ = self.enc(x)
        return h


def collate(items):
    n = max(len(it['tok']) for it in items)
    out = {}
    for k in ('tok', 'kind', 'li', 'pos', 'sys', 'mag', 'tot', 'isnum', 'sys_in', 'mag_in', 'tgt'):
        fill = 0 if k in ('tok', 'kind', 'li', 'pos', 'isnum') else -1
        out[k] = torch.tensor([it[k] + [fill] * (n - len(it[k])) for it in items], dtype=torch.long)
    return out


def make_view(e, mode, rng=None, pattern=0, delmask=None, pdel=0.0):
    """mode 'num': hide all numerals; targets sys/mag/tot at numeral slots.
    mode 'head': hide a subset of ctx tokens; targets their ids.
    mode 'ent' : hide all tokens of a subset of numeral lines; targets their ids.
    delmask: set of positions turned into <del> (never a target)."""
    tok = list(e['tok']); n = len(tok)
    sys_in = list(e['sys']); mag_in = list(e['mag'])
    tgt = [-1] * n
    tsys = [-1] * n; tmag = [-1] * n; ttot = [-1] * n
    if mode == 'num':
        for i in range(n):
            if e['isnum'][i]:
                sys_in[i] = -1; mag_in[i] = -1
                tsys[i] = e['sys'][i]; tmag[i] = e['mag'][i]; ttot[i] = e['tot'][i]
    elif mode == 'head':
        ctxp = [i for i in range(n) if e['kind'][i] == 0]
        for k, i in enumerate(ctxp):
            hide = (rng.random() < 0.5) if rng else (k % 2 == pattern)
            if hide:
                tgt[i] = tok[i]; tok[i] = 2
    elif mode == 'ent':
        lines = sorted({e['li'][i] for i in range(n) if e['isnum'][i]})
        for k, L in enumerate(lines):
            hide = (rng.random() < 0.5) if rng else (k % 2 == pattern)
            if hide:
                for i in range(n):
                    if e['li'][i] == L and e['kind'][i] == 1:
                        tgt[i] = tok[i]; tok[i] = 2
    if pdel and rng:
        for i in range(n):
            if tgt[i] < 0 and tok[i] > 6 and rng.random() < pdel:
                tok[i] = 3
    if delmask:
        for i in delmask:
            if tgt[i] < 0 and tok[i] != 4:
                tok[i] = 3
    # unk targets ignored
    tgt = [x if x > 6 else -1 for x in tgt]
    return dict(tok=tok, kind=e['kind'], li=e['li'], pos=e['pos'], isnum=e['isnum'], sys_in=sys_in,
                mag_in=mag_in, sys=tsys, mag=tmag, tot=ttot, tgt=tgt)


def losses(model, b):
    """Per-target losses: returns dict task -> (row, col, loss) tensors."""
    h = model(b)
    res = {}
    m = b['sys'] >= 0
    if m.any():
        res['SYS'] = (m.nonzero(), F.cross_entropy(model.out_sys(h[m]), b['sys'][m], reduction='none'))
    m = b['mag'] >= 0
    if m.any():
        res['MAG'] = (m.nonzero(), F.cross_entropy(model.out_mag(h[m]), b['mag'][m], reduction='none'))
    m = b['tot'] >= 0
    if m.any():
        res['TOT'] = (m.nonzero(), F.cross_entropy(model.out_tot(h[m]), b['tot'][m], reduction='none'))
    m = b['tgt'] >= 0
    if m.any():
        res['TOK'] = (m.nonzero(), F.cross_entropy(model.out_tok(h[m]), b['tgt'][m], reduction='none'))
    return res


def random_arch(r):
    t = r.choice(['tf', 'tf', 'gru'])
    return dict(type=t, d=r.choice([32, 48, 64]), layers=r.choice([1, 2, 3]) if t == 'tf' else r.choice([1, 2]),
                heads=r.choice([2, 4]), drop=r.choice([0.0, 0.1, 0.2]), lr=r.choice([1e-3, 2e-3, 3e-3]),
                epochs=r.choice([20, 25, 30]), pdel=r.choice([0.05, 0.1, 0.15]))


def train(E_tr, V, arch, seed, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Net(len(V.itos), max(1, len(V.sys)), arch)
    opt = torch.optim.Adam(model.parameters(), lr=arch['lr'])
    bs = 32
    for ep in range(arch['epochs']):
        model.train()
        idx = list(range(len(E_tr)))
        rng.shuffle(idx)
        tot = 0.0
        for s in range(0, len(idx), bs):
            items = []
            for i in idx[s:s + bs]:
                mode = rng.choice(['num', 'head', 'ent', 'ent'])
                items.append(make_view(E_tr[i], mode, rng=rng, pdel=arch['pdel']))
            b = collate(items)
            R = losses(model, b)
            L = sum(v[1].mean() * (0.5 if k == 'TOK' else 1.0) for k, v in R.items())
            opt.zero_grad(); L.backward(); opt.step()
            tot += float(L)
        if log and ep % 10 == 0:
            log('  ep %d loss %.3f' % (ep, tot / max(1, len(idx) // bs)))
    model.eval()
    return model


VIEWS = [('num', 0), ('head', 0), ('head', 1), ('ent', 0), ('ent', 1)]
TASKS = ['SYS', 'MAG', 'TOT', 'HEAD', 'ENT']


@torch.no_grad()
def eval_views(model, E, jobs, bs=256):
    """jobs: list of (tablet_index, view_index, delmask_positions or None).
    Returns per job a dict {(task, position): loss}."""
    out = [None] * len(jobs)
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        items = [make_view(E[ti], VIEWS[vi][0], pattern=VIEWS[vi][1], delmask=dm) for ti, vi, dm in chunk]
        b = collate(items)
        R = losses(model, b)
        acc = [dict() for _ in chunk]
        for k, (ix, lv) in R.items():
            for r_, c_, v in zip(ix[:, 0].tolist(), ix[:, 1].tolist(), lv.tolist()):
                kk = k if k != 'TOK' else ('HEAD' if VIEWS[chunk[r_][1]][0] == 'head' else 'ENT')
                acc[r_][(kk, c_)] = v
        out[s:s + len(chunk)] = acc
    return out
