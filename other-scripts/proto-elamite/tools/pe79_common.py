#!/usr/bin/env python3
"""PE-79 GRAFT THE WORLD'S LEDGERS ONTO PROTO-ELAMITE: shared code.

Pivot (the requested 'universal role-tagger, leave-one-civilisation-out incl. khipu' was already run as
la57 / pe61).  Inversion: Proto-Elamite is not tagged; it is the JUDGE.  A structural PE grammar
(sign | slot context [, previous sign]) is learned from PE alone.  Records of known administrations
(proto-cuneiform, Ur III, Old Babylonian, Ebla, Old Assyrian, Linear B, Inca khipus) are reduced to a
role skeleton (role symbol per token, numerals moved after the signs = PE entry order, line breaks) and
GRAFTED onto PE: a hypothesis H assigns one PE sign to each role (COM UNI PER PLA TRA HDR TOT); tokens of
unknown role are wildcards.  Score of H on a corpus = mean PMI log P_PE(H[r] | context) - log P_PE(H[r])
over role tokens: how 'Proto-Elamite' the transplanted ledger reads.  Massive random search over H and
over PE-grammar architectures; leave-one-civilisation-out transfer; nulls with role labels permuted
inside each training corpus; a planted corpus (PE held-out tablets with planted sign->role map).
No sign values, shapes or readings are used.
"""
import os, sys, json, random, hashlib, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PED = os.path.join(HERE, '..', 'data')
CK = os.path.join(PED, 'pe79_ckpt')
os.makedirs(CK, exist_ok=True)
LAT = os.path.join(HERE, '..', '..', 'linear-a', 'tools')
sys.path.insert(0, HERE)
sys.path.insert(1, LAT)
import la57_common as L          # noqa: E402
L.CK = CK                         # la57 caches go to the pe79 checkpoint dir

ROLES = ['COM', 'UNI', 'PER', 'PLA', 'TRA', 'HDR', 'TOT']
RI = {r: i for i, r in enumerate(ROLES)}
WILD, START = len(ROLES), len(ROLES) + 1      # prev-role codes: wildcard (unknown role), line start
KNOWN = ['PC', 'UR3', 'OB', 'EB', 'OA', 'LB', 'KH']


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip() + '\n')


# ------------------------------------------------------------------ PE in value-free form (damage-aware load)
def pe_docs():
    fn = os.path.join(CK, 'pe_docs.json')
    if os.path.exists(fn):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(fn))]
    from common import load, is_sign
    import pe56_common as P
    out = []
    for t in P._from_pe_lines(load(), is_sign):
        toks = []
        for l in t['lines']:
            line = [('T', s) for s in l['s']]
            if l['sys'] is not None:
                v = l['v']
                meas = l['sys'] not in ('SDB',) or (v is not None and abs(v - round(v)) > 1e-9)
                line.append(('N', float(v) if v is not None else None, bool(meas)))
            if line:
                toks.extend(line); toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if any(x[0] == 'T' for x in toks):
            out.append({'id': t['id'], 'sys': 'PE', 'site': t['site'], 'toks': toks})
    json.dump(out, open(fn, 'w'))
    return out


def lines_of(d):
    ls, cur = [], []
    for x in d['toks']:
        if x[0] == 'L':
            if cur: ls.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur: ls.append(cur)
    return ls


def pe_order(ls):
    """numerals moved after the signs of each line (PE entry order)."""
    return [[x for x in l if x[0] == 'T'] + [x for x in l if x[0] == 'N'] for l in ls]


def ncls(l):
    ns = [x for x in l if x[0] == 'N']
    if not ns:
        return 'H'
    x = ns[0]
    if x[2]:
        return 'm'
    if x[1] is None:
        return 'u'
    return 's' if x[1] < 10 else 'b'


def contexts(d, arch):
    """yield (ctx tuple, token, prev token or None) for each T token, PE order."""
    ls = pe_order(lines_of(d))
    nl = len(ls)
    for li, l in enumerate(ls):
        ts = [x for x in l if x[0] == 'T']
        nc = ncls(l)
        if arch['ncls'] == 0:
            nc = 'H' if nc == 'H' else 'E'
        lp = 'F' if li == 0 else ('L' if li == nl - 1 else 'M')
        for k, x in enumerate(ts):
            pos = 'o' if len(ts) == 1 else ('f' if k == 0 else ('l' if k == len(ts) - 1 else 'm'))
            c = (nc, pos if arch['pos'] else '-', lp if arch['lpos'] else '-')
            yield c, x[1], (ts[k - 1][1] if k > 0 else None)


# ------------------------------------------------------------------ the PE grammar (the judge)
class Judge:
    def __init__(self, docs, arch, K=120):
        self.arch = arch
        cnt = collections.Counter(x[1] for d in docs for x in d['toks'] if x[0] == 'T')
        self.signs = [s for s, _ in cnt.most_common(K)]
        self.si = {s: i for i, s in enumerate(self.signs)}
        K = len(self.signs)
        cc = collections.defaultdict(lambda: np.zeros(K))
        cb = collections.defaultdict(lambda: np.zeros(K))
        tot = np.zeros(K)
        for d in docs:
            for c, s, p in contexts(d, arch):
                if s not in self.si:
                    continue
                j = self.si[s]
                tot[j] += 1; cc[c][j] += 1
                pk = '^' if p is None else (p if p in self.si else '?')
                cb[(c, pk)][j] += 1
        self.ctxs = sorted(cc)
        self.ci = {c: i for i, c in enumerate(self.ctxs)}
        p0 = (tot + 0.5) / (tot.sum() + 0.5 * K)
        lam = arch['lam']
        # Witten-Bell-ish interpolation
        pc = np.zeros((len(self.ctxs), K))
        for c, v in cc.items():
            n, t = v.sum(), (v > 0).sum()
            w = n / (n + t * (1 / lam - 1) + 1e-9) if n else 0
            pc[self.ci[c]] = w * v / max(n, 1) + (1 - w) * p0
        self.lp0 = np.log(p0)
        self.pmi_c = np.log(pc) - self.lp0                     # [ctx, sign]
        # bigram table [ctx, prev(K + start), sign]; prev = wildcard -> pmi_c
        self.pmi_b = None
        if arch['prev']:
            pb = np.repeat(pc[:, None, :], K + 1, 1)
            for (c, pk), v in cb.items():
                if pk == '?':
                    continue
                pi = K if pk == '^' else self.si[pk]
                n, t = v.sum(), (v > 0).sum()
                w = n / (n + t * (1 / lam - 1) + 1e-9)
                pb[self.ci[c], pi] = w * v / n + (1 - w) * pc[self.ci[c]]
            self.pmi_b = np.log(pb) - self.lp0[None, None, :]

    def ctx_index(self, c):
        return self.ci.get(c)


# ------------------------------------------------------------------ known administrations as role skeletons
def known_docs():
    fn = os.path.join(CK, 'known_roles.json')
    if os.path.exists(fn):
        J = json.load(open(fn))
        return {k: [dict(d, toks=[tuple(x) for x in d['toks']]) for d in v] for k, v in J.items()}
    out = {}
    for k in KNOWN:
        ld, tr = L.LOADERS[k]
        docs = ld()
        if k == 'KH':
            rd = []
            for d in docs:      # each cord = its own entry line (cord colour then its knot value)
                occ = iter(d['occ'])
                toks = []
                for x in d['toks']:
                    if x[0] == 'T':
                        if toks and toks[-1][0] != 'L':
                            toks.append(('L',))
                        toks.append(('T', 'TOT' if next(occ) == 'TOT' else 'COM'))
                    elif x[0] == 'N':
                        toks.append(x)
                rd.append({'id': d['id'], 'toks': toks, 'types': [x[1] for x in d['toks'] if x[0] == 'T']})
        else:
            truth = tr(docs)
            rd = []
            for d in docs:
                toks, types = [], []
                for x in d['toks']:
                    if x[0] == 'T':
                        r = truth.get(x[1])
                        toks.append(('T', r if r in RI else 'O')); types.append(x[1])
                    else:
                        toks.append(x)
                rd.append({'id': d['id'], 'toks': toks, 'types': types})
        out[k] = rd
    json.dump(out, open(fn, 'w'))
    return out


def permute_roles(docs, rng, khipu=False):
    """kill control: role labels permuted across TYPES inside the corpus (token level for khipu)."""
    if khipu:
        labs = [x[1] for d in docs for x in d['toks'] if x[0] == 'T']
        rng.shuffle(labs); it = iter(labs)
        return [dict(d, toks=[('T', next(it)) if x[0] == 'T' else x for x in d['toks']]) for d in docs]
    tr = {}
    for d in docs:
        ti = iter(d['types'])
        for x in d['toks']:
            if x[0] == 'T':
                tr[next(ti)] = x[1]
    ts = sorted(tr); vals = [tr[t] for t in ts]; rng.shuffle(vals)
    m = dict(zip(ts, vals))
    out = []
    for d in docs:
        ti = iter(d['types'])
        out.append(dict(d, toks=[('T', m[next(ti)]) if x[0] == 'T' else x for x in d['toks']]))
    return out


def skeleton(docs, judge):
    """counts over (ctx index, role, prev code) for role tokens; ctx unseen in PE are dropped."""
    c = collections.Counter()
    for d in docs:
        for ctx, r, p in contexts(d, judge.arch):
            if r not in RI:
                continue
            ci = judge.ctx_index(ctx)
            if ci is None:
                continue
            pc = START if p is None else (RI[p] if p in RI else WILD)
            c[(ci, RI[r], pc)] += 1
    a = np.array([[k[0], k[1], k[2], v] for k, v in c.items()], dtype=np.int64)
    return a


def score_H(H, sk, judge):
    """H: [nH, 7] sign indices.  Role-balanced score: mean over roles present of the mean PMI of that role's
    tokens (so a role with few tokens, e.g. khipu top cords, weighs as much as a common one)."""
    if len(sk) == 0:
        return np.full(len(H), np.nan)
    K = len(judge.signs)
    tot = np.zeros((len(ROLES), len(H))); n = np.zeros(len(ROLES))
    for ci, r, pc, w in sk:
        s = H[:, r]
        if judge.pmi_b is None or pc == WILD:
            v = judge.pmi_c[ci, s]
        elif pc == START:
            v = judge.pmi_b[ci, K, s]
        else:
            v = judge.pmi_b[ci, H[:, pc], s]
        tot[r] += w * v; n[r] += w
    pr = n > 0
    return (tot[pr] / n[pr, None]).mean(0)


def random_H(n, K, rng):
    return np.argsort(rng.random((n, K)), 1)[:, :len(ROLES)].astype(np.int64)


def roles_present(sk):
    return sorted({int(r) for r in sk[:, 1]}) if len(sk) else []
