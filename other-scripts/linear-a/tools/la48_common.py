#!/usr/bin/env python3
"""LA-48 shared code: THE PHYSICS OF GOODS.

The archive is treated as a physical system in which goods are conserved. Words are opaque nodes
(no sound values, no meanings). Each document is a set of flows:
  header words (words with no number before the first counted entry) are the account holder of the
  document; entry heads (the word in front of a number) are the counterparties. A document label
  d in {+1 IN, -1 OUT, 0 STOCK} fixes the direction: on an IN document the counterparties give and
  the holder receives; on OUT the reverse; a STOCK document records holdings (storage term).
  Node = (site, word, commodity). For a node, each occurrence contributes  d * slot * q  (slot +1
  for holder, -1 for counterparty); STOCK occurrences enter as storage (-slot * q).
  Kirchhoff: a pass-through node balances exactly (net 0, with both signs present). A source or sink
  node has all its flows on one side.
Quantities are exact vectors: integer part plus the COUNT of each fraction sign (no fraction values
are assumed; every fraction sign is its own exact dimension). Linear B and Ur III quantities are
exact rationals in base units, scaled to integers. Vectors are hashed to one 64-bit integer by
random odd weights (wrap-around arithmetic), so net == 0 is an exact test (collision ~2^-64).
Arithmetic totals (a number equal to the sum of the entries above it) are dropped blindly.
"""
import json, os, re, sys, math, random, hashlib, collections
from fractions import Fraction as Fr
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la48_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
PE = os.path.join(HERE, '..', '..', 'proto-elamite', 'data')


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def base_of(v):
    parts = [p.strip("'[] ") for p in v.split('+')]
    for p in parts:
        q = p.lstrip('*')
        if q in ('GRA', 'OLE', 'OLIV', 'VIN', 'VINb', 'CYP', 'VIR', 'AROM', 'HIDE', 'CAP'):
            return 'VIN' if q == 'VINb' else q
    return parts[0]


# ------------------------------------------------------------------ quantity vectors
def qadd(a, b, k=1):
    out = dict(a)
    for x, v in b.items():
        out[x] = out.get(x, 0) + k * v
        if out[x] == 0:
            del out[x]
    return out


def qzero(a):
    return not a


def drop_totals(entries):
    """entries: list of [head, comm, qvec]. Drop an entry whose qvec equals the sum of >= 2 entries of the
    same commodity since the last dropped total (blind arithmetic rule)."""
    out, run, n = [], {}, collections.Counter()
    for e in entries:
        h, c, q = e
        s = run.get(c, {})
        if n[c] >= 2 and s == q:
            run[c] = {}; n[c] = 0
            continue
        out.append(e)
        run[c] = qadd(s, q); n[c] += 1
    return out


# ------------------------------------------------------------------ Linear A
def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        header, entries = [], []
        head, comm, last_logo = None, '-', None
        seen_entry = False
        pending_word = None
        for t in ins['tokens']:
            if t['t'] == 'word':
                w = '-'.join(t['s'])
                if pending_word is not None and not seen_entry and len(header) < 3:
                    header.append(pending_word)
                pending_word = w
                head = None
            elif t['t'] == 'logo':
                last_logo = base_of(t['v'])
            elif t['t'] == 'num':
                q = {}
                if t['v']:
                    q['#'] = int(t['v'])
                for f in t['frac']:
                    q[f] = q.get(f, 0) + 1
                if not q:
                    continue
                if pending_word is not None:
                    head = pending_word; pending_word = None
                entries.append([head, last_logo or '-', q])
                seen_entry = True
        if pending_word is not None and not seen_entry and len(header) < 3:
            header.append(pending_word)
        entries = drop_totals(entries)
        if entries:
            out.append({'id': ins['id'], 'site': ins['site'], 'group': ins.get('scribe') or '',
                        'support': ins['support'], 'header': header, 'entries': entries})
    return out


# ------------------------------------------------------------------ Linear B (DAMOS)
def lb_docs(sites=('KN', 'PY')):
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return [d for d in json.load(open(fn)) if d['site'] in sites]
    from la41_common import lb_docs as L41
    L = L41(('KN', 'PY'))
    out = []
    for k, d in L.items():
        header, entries = [], []
        head = None; last_logo = None; seen = False
        for it in d['items']:
            w = '-'.join(it['w']) if it['w'] else None
            if it['logo']:
                last_logo = it['logo']
            if it['num'] is None:
                if w and not seen and len(header) < 3:
                    header.append(w)
                continue
            if w:
                head = w
            v = Fr(it['val']).limit_denominator(720)
            if v == 0:
                continue
            entries.append([head, last_logo or '-', {'#': int(v * 720)}])
            seen = True
        entries = drop_totals(entries)
        if entries:
            out.append({'id': k, 'site': d['site'], 'group': d['support'], 'header': header, 'entries': entries})
    json.dump(out, open(fn, 'w'))
    return [d for d in out if d['site'] in sites]


# truth for LB series direction (control scoring only; never an input)
LB_DIR = {'Ma': 'IN', 'Mc': 'IN', 'Na': 'IN', 'Nn': 'IN', 'Ng': 'IN', 'Mn': 'IN', 'Nc': 'IN',
          'Fr': 'OUT', 'Fn': 'OUT', 'Un': 'OUT', 'Fp': 'OUT', 'Fs': 'OUT', 'Es': 'OUT', 'Gg': 'OUT',
          'Da': 'STOCK', 'Db': 'STOCK', 'Dc': 'STOCK', 'Dd': 'STOCK', 'De': 'STOCK', 'Df': 'STOCK',
          'Dg': 'STOCK', 'Dk': 'STOCK', 'Dl': 'STOCK', 'Cn': 'STOCK', 'Sc': 'STOCK', 'Ae': 'STOCK',
          'Ab': 'OUT', 'Ak': 'STOCK', 'Ai': 'STOCK'}


# ------------------------------------------------------------------ Ur III (Puzrish-Dagan, CDLI via pe38)
def ur3_docs(n=None, rs=None):
    fn = os.path.join(CK, 'ur3_pd_docs.json')
    if os.path.exists(fn):
        out = json.load(open(fn))
    else:
        src = json.load(open(os.path.join(PE, 'pe38_ckpt', 'ur3_docs.json')))
        out = []
        for d in src:
            if d['site'] != 'Puzriš-Dagan':
                continue
            header, entries = [], []
            toks_all = []
            for l in d['lines']:
                toks = [t for t in l['toks'] if t]
                toks_all += toks
                if l['val'] is not None and l['sys'] in (1, 2) and toks:
                    try:
                        v = Fr(l['val'])
                    except Exception:
                        continue
                    if v <= 0:
                        continue
                    entries.append([None, toks[0], {'#': int(v * 720)}])
                elif l['val'] is None:
                    header += toks
            entries = drop_totals(entries)
            if not entries:
                continue
            s = set(toks_all)
            cls = 'OUT' if ('ba-zi' in s or 'zi-ga' in s) else ('IN' if 'mu-kux(du)' in s else
                                                              ('TRANSFER' if 'i3-dab5' in s else 'OTHER'))
            out.append({'id': d['id'], 'site': 'PD', 'group': cls, 'header': header, 'entries': entries,
                        'lines': [l['toks'] for l in d['lines'] if l['val'] is None]})
        json.dump(out, open(fn, 'w'))
    if n is not None:
        rs = rs or random.Random(seed('ur3'))
        out = list(out); rs.shuffle(out); out = out[:n]
    return out


def ur3_node_truth(docs):
    """SOURCE: token X-ta after 'ki' on a ba-zi/zi-ga text (expended from X).
    RECEIVER: token before i3-dab5 on its line. DELIVERER: token after mu-kux(du).
    DATE/GRAMMAR: tokens of lines starting 'iti', 'mu', 'u4'."""
    lab = collections.defaultdict(collections.Counter)
    for d in docs:
        for l in d['lines']:
            l = [t for t in l if t]
            if not l:
                continue
            if l[0] in ('iti', 'mu', 'u4'):
                for t in l:
                    lab[t]['DATE'] += 1
            for i, t in enumerate(l):
                if t == 'ki' and i + 1 < len(l) and l[i + 1].endswith('-ta'):
                    lab[l[i + 1]]['SOURCE'] += 1
                if t == 'i3-dab5' and i > 0:
                    lab[l[i - 1]]['RECEIVER'] += 1
                if t == 'mu-kux(du)' and i + 1 < len(l):
                    lab[l[i + 1]]['DELIVERER'] += 1
    return {w: c.most_common(1)[0][0] for w, c in lab.items()}


# ------------------------------------------------------------------ planted flow network
def planted_docs(n_docs, rs, qpool, survival=1.0, n_pass=25, n_leaf=250, n_hub=2, comms=('C1', 'C2', 'C3'),
                 entries_mean=4.0, agg=1):
    """A real conserved economy. Leaves deliver to pass-through officials (IN docs headed by the
    official); each official forwards exactly what it collected to a hub (IN doc headed by the hub listing
    officials with the exact aggregated sums); hubs issue rations to leaves (OUT docs headed by hub);
    hubs record stock (STOCK docs). Then only `survival` of the documents survive, and n_docs are kept."""
    docs = []
    hubs = ['H%d' % i for i in range(n_hub)]
    passes = ['P%d' % i for i in range(n_pass)]
    leaves = ['L%d' % i for i in range(n_leaf)]
    k = 0
    target = int(n_docs / max(survival, 1e-9)) + 5
    truth_doc = {}
    while len(docs) < target:
        c = rs.choice(comms)
        hub = rs.choice(hubs)
        # one collection round: official p collects from m leaves, forwards to hub
        p = rs.choice(passes)
        tot = {}
        for _r in range(rs.randint(1, agg) if agg > 1 else 1):
            m = max(1, int(rs.expovariate(1 / entries_mean)) + 1)
            ent = []
            for _ in range(m):
                q = dict(rs.choice(qpool))
                ent.append([rs.choice(leaves), c, q]); tot = qadd(tot, q)
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'IN', 'header': [p], 'entries': ent}); k += 1
        # forwarding record: hub lists several officials' sums (this one plus others' fresh rounds)
        fw = [[p, c, tot]]
        if rs.random() < 0.5:
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'IN', 'header': [hub], 'entries': fw}); k += 1
        else:
            # forwarded together with one other official's round
            p2 = rs.choice(passes); m2 = max(1, int(rs.expovariate(1 / entries_mean)) + 1)
            ent2 = []; tot2 = {}
            for _ in range(m2):
                q = dict(rs.choice(qpool)); ent2.append([rs.choice(leaves), c, q]); tot2 = qadd(tot2, q)
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'IN', 'header': [p2], 'entries': ent2}); k += 1
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'IN', 'header': [hub],
                         'entries': fw + [[p2, c, tot2]]}); k += 1
        if rs.random() < 0.6:
            m = max(1, int(rs.expovariate(1 / entries_mean)) + 1)
            ent = [[rs.choice(leaves), c, dict(rs.choice(qpool))] for _ in range(m)]
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'OUT', 'header': [hub], 'entries': ent}); k += 1
        if rs.random() < 0.1:
            docs.append({'id': 'pl%d' % k, 'site': 'S', 'group': 'STOCK', 'header': [hub],
                         'entries': [[None, cc, dict(rs.choice(qpool))] for cc in comms]}); k += 1
    rs.shuffle(docs)
    keep = [d for d in docs if rs.random() < survival][:n_docs]
    return keep


ROLE_TRUE = lambda w: 'HUB' if w.startswith('H') else ('PASS' if w.startswith('P') else 'LEAF')


# ------------------------------------------------------------------ nulls
def shuffle_quantities(docs, rs):
    """Permute entry quantity vectors among entries of the same site x commodity (across documents)."""
    pools = collections.defaultdict(list)
    for di, d in enumerate(docs):
        for ei, e in enumerate(d['entries']):
            pools[(d['site'], e[1])].append((di, ei))
    out = [dict(d, entries=[list(e) for e in d['entries']]) for d in docs]
    for key, idx in pools.items():
        qs = [docs[di]['entries'][ei][2] for di, ei in idx]
        rs.shuffle(qs)
        for (di, ei), q in zip(idx, qs):
            out[di]['entries'][ei][2] = q
    return out


def poisson_quantities(docs, rs):
    nr = np.random.default_rng(rs.randrange(1 << 30))
    out = []
    for d in docs:
        ent = []
        for h, c, q in d['entries']:
            q2 = dict(q)
            if '#' in q2:
                scale = 720 if q2['#'] % 720 == 0 and q2['#'] >= 720 else 1
                v = int(nr.poisson(q2['#'] / scale)) * scale
                if v:
                    q2['#'] = v
                else:
                    del q2['#']
                if not q2:
                    q2 = {'#': scale}
            ent.append([h, c, q2])
        out.append(dict(d, entries=ent))
    return out


# ------------------------------------------------------------------ build occurrence arrays
def build(docs, wseed=12345, header_slot=True):
    """Returns dict with numpy arrays for the engine.
    Occurrence = (node, doc, slot, val). Holder occurrence carries the sum of the doc's entries of that
    commodity whose head is not the holder itself."""
    rs = random.Random(wseed)
    W = {}

    def wt(key):
        if key not in W:
            W[key] = (int(hashlib.sha256(('%d|%r' % (wseed, key)).encode()).hexdigest()[:16], 16) >> 2) | 1
        return W[key]

    def hval(c, q):
        h = 0
        for comp, v in q.items():
            h = (h + wt((c, comp)) * v) & ((1 << 64) - 1)
        if h >= 1 << 63:
            h -= 1 << 64
        return h

    nodes = {}
    occ = []
    for di, d in enumerate(docs):
        site = d['site']
        tot = collections.defaultdict(dict)
        for h, c, q in d['entries']:
            tot[c] = qadd(tot[c], q)
            if h is not None:
                key = (site, h, c)
                nodes.setdefault(key, len(nodes))
                occ.append((nodes[key], di, -1, hval(c, q), q))
        if header_slot:
            for hw in d['header']:
                for c, q in tot.items():
                    if q:
                        key = (site, hw, c)
                        nodes.setdefault(key, len(nodes))
                        occ.append((nodes[key], di, +1, hval(c, q), q))
    nn = len(nodes)
    occ.sort(key=lambda x: (x[0], x[1]))
    o_node = np.array([o[0] for o in occ], np.int64)
    o_doc = np.array([o[1] for o in occ], np.int64)
    o_slot = np.array([o[2] for o in occ], np.int64)
    o_val = np.array([o[3] for o in occ], np.int64)
    nptr = np.zeros(nn + 1, np.int64)
    for o in occ:
        nptr[o[0] + 1] += 1
    nptr = np.cumsum(nptr)
    # doc -> nodes touched (unique)
    dn = collections.defaultdict(set)
    for o in occ:
        dn[o[1]].add(o[0])
    nd = len(docs)
    dptr = np.zeros(nd + 1, np.int64); didx = []
    for i in range(nd):
        s = sorted(dn.get(i, ()))
        didx += s; dptr[i + 1] = dptr[i] + len(s)
    names = [None] * nn
    for k, v in nodes.items():
        names[v] = k
    return {'o_node': o_node, 'o_doc': o_doc, 'o_slot': o_slot, 'o_val': o_val, 'nptr': nptr,
            'dptr': dptr, 'didx': np.array(didx, np.int64), 'names': names, 'nd': nd, 'occ_q': [o[4] for o in occ]}


# ------------------------------------------------------------------ engine
@njit(cache=True)
def node_eval(n, d, nptr, o_doc, o_slot, o_val):
    """returns (pure, k2, k3, nflow): pure=1 if >=2 flow occurrences all one sign;
    k2/k3 = 1 if node exactly balances with 2 / >=3 terms and both directions present."""
    net = np.int64(0)
    pos = 0; neg = 0; nst = 0; nflow = 0
    for j in range(nptr[n], nptr[n + 1]):
        dd = d[o_doc[j]]
        if dd == 0:
            net += -o_slot[j] * o_val[j]
            nst += 1
        else:
            s = dd * o_slot[j]
            net += s * o_val[j]
            nflow += 1
            if s > 0:
                pos += 1
            else:
                neg += 1
    terms = nflow + nst
    pure = 1 if (nflow >= 2 and (pos == 0 or neg == 0)) else 0
    k2 = 0; k3 = 0
    if terms >= 2 and net == 0 and ((pos > 0 and neg > 0) or (nst > 0 and nflow > 0)):
        if terms == 2:
            k2 = 1
        else:
            k3 = 1
    return pure, k2, k3, nflow


@njit(cache=True)
def total_score(d, nn, nptr, o_doc, o_slot, o_val, wG, wK2, wK3):
    s = 0.0
    for n in range(nn):
        p, a, b, f = node_eval(n, d, nptr, o_doc, o_slot, o_val)
        s += wG * p + wK2 * a + wK3 * b
    return s


@njit(cache=True)
def anneal(d, nn, nptr, o_doc, o_slot, o_val, dptr, didx, wG, wK2, wK3, nsteps, T0, T1, allow0, seedv):
    np.random.seed(seedv)
    nd = d.shape[0]
    cur = total_score(d, nn, nptr, o_doc, o_slot, o_val, wG, wK2, wK3)
    best = cur; bestd = d.copy()
    for it in range(nsteps):
        T = T0 * (T1 / T0) ** (it / max(1, nsteps - 1))
        t = np.random.randint(nd)
        if dptr[t + 1] == dptr[t]:
            continue
        old = d[t]
        if allow0:
            nv = old
            while nv == old:
                nv = np.random.randint(3) - 1
        else:
            nv = -old
        before = 0.0
        for j in range(dptr[t], dptr[t + 1]):
            p, a, b, f = node_eval(didx[j], d, nptr, o_doc, o_slot, o_val)
            before += wG * p + wK2 * a + wK3 * b
        d[t] = nv
        after = 0.0
        for j in range(dptr[t], dptr[t + 1]):
            p, a, b, f = node_eval(didx[j], d, nptr, o_doc, o_slot, o_val)
            after += wG * p + wK2 * a + wK3 * b
        delta = after - before
        if delta >= 0 or np.random.random() < np.exp(delta / T):
            cur += delta
            if cur > best:
                best = cur; bestd[:] = d
        else:
            d[t] = old
    return best, bestd


@njit(cache=True)
def random_scores(nn, nd, nptr, o_doc, o_slot, o_val, wG, wK2, wK3, nsamp, allow0, seedv):
    np.random.seed(seedv)
    out = np.zeros(nsamp)
    d = np.zeros(nd, np.int64)
    for s in range(nsamp):
        for i in range(nd):
            if allow0:
                d[i] = np.random.randint(3) - 1
            else:
                d[i] = 1 if np.random.random() < 0.5 else -1
        out[s] = total_score(d, nn, nptr, o_doc, o_slot, o_val, wG, wK2, wK3)
    return out


def components(B, d):
    nn = len(B['names'])
    res = []
    for n in range(nn):
        res.append(node_eval(n, d, B['nptr'], B['o_doc'], B['o_slot'], B['o_val']))
    return np.array(res)


def search(B, wG, wK2, wK3, restarts=8, steps=200000, allow0=True, sd=1):
    nn = len(B['names']); nd = B['nd']
    best = -1; bd = None
    for r in range(restarts):
        rng = np.random.default_rng(sd * 1000 + r)
        d = rng.integers(-1, 2, nd) if allow0 else rng.choice([-1, 1], nd)
        d = d.astype(np.int64)
        s, dd = anneal(d, nn, B['nptr'], B['o_doc'], B['o_slot'], B['o_val'], B['dptr'], B['didx'],
                       wG, wK2, wK3, steps, 2.0, 0.02, allow0, sd * 7919 + r)
        if s > best:
            best, bd = s, dd.copy()
    return best, bd


def node_multiplicity(B):
    nptr = B['nptr']
    return np.diff(nptr)


def fmt(x, nd=3):
    return ('%.' + str(nd) + 'g') % x


def balance_kind(B, d, n):
    """For a balanced node: 'PAIR' if its signed terms split into +x / -x pairs of equal amount (repetition),
    else 'AGG' (a genuine aggregation such as a + b = c)."""
    terms = []
    for j in range(B['nptr'][n], B['nptr'][n + 1]):
        dd = d[B['o_doc'][j]]
        s = -B['o_slot'][j] if dd == 0 else dd * B['o_slot'][j]
        terms.append((int(B['o_val'][j]), int(s)))
    c = collections.Counter()
    for v, s in terms:
        c[v] += s
    return 'PAIR' if all(x == 0 for x in c.values()) else 'AGG'


def balance_census(B, d):
    comp = components(B, d)
    out = collections.Counter()
    for n in range(len(B['names'])):
        if comp[n, 1] or comp[n, 2]:
            out[balance_kind(B, d, n)] += 1
    return out


@njit(cache=True)
def residuals_nb(d, nn, nptr, o_doc, o_slot, o_val):
    r = np.zeros(nn, np.int64); k = np.zeros(nn, np.int64)
    for n in range(nn):
        s = np.int64(0)
        for j in range(nptr[n], nptr[n + 1]):
            dd = d[o_doc[j]]
            sg = -o_slot[j] if dd == 0 else dd * o_slot[j]
            s += sg * o_val[j]
        r[n] = s; k[n] = nptr[n + 1] - nptr[n]
    return r, k
