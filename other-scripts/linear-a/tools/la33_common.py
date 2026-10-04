#!/usr/bin/env python3
"""LA-33 shared code: QUEUES AT THE STOREROOM.

Idea (queueing / point processes, not linguistics): a storeroom that hands out scheduled
rations to a fixed staff writes regular lists (same recipients, similar or identical amounts,
little variance in list length). A storeroom that serves demand (requests, deliveries,
offerings) writes Poisson-like lists (overdispersed list lengths, heavy-tailed amounts,
recipients arriving at random from a large pool). We measure those queue statistics on
groups of documents, train a classifier on thousands of simulated archives (scheduled,
Poisson, mixed), and check it on documents of known kind (Ur III rations vs deliveries,
Linear B personnel/census vs disbursement series), planted archives and shuffles.

Data
  la_docs()  Linear A tablets: list of docs, each a list of entries (recipient word, amount,
             commodity). Recipient = the entry word (KU-RO / KI-RO / PO-TO-KU-RO removed).
  lb_docs()  Linear B (DAMOS) lines with a word and a number; kind fixed a priori by series.
  ur_docs()  Ur III (CDLI ATF, scratchpad only, never committed). Kind fixed a priori by a
             keyword in the text: rations (sze-ba, siki-ba, i3-ba) and regular offerings
             (sa2-du11) = SCHED; deliveries (mu-kux(DU)), expenditures (ba-zi), receipts
             (szu ba-ti) = DEMAND. Texts with both kinds are dropped.
"""
import json, math, os, re, random, sys, csv
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la33_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
SCR = os.environ.get('LA33_SCR', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad')
TOTALW = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}

# ------------------------------------------------------------------ Linear A
def la_docs():
    from la6_common import la_entries
    C = {x['id']: x for x in json.load(open(os.path.join(D, 'corpus.json')))}
    E = la_entries()
    docs = defaultdict(list)
    for e in E:
        ins = C[e['doc']]
        if ins['support'] != 'Tablet': continue
        lab = e['label']
        if lab is None or lab in TOTALW or e['role'] != 'entry': continue
        if lab not in ins['words']: continue  # logogram labels are not recipients
        amt = float(sum(e['com'].values())) + float(sum(e['bare']))
        if amt <= 0: continue
        com = max(e['com'], key=lambda k: e['com'][k]) if e['com'] else 'BARE'
        docs[e['doc']].append((lab, amt, com))
    out = []
    for d, ents in docs.items():
        ins = C[d]
        coms = Counter(c for _, _, c in ents)
        out.append({'id': d, 'site': d[:2] if not d[:2].isdigit() else d, 'scribe': ins.get('scribe') or '',
                    'kind': '?', 'ents': ents, 'com': coms.most_common(1)[0][0]})
    return out

# ------------------------------------------------------------------ Linear B
LB_SCHED = {'PY Ab', 'PY Aa', 'PY Ad', 'KN Ak', 'KN Ai', 'KN Da', 'KN Db', 'KN Dv', 'KN Dk', 'KN Dl',
            'PY Ma', 'PY Eb', 'PY Ea', 'KN Fp'}
LB_DEMAND = {'PY Fr', 'KN Fh', 'PY Un', 'PY Fn', 'KN F', 'TH Fq', 'TH Gp', 'PY Vn', 'KN Ga'}
LBW = re.compile(r"^[a-z*0-9₂₃]+(-[a-z*0-9₂₃]+)+$")
LBN = re.compile(r"^\d+$")


def lb_docs():
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?)', h)
        if not m: continue
        ser = m.group(1) + ' ' + m.group(2)
        kind = 'S' if ser in LB_SCHED else ('D' if ser in LB_DEMAND else '?')
        ents = []
        for ln in (d.get('content') or '').split('\n'):
            toks = [t.strip('[]⟦⟧,/') for t in ln.split()]
            toks = [re.sub(r'[̣]', '', t) for t in toks]
            w = next((t for t in toks if LBW.match(t)), None)
            nums = [int(t) for t in toks if LBN.match(t)]
            if w and nums:
                ents.append((w, float(sum(nums)), 'X'))
        if ents:
            out.append({'id': h, 'site': m.group(1), 'series': ser, 'kind': kind, 'ents': ents})
    return out

# ------------------------------------------------------------------ Ur III
SKEY = ['sze-ba', 'siki-ba', 'i3-ba', 'sa2-du11']
DKEY = ['mu-kux(DU)', 'ba-zi', 'szu ba-ti']
_num = re.compile(r"^(\d+)\(([a-z0-9']+)(@c)?\)$")
_W = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, 'asz': 1, 'barig': 60, 'ban2': 10}


def _clean(t):
    return re.sub(r'[#!?*\[\]<>]', '', t)


def _parse_line(body):
    toks = [_clean(t) for t in body.split()]
    v = 0; pend = 0; j = 0; seen = False
    for j, t in enumerate(toks):
        m = _num.match(t)
        if m:
            n, u, c = int(m.group(1)), m.group(2), m.group(3)
            if u == 'asz' and c: v += n * 300
            elif u in ('barig', 'ban2'): v += n * _W[u]
            elif u in _W: pend += n * _W[u]
            else: return None, toks
            seen = True; continue
        if t == 'gur': v += pend * 300; pend = 0; continue
        if t == 'sila3': v += pend; pend = 0; continue
        break
    else:
        j = len(toks)
    v += pend
    if not seen: return None, toks
    return float(v), toks[j:]


def ur_docs(cache=True):
    fn = os.path.join(CK, 'ur_docs.json')
    if cache and os.path.exists(fn):
        return json.load(open(fn))
    per = {}
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')):
        if r['period'].startswith('Ur III'):
            per[r['id_text']] = r['provenience'].split(' (')[0]
    texts = {}; pid = None; buf = []
    for raw in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            if pid and buf: texts[pid] = buf
            pid = raw.split()[0][1:]; buf = []; continue
        if pid: buf.append(raw.rstrip('\n'))
    if pid and buf: texts[pid] = buf
    raw_docs = []
    tokfreq = Counter()
    for pid, lines in texts.items():
        key = str(int(pid[1:])) if pid[1:].isdigit() else pid
        prov = per.get(key) or per.get(pid[1:])
        if prov is None: continue
        body = '\n'.join(lines)
        s = any(k in body for k in SKEY); dd = any(k in body for k in DKEY)
        if s == dd: continue
        kind = 'S' if s else 'D'
        rows = []
        for l in lines:
            m = re.match(r"^\d+'?\.\s*(.*)$", l.strip())
            if not m: continue
            b = m.group(1)
            if '...' in b or 'szu-nigin' in b: rows.append((None, [])); continue
            v, rest = _parse_line(b)
            rows.append((v, rest))
            for t in rest: tokfreq[t] += 1
        raw_docs.append((pid, prov, kind, rows))
    stop = set(t for t, _ in tokfreq.most_common(400))
    out = []
    for pid, prov, kind, rows in raw_docs:
        ents = []
        for i, (v, rest) in enumerate(rows):
            if v is None or v <= 0: continue
            cand = [t for t in rest if t not in stop and not re.search(r'\d|^x$|\.\.\.', t)]
            rec = cand[0] if cand else None
            k = i + 1
            while rec is None and k < len(rows) and rows[k][0] is None and k <= i + 2:
                cand = [t for t in rows[k][1] if t not in stop and not re.search(r'\d|^x$', t)]
                rec = cand[0] if cand else None; k += 1
            if rec is None: continue
            ents.append((rec, v, 'X'))
        if ents:
            out.append({'id': pid, 'site': prov, 'kind': kind, 'ents': ents})
    json.dump(out, open(fn, 'w'))
    return out

# ------------------------------------------------------------------ summary statistics
FEAT = ['logT', 'logn', 'disp_n', 'recur', 'jacc', 'mode', 'cv', 'consist', 'sdlog', 'tail', 'single', 'ones']


def feats(docs, rng=None, maxpairs=1500):
    """docs: list of entry lists [(rec, amt, com)]. Returns feature vector (FEAT order)."""
    rng = rng or np.random.default_rng(0)
    docs = [d for d in docs if len(d)]
    T = len(docs)
    n = np.array([len(d) for d in docs], float)
    occ = defaultdict(set)
    for i, d in enumerate(docs):
        for r, _, _ in d: occ[r].add(i)
    recur = np.mean([len(occ[r]) >= 2 for d in docs for r, _, _ in d])
    sets = [set(r for r, _, _ in d) for d in docs]
    if T >= 2:
        I = rng.integers(0, T, size=(maxpairs, 2)); I = I[I[:, 0] != I[:, 1]]
        jacc = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in I])
    else:
        jacc = 0.0
    modes, cvs = [], []
    for d in docs:
        if len(d) < 2: continue
        a = np.array([x for _, x, _ in d])
        modes.append(Counter(a).most_common(1)[0][1] / len(a))
        cvs.append(a.std() / a.mean())
    last = {}; same = []
    for i, d in enumerate(docs):
        for r, x, _ in d:
            if r in last and last[r][0] != i: same.append(abs(last[r][1] - x) < 1e-9)
            last[r] = (i, x)
    A = np.array([x for d in docs for _, x, _ in d])
    la = np.log(A)
    tc = Counter(r for d in docs for r, _, _ in d)
    return np.array([
        math.log(T), math.log(n.mean()), n.var() / n.mean(), recur, jacc,
        np.median(modes) if modes else 1.0, np.median(cvs) if cvs else 0.0,
        np.mean(same) if same else 0.5, la.std(),
        np.percentile(A, 90) / max(np.percentile(A, 50), 1e-9) if len(A) > 2 else 1.0,
        np.mean([c == 1 for c in tc.values()]), np.mean(A == 1)])

# ------------------------------------------------------------------ simulators
def _round_amt(x):
    return float(max(1, int(round(x))))


def sim_tablet_sched(st, rng):
    """st: dict of a scheduled storeroom state; returns one list."""
    S = st['staff']
    m = st['m']
    n = max(1, int(round(rng.normal(m, st['sdn'])))) if st['sdn'] > 0 else int(m)
    n = min(n, len(S))
    # roster: a block of the staff (a work-team) or a random subset
    if rng.random() < st['block']:
        s0 = rng.integers(0, len(S)); idx = [(s0 + k) % len(S) for k in range(n)]
    else:
        idx = rng.choice(len(S), size=n, replace=False)
    out = []
    for k in idx:
        amt = st['cls'][k]
        if rng.random() < st['jit']: amt = _round_amt(amt * math.exp(rng.normal(0, 0.4)))
        out.append((S[k], float(amt), 'X'))
    # turnover
    for k in np.nonzero(rng.random(len(S)) < st['turn'])[0]:
        st['newid'] += 1; S[k] = 's%d' % st['newid']
    return out


def new_sched(rng):
    size = int(math.exp(rng.uniform(math.log(4), math.log(400))))
    K = rng.integers(1, 5)
    base = math.exp(rng.uniform(0, math.log(300)))
    levels = sorted(set(_round_amt(base * f) for f in rng.choice([0.25, 0.5, 1, 1.5, 2, 3], size=K)))
    p = rng.dirichlet(np.ones(len(levels)) * 2)
    st = {'staff': ['s%d' % i for i in range(size)], 'newid': size,
          'cls': [levels[i] for i in rng.choice(len(levels), size=size, p=p)],
          'm': float(min(size, math.exp(rng.uniform(0, math.log(40))))), 'sdn': rng.uniform(0, 2),
          'block': rng.uniform(0, 1), 'jit': rng.uniform(0, 0.3), 'turn': math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))}
    return st


def sim_tablet_pois(st, rng):
    m = st['m']
    lam = rng.gamma(st['k'], m / st['k'])  # gamma-mixed Poisson = overdispersed demand
    n = max(1, rng.poisson(lam))
    out = []
    for _ in range(n):
        # Chinese-restaurant / Pitman-Yor arrivals of partners
        N = st['N']
        if N == 0 or rng.random() < (st['theta'] + st['d'] * len(st['tab'])) / (N + st['theta']):
            st['tab'].append(1); j = len(st['tab']) - 1
        else:
            while True:  # pick a previous customer's table, accept with (c - d) / c
                j = st['cust'][rng.integers(0, N)]
                c = st['tab'][j]
                if rng.random() < (c - st['d']) / c: break
            st['tab'][j] += 1
        st['cust'].append(j)
        st['N'] += 1
        amt = _round_amt(math.exp(rng.normal(st['mu'], st['sig'])))
        out.append(('p%d' % j, amt, 'X'))
    return out


def new_pois(rng):
    return {'m': math.exp(rng.uniform(0, math.log(40))), 'k': math.exp(rng.uniform(math.log(0.3), math.log(50))),
            'theta': math.exp(rng.uniform(math.log(1), math.log(2000))), 'd': rng.uniform(0, 0.7),
            'tab': [], 'cust': [], 'N': 0, 'mu': rng.uniform(0, math.log(200)), 'sig': rng.uniform(0.3, 2.0)}


def sim_archive(T, regime, rng, w=None, survive=None):
    """regime 'S', 'P' or 'M' (mixed: each tablet scheduled with prob w).
    survive: fraction of the written archive that survives (random loss)."""
    survive = survive if survive is not None else math.exp(rng.uniform(math.log(0.1), 0))
    Tw = int(math.ceil(T / survive))
    ss, sp = new_sched(rng), new_pois(rng)
    if regime == 'S': w = 1.0
    elif regime == 'P': w = 0.0
    elif w is None: w = rng.uniform(0.15, 0.85)
    tabs, lab = [], []
    for _ in range(Tw):
        if rng.random() < w: tabs.append(sim_tablet_sched(ss, rng)); lab.append(1)
        else: tabs.append(sim_tablet_pois(sp, rng)); lab.append(0)
    keep = sorted(rng.choice(Tw, size=T, replace=False))
    return [tabs[i] for i in keep], [lab[i] for i in keep], w


def shuffle_entries(docs, rng):
    """Shuffle (recipient, amount) entries across tablets, keeping list lengths."""
    allE = [e for d in docs for e in d]
    rng.shuffle(allE)
    out = []; k = 0
    for d in docs:
        out.append(allE[k:k + len(d)]); k += len(d)
    return out
