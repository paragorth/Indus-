"""pe68 PREDICT WHAT THE BROKEN TABLETS LOST, THEN CHECK AGAINST THE JOINS: shared library.

Fragment map (cycle 1): for each Proto-Elamite join, which transliterated lines sit on which fragment.
Made by eye from the CDLI line art (Scheil's per-fragment copies in MDP 6 / MDP 17 / MDP 26S, Dahl's
join drawings) and the joined photographs, before any prediction was made. Indices are 0-based positions
in the tablet's line list in data/pe_corpus.json. Confidence H/M/L describes the line-to-fragment map only.
Map errors are blind to the reading: the reading and every null see the same visible / hidden split.
"""
import os, sys, json, math, random, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe68_ckpt')
os.makedirs(CK, exist_ok=True)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ cycle-1 fragment map (frozen before cycle 2)
# 'obv_split' entries are resolved at run time: (fraction of obverse lines from the start, label of the
# first part, label of the rest, rest also gets the reverse?)
FRAGMAP = {
    'P008275': {'frags': {'226': [0, 1, 2, 7], '212': [5, 6, 12, 18, 19, 25, 26, 28, 29]}, 'rest': '77',
                'conf': 'L', 'note': 'Dahl drawing + photo labels: 226 = top-left corner (header, row starts), '
                '212 = bottom band (row ends) and reverse strip; 77 the large middle',
                'joined': 'Dahl 2005'},
    'P008279': {'frags': {'347': [0, 1]}, 'rest': '81', 'conf': 'M',
                'note': 'Scheil copy: 347 is the top-right corner with the header', 'joined': 'Meriggi 1974'},
    'P008294': {'frags': {'325+380': [0, 1, 2, 3, 4, 58, 59, 60, 61, 62, 63, 64, 65]}, 'rest': '96',
                'conf': 'M', 'note': 'Scheil copies: 325 and 380 are top pieces (first herd on the obverse, '
                'last herd on the reverse; ATF note: rev 6.e from MDP 17, 380)', 'joined': 'Dahl 2005'},
    'P008387': {'frags': {'336': [0, 1, 2, 10]}, 'rest': '189', 'conf': 'M',
                'note': 'Scheil copy: 336 carries the header box and the reverse', 'joined': 'Dahl 2012'},
    'P008403': {'frags': {'205': [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]}, 'rest': '213',
                'conf': 'L', 'note': 'Scheil copy of 205 only ("suite de 213"): 205 = obverse M297 run and '
                'reverse column 1 (1(N14) 6(N01), 2(N14) visible)', 'joined': 'Meriggi 1974'},
    'P008448': {'frags': {'250': [0, 1, 2, 3, 4, 5]}, 'rest': '251', 'conf': 'M',
                'note': 'Scheil copies 250 f/r, 251 f/r: 250 = upper obverse with header; 251 = lower obverse '
                'and the reverse total 7(N14) 6(N01)', 'joined': 'Meriggi 1974'},
    'P008471': {'frags': {'273': [6, 8], '291': [7, 9]}, 'rest': '280', 'conf': 'L',
                'note': 'Scheil copies: 280 f/r carries the large N48/N34 lines and the reverse', 'joined': 'pre-Sb'},
    'P008148': {'frags': {'5025': [8, 9]}, 'rest': '366', 'conf': 'L',
                'note': 'small bottom-left corner piece', 'joined': 'Dahl 2012'},
    'P009251': {'frags': {'5241': list(range(0, 13))}, 'rest': '4846', 'conf': 'M',
                'note': 'MDP 26S copies side by side: 5241 left (header, first rows), 4846 right', 'joined': 'Dahl 2012'},
    'P009262': {'frags': {'A': [0, 1, 7], 'B': [2, 3, 4]}, 'rest': 'C', 'conf': 'L',
                'note': 'three unlabelled pieces left to right; reverse strip under the left piece',
                'joined': 'Dahl 2012'},
    'P008495': {'frags': {'297': [0, 1, 2, 3]}, 'rest': '299', 'conf': 'L',
                'note': 'only 297 drawn (about four entries)', 'joined': 'Dahl 2012'},
    'P008043': {'obv_split': [0.6, '246+332', '269+302', 'first'], 'conf': 'L',
                'note': '246+332 joined 1989, 269+302 added 2011; the later pieces modelled as the last 40% '
                'of the obverse', 'joined': 'Damerow-Englund 1989 / Dahl 2011'},
    'P008105': {'obv_split': [0.667, '316+322+324', '335+15247', 'first'], 'conf': 'L',
                'note': 'MDP 6 pieces vs the MDP 26S piece and Sb 15247; later pieces modelled as the last third '
                'of the obverse', 'joined': 'Dahl 2012'},
}
EXCLUDED = {'P008512': 'one transliterated line', 'P008117': 'not a physical join (Sb 15165+ ?, separate fragment)'}
JOINED_IDS = set(FRAGMAP) | set(EXCLUDED)


def resolve_fragments(pid, lines):
    """-> dict fragment label -> sorted list of line indices (all lines of the tablet covered)."""
    spec = FRAGMAP[pid]
    n = len(lines)
    if 'obv_split' in spec:
        f, a, b, rev_to = spec['obv_split']
        obv = [i for i, l in enumerate(lines) if l['surf'] == 'obverse']
        other = [i for i in range(n) if i not in set(obv)]
        k = int(round(f * len(obv)))
        out = {a: obv[:k] + (other if rev_to == 'first' else []), b: obv[k:] + (other if rev_to != 'first' else [])}
        return {k2: sorted(v) for k2, v in out.items()}
    used = set()
    out = {}
    for k, v in spec['frags'].items():
        out[k] = sorted(i for i in v if i < n)
        used |= set(out[k])
    out[spec['rest']] = [i for i in range(n) if i not in used]
    return out


# ------------------------------------------------------------------ unified tablets
def _pe_like(T, capmap, cntmap, tablet_type, ncls, value):
    out = []
    for t in T:
        tt = tablet_type(t)
        L = []
        for i, l in enumerate(t['lines']):
            nums = l['nums']
            cls = None
            v = None
            if nums and nums[0][1] != 'n' and l.get('numclean', True):
                c = ncls(nums)
                cls = 'CAP' if c == 'CAP' or (c == 'MOD' and any(cc.split('@')[0] in ('N39B', 'N30C', 'N24', 'N30D', 'N39C') for _, cc in nums)) else 'CNT'
                m = capmap if (cls == 'CAP' or tt == 'CAPT') else cntmap
                val = value([(n, cc.split('@')[0]) for n, cc in nums], m)
                if val is None and m is capmap:
                    val = value(nums, cntmap)
                v = float(val) if val else None
            fin = l['signs'][-1] if (l['signs'] and l['clean'] and nums) else None
            ev = 0
            if cls == 'CNT':
                c = ncls(nums)
                n01 = sum(n for n, cc in nums if cc == 'N01')
                ev = int(c in ('FRAC', 'BIS') or (c == 'AMB' and n01 >= 6))
            L.append({'i': i, 'surf': l['surf'], 'role': l['role'], 'cls': cls, 'v': v, 'fin': fin, 'ev': ev,
                      'hdr': l['signs'][0] if (l['role'] == 'H' and l['signs']) else None, 'signs': l['signs']})
        out.append({'id': t['id'], 'site': t.get('site', ''), 'lines': L})
    return out


def corpus(name):
    fn = os.path.join(CK, 'tabs_%s.json' % name)
    if os.path.exists(fn):
        return json.load(open(fn))
    import pe59_lib as P
    if name == 'PE':
        cap, cnt = P.pe_maps()
        T = _pe_like(P.build_pe(), cap, cnt['sex2'], P.tablet_type, P.ncls, P.value)
    elif name == 'PC':
        cap, cnt = P.pc_maps()
        T = _pe_like(P.build_pc(), cap, cnt['S'], P.tablet_type, P.ncls, P.value)
    elif name == 'U3':
        from fractions import Fraction as Fr
        src = json.load(open(os.path.join(DATA, 'pe38_ckpt', 'ur3_docs.json')))
        rng = random.Random(seed('pe68-u3'))
        rng.shuffle(src)
        T = []
        for d in src:
            L = []
            nnum = 0
            for i, l in enumerate(d['lines']):
                toks = [x for x in l['toks'] if x and '...' not in x and x != 'x']
                cls = v = None
                if l['sys'] in (1, 2) and l['val'] is not None:
                    cls = 'CAP' if l['sys'] == 2 else 'CNT'
                    try:
                        v = float(Fr(l['val']))
                    except Exception:
                        v = None
                    nnum += 1
                role = 'E' if cls else ('H' if i == 0 and toks else 'N')
                L.append({'i': i, 'surf': l.get('surf', 'obverse'), 'role': role, 'cls': cls, 'v': v, 'ev': int(cls == 'CNT'),
                          'fin': toks[-1] if (toks and cls) else None, 'hdr': None, 'signs': toks})
            if 4 <= len(L) <= 60 and nnum >= 3:
                T.append({'id': d['id'], 'site': d['site'], 'lines': L})
            if len(T) >= 4000:
                break
    json.dump(T, open(fn, 'w'))
    return T


# ------------------------------------------------------------------ role sets ("readings")
U3_KNOWN = {'MEASURED': {'gur', 'sila3', 'sze', 'zi3', 'kasz', 'i3', 'dabin', 'esza', 'ninda', 'zu2-lum', 'gu2',
                         'zi3-sig15', 'kasz-saga', 'kasz-du', 'sze-ba', 'ninda-ba', 'kasz-ba'},
            'COUNTED': {'udu', 'u8', 'sila4', 'masz2', 'ud5', 'gu4', 'ab2', 'masz', 'udu-niga', 'gu4-niga',
                        'amar', 'kir11', 'sal-sila4', 'anszе', 'dur3', 'eme6', 'gukkal', 'masz-gal'},
            'PERSON': {'gurusz', 'geme2', 'dumu', 'erin2', 'lu2', 'ugula', 'dumu-munus', 'szesz'}}


def role_sets(name):
    """final-sign classes of the reading: CAP-compatible and CNT-compatible sets."""
    if name == 'PE':
        import pe59_lib as P
        R = P.READING['roles']
        cap = set(R['MEASURED']['signs']) | set(R['ALLOT']['signs']) | set(R['PERSON']['signs'])
        cnt = set(R['COUNTED']['signs']) | set(R['PERSON']['signs']) | set(R['FRACLINE']['signs'])
        return {'CAP': cap, 'CNT': cnt}
    if name == 'PC':
        import pe59_lib as P
        K = P.PC_KNOWN
        return {'CAP': set(K['MEASURED']) | set(K['BISCLASS']) | set(K['PERSON']),
                'CNT': set(K['COUNTED']) | set(K['PERSON'])}
    K = U3_KNOWN
    return {'CAP': K['MEASURED'] | K['PERSON'], 'CNT': K['COUNTED'] | K['PERSON']}


def shuffled_role_sets(rs, train, rng):
    freq = Counter(l['fin'] for t in train for l in t['lines'] if l['fin'])
    vocab = sorted(freq, key=lambda s: -freq[s])
    rank = {s: i for i, s in enumerate(vocab)}
    allsig = sorted(rs['CAP'] | rs['CNT'], key=lambda s: rank.get(s, 10 ** 6))
    used, perm = set(), {}
    for s in allsig:
        r = rank.get(s, len(vocab) - 1)
        for d in range(len(vocab)):
            got = None
            for j in (r + d, r - d):
                if 0 <= j < len(vocab) and vocab[j] not in used and rng.random() < 0.5:
                    got = vocab[j]
                    break
            if got:
                used.add(got)
                perm[s] = got
                break
        else:
            perm[s] = vocab[rng.randrange(len(vocab))]
    return {k: {perm[s] for s in v} for k, v in rs.items()}


# ------------------------------------------------------------------ visible-part features
KINDS = ['CAPT', 'CNTT', 'AMBT', 'NONE']


def tablet_kind(lines):
    """CAPT: a capacity sub-unit sign; CNTT: count-only evidence (fraction, bisexagesimal, N01 >= 6);
    AMBT: numerals readable in either system; NONE: no clean numeral."""
    cl = [l['cls'] for l in lines if l['cls']]
    if 'CAP' in cl:
        return 'CAPT'
    if any(l.get('ev') for l in lines):
        return 'CNTT'
    return 'AMBT' if cl else 'NONE'


class Stats:
    """everything a completion model may learn from the training tablets."""

    def __init__(self, train, rs):
        self.rs = rs
        fins = Counter()
        fin_k = {k: Counter() for k in KINDS}
        hdr_k = {k: Counter() for k in KINDS}
        hdr = Counter()
        capk = Counter()
        xs = {'CAP': [], 'CNT': []}
        for t in train:
            k = tablet_kind(t['lines'])
            for l in t['lines']:
                if l['fin']:
                    fins[l['fin']] += 1
                    fin_k[k][l['fin']] += 1
                if l['hdr']:
                    hdr[l['hdr']] += 1
                    hdr_k[k][l['hdr']] += 1
                if l['cls']:
                    capk[l['cls']] += 1
                    if l['v'] and l['v'] > 0:
                        xs[l['cls']].append(math.log10(l['v']))
        self.vocab = sorted(fins)
        self.V = len(self.vocab) + 1
        self.fins, self.N = fins, sum(fins.values())
        self.fin_k = fin_k
        self.hdr, self.hdr_k = hdr, hdr_k
        self.HV = len(hdr) + 1
        self.base_cap = capk['CAP'] / max(1, capk['CAP'] + capk['CNT'])
        self.mu = {c: float(np.mean(v)) if v else 0.0 for c, v in xs.items()}
        self.sd = {c: float(np.std(v)) + 0.05 if v else 1.0 for c, v in xs.items()}
        # role distributions: reading's sign sets weighted by training frequency on that tablet kind
        self.role = {}
        for k, S in (('CAPT', rs['CAP']), ('CNTT', rs['CNT']), ('AMBT', rs['CAP'] | rs['CNT']),
                     ('NONE', rs['CAP'] | rs['CNT'])):
            src = fin_k[k] if sum(fin_k[k].values()) > 50 else fins
            c = {s: src[s] + 0.5 for s in S if s in fins or s in src}
            z = sum(c.values()) or 1.0
            self.role[k] = {s: v / z for s, v in c.items()}

    def uni(self, s):
        return (self.fins.get(s, 0) + 0.5) / (self.N + 0.5 * self.V)

    def hdr_p(self, h, k=None, a=1.0):
        C = self.hdr_k[k] if k else self.hdr
        n = sum(C.values())
        return (C.get(h, 0) + a * (self.hdr.get(h, 0) + 0.5) / (sum(self.hdr.values()) + 0.5 * self.HV)) / (n + a)


def case_from(t, vis_idx, hid_idx, tag):
    L = t['lines']
    V = [L[i] for i in vis_idx]
    H = [L[i] for i in hid_idx]
    return {'id': t['id'], 'tag': tag, 'V': V, 'H': H, 'kind_full': tablet_kind(L)}


def precompute(case, st, rs_list):
    """arrays for fast scoring of many random completion models on one case.
    rs_list: list of Stats-compatible role dicts (index 0 = the reading, others = shuffled readings)."""
    V, H = case['V'], case['H']
    vk = tablet_kind(V)
    vf = Counter(l['fin'] for l in V if l['fin'])
    nv = sum(vf.values())
    hf = [l['fin'] for l in H if l['fin']]
    u = np.array([st.uni(s) for s in hf])
    c = np.array([vf.get(s, 0) / nv if nv else 0.0 for s in hf])
    r = np.array([[R[vk].get(s, 0.0) for s in hf] for R in rs_list]) if hf else np.zeros((len(rs_list), 0))
    vcls = [l['cls'] for l in V if l['cls']]
    hcls = [l['cls'] for l in H if l['cls']]
    vx = [math.log10(l['v']) for l in V if l['v'] and l['v'] > 0]
    hx = [(math.log10(l['v']), l['cls']) for l in H if l['v'] and l['v'] > 0]
    hh = [l['hdr'] for l in H if l['hdr']]
    return {'vk': vk, 'n_cap_v': vcls.count('CAP'), 'n_num_v': len(vcls),
            'h_cap': hcls.count('CAP'), 'h_cnt': hcls.count('CNT'),
            'u': u, 'c': c, 'r': r, 'nv_fin': nv, 'vx': vx, 'hx': hx, 'hh': hh,
            'hdr_p_k': [math.log2(st.hdr_p(h, vk)) for h in hh], 'hdr_p0': [math.log2(st.hdr_p(h)) for h in hh]}


def sys_rates(train_cases_pc):
    """P(hidden numeric line is CAP | visible kind), estimated on training pseudo-cuts (the one-system rule)."""
    a = defaultdict(lambda: [0.5, 1.0])
    for p in train_cases_pc:
        a[p['vk']][0] += p['h_cap']
        a[p['vk']][1] += p['h_cap'] + p['h_cnt']
    return {k: v[0] / v[1] for k, v in a.items()}


def score_model(p, m, st, rates, ridx=0):
    """bits per case (lower is better), by feature. m: model dict."""
    out = {}
    # (a) system of hidden numeric lines
    n1, n0 = p['h_cap'], p['h_cnt']
    if n1 + n0:
        if m['SYS']:
            q = rates.get(p['vk'], st.base_cap)
            q = m['sysmix'] * q + (1 - m['sysmix']) * (p['n_cap_v'] + m['a'] * st.base_cap) / (p['n_num_v'] + m['a'])
        else:
            q = (p['n_cap_v'] + m['a'] * st.base_cap) / (p['n_num_v'] + m['a'])
        q = min(max(q, 1e-3), 1 - 1e-3)
        out['sys'] = -(n1 * math.log2(q) + n0 * math.log2(1 - q))
    # (b) final (class) signs
    if len(p['u']):
        lc, lr, lu = m['lam']
        if not p['nv_fin']:
            lu += lc; lc = 0.0
        if not m['ROLE']:
            lu += lr; lr = 0.0
        prob = lc * p['c'] + lr * p['r'][ridx] + lu * p['u']
        prob = np.maximum(prob, 1e-6) if lu == 0 else prob
        out['sign'] = float(-np.log2(prob).sum())
    # (c) magnitude (0.1-dex bins)
    if p['hx']:
        bits = 0.0
        mv = float(np.mean(p['vx'])) if p['vx'] else None
        for x, cl in p['hx']:
            if m['SYS']:
                q = rates.get(p['vk'], st.base_cap)
                mu0 = q * st.mu['CAP'] + (1 - q) * st.mu['CNT']
                sd0 = max(st.sd['CAP'], st.sd['CNT'])
            else:
                mu0 = st.base_cap * st.mu['CAP'] + (1 - st.base_cap) * st.mu['CNT']
                sd0 = max(st.sd['CAP'], st.sd['CNT'])
            mu = mu0 if mv is None else m['w'] * mv + (1 - m['w']) * mu0
            sd = sd0 * m['sds']
            dens = math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) * 0.1
            dens = 0.98 * dens + 0.02 * 0.1 / 8.0
            bits += -math.log2(dens)
        out['mag'] = bits
    # (d) header
    if p['hh']:
        out['hdr'] = -sum(p['hdr_p_k'] if m['HDR'] else p['hdr_p0'])
    out['total'] = sum(out.values())
    return out


def random_model(rng):
    lam = rng.dirichlet([1.0, 1.0, 1.0])
    lam[2] = max(lam[2], 0.02)
    lam = lam / lam.sum()
    return {'lam': [float(x) for x in lam], 'a': float(np.exp(rng.uniform(math.log(0.3), math.log(30)))),
            'w': float(rng.uniform(0, 1)), 'sds': float(np.exp(rng.uniform(math.log(0.4), math.log(1.6)))),
            'sysmix': float(rng.uniform(0.2, 1.0)),
            'SYS': int(rng.random() < 0.5), 'ROLE': int(rng.random() < 0.5), 'HDR': int(rng.random() < 0.5)}


CORPUS_NULL = {'lam': [0.0, 0.0, 1.0], 'a': 1e9, 'w': 0.0, 'sds': 1.0, 'sysmix': 0.0, 'SYS': 0, 'ROLE': 0, 'HDR': 0}


def pseudo_cuts(T, fracs, rng, per_tab=1, min_lines=6):
    """cut a tablet into a visible contiguous block (share drawn from the real fragment shares) and hidden rest."""
    out = []
    for t in T:
        n = len(t['lines'])
        if n < min_lines or sum(1 for l in t['lines'] if l['cls']) < 3:
            continue
        for _ in range(per_tab):
            f = rng.choice(fracs)
            k = min(n - 1, max(2, int(round(f * n))))
            s = rng.randrange(0, n - k + 1)
            vis = list(range(s, s + k))
            hid = [i for i in range(n) if i < s or i >= s + k]
            out.append(case_from(t, vis, hid, 'pseudo'))
    return out
