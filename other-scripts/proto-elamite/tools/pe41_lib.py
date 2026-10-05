"""pe41: let machines reinvent writing, then compare.

Toy early-state economies + populations of scribes that must keep records for a reader
with a small sign budget and no language. The population learns (Lewis signalling game
for its lexicon, cultural hill-climbing for what to write where, cost/collision trade-off
for name length). Every invented sign has a known meaning class. Invented systems are
then aligned to a target corpus (PE, proto-cuneiform, shuffled PE, a held-out invented
system) by structure alone: rank-normalised per-sign behaviour + co-occurrence graph
matching (Hungarian assignment with a neighbour-consistency refinement).
"""
import json, math, os, random
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
LABELS = ['OFFICE', 'PLACE', 'ACTION', 'COMMODITY', 'QUALIFIER', 'NAME', 'TOTAL']
L2I = {l: i for i, l in enumerate(LABELS)}
TTYPES = ['ration', 'delivery', 'labor', 'inventory', 'debt']

# ----------------------------------------------------------------------------- economy

def sample_params(rng):
    p = {}
    p['n_off'] = int(rng.integers(1, 26))
    p['n_place'] = int(rng.integers(0, 13))
    p['n_grain'] = int(rng.integers(1, 7))
    p['n_anim'] = int(rng.integers(1, 13))
    p['n_prod'] = int(rng.integers(0, 13))
    p['n_work'] = int(rng.integers(1, 7))
    p['n_qual'] = int(rng.integers(0, 11))
    p['n_act'] = int(rng.integers(1, 6))
    p['n_pers'] = int(round(math.exp(rng.uniform(math.log(10), math.log(1500)))))
    p['mix'] = rng.dirichlet(np.ones(5) * 0.7).tolist()
    p['mean_entries'] = float(math.exp(rng.uniform(math.log(1.5), math.log(15))))
    p['purity'] = float(rng.uniform(0.3, 1.0))
    p['p_total'] = float(rng.uniform(0, 1))
    p['p_named'] = rng.uniform(0, 1, 5).tolist()   # per tablet type
    p['p_qual'] = float(rng.uniform(0, 0.8))
    p['zipf'] = float(rng.uniform(0.4, 1.6))
    p['p_place_rel'] = float(rng.uniform(0, 1))
    p['num_follows_tablet'] = float(rng.uniform(0, 1))
    p['p_frac'] = float(rng.uniform(0, 0.4))
    # writing pressures
    p['cost'] = float(rng.uniform(0, 0.5))
    p['K'] = int(round(math.exp(rng.uniform(math.log(20), math.log(600)))))
    p['f_name'] = float(rng.uniform(0.05, 0.7))
    p['ctx_off'] = float(rng.uniform(0, 1))
    p['ctx_act'] = float(rng.uniform(0, 1))
    p['ctx_place'] = float(rng.uniform(0, 1))
    p['ctx_tot'] = float(rng.uniform(0, 1))
    p['ctx_comm'] = float(rng.uniform(0, 0.6))
    p['p_slot'] = float(rng.uniform(0, 1))      # reader knows slot type from order
    p['order_noise'] = float(rng.uniform(0, 0.4))
    p['innov'] = float(math.exp(rng.uniform(math.log(1e-3), math.log(0.2))))
    p['name_alpha'] = float(rng.uniform(0, 2.0))  # skew of name-sign pool
    p['rebus'] = float(rng.uniform(0, 0.5))       # name pool overlaps lexicon signs
    p['sharp'] = float(rng.uniform(1, 4))         # conventionalisation of the lexicon
    p['n_tab'] = int(rng.integers(500, 1600))
    return p


def zipf_w(n, s, rng):
    if n <= 0:
        return np.zeros(0)
    w = 1.0 / np.arange(1, n + 1) ** s
    w = w[rng.permutation(n)] if False else w
    return w / w.sum()


class Economy:
    def __init__(self, p, rng):
        self.p = p; self.rng = rng
        # items: (type, local index, unit kind)  unit: C (capacity) / S (counted)
        items = []
        for i in range(p['n_off']): items.append(('OFFICE', i, None))
        for i in range(p['n_place']): items.append(('PLACE', i, None))
        for i in range(p['n_act']): items.append(('ACTION', i, None))
        comm = []
        for i in range(p['n_grain']): comm.append(('grain', 'C'))
        for i in range(p['n_anim']): comm.append(('anim', 'S'))
        for i in range(p['n_prod']): comm.append(('prod', 'C' if rng.random() < 0.4 else 'S'))
        for i in range(p['n_work']): comm.append(('work', 'S'))
        self.comm = comm
        for i, c in enumerate(comm): items.append(('COMMODITY', i, c))
        for i in range(p['n_qual']): items.append(('QUALIFIER', i, None))
        items.append(('TOTAL', 0, None))
        self.items = items
        self.idx = {}
        for k, (t, i, _) in enumerate(items):
            self.idx[(t, i)] = k
        z = p['zipf']
        self.w_off = zipf_w(p['n_off'], z, rng)
        self.w_place = zipf_w(p['n_place'], z, rng)
        self.w_pers = zipf_w(p['n_pers'], z * 0.7, rng)
        self.w_qual = zipf_w(p['n_qual'], z, rng)
        kinds = np.array([c[0] for c in comm])
        self.by_kind = {k: np.where(kinds == k)[0] for k in ['grain', 'anim', 'prod', 'work']}
        self.cw = zipf_w(len(comm), z * 0.8, rng)
        # which commodity kinds each tablet type deals with
        self.tkinds = {'ration': ['grain', 'prod'], 'delivery': ['anim', 'prod'], 'labor': ['work'],
                       'inventory': ['grain', 'anim', 'prod', 'work'], 'debt': ['grain', 'anim']}

    def comm_pool(self, ttype):
        pool = np.concatenate([self.by_kind[k] for k in self.tkinds[ttype]])
        if len(pool) == 0:
            pool = np.arange(len(self.comm))
        w = self.cw[pool]; return pool, w / w.sum()

    def tablet(self):
        p, rng = self.p, self.rng
        tt = int(rng.choice(5, p=p['mix']))
        ttype = TTYPES[tt]
        off = int(rng.choice(p['n_off'], p=self.w_off))
        act = tt % p['n_act']
        place = int(rng.choice(p['n_place'], p=self.w_place)) if (p['n_place'] and rng.random() < p['p_place_rel']) else None
        pool, pw = self.comm_pool(ttype)
        main = int(rng.choice(pool, p=pw))
        n = 1 + rng.poisson(p['mean_entries'] - 1)
        ents = []
        for _ in range(n):
            c = main if rng.random() < p['purity'] else int(rng.choice(pool, p=pw))
            q = int(rng.choice(p['n_qual'], p=self.w_qual)) if (p['n_qual'] and self.comm[c][0] in ('anim', 'prod', 'work') and rng.random() < p['p_qual']) else None
            pers = int(rng.choice(p['n_pers'], p=self.w_pers)) if rng.random() < p['p_named'][tt] else None
            ents.append((c, q, pers))
        tot = rng.random() < p['p_total'] and n > 1
        return dict(tt=tt, off=off, act=act, place=place, main=main, ents=ents, total=tot)

# ----------------------------------------------------------------------------- learning

POL = ['hOFF', 'hACT', 'hPLACE', 'hCOMM', 'eCOMMmain', 'eCOMMother', 'eQUAL', 'eACT', 'tMARK', 'tCOMM', 'eNAME']


def expected_reward(pol, tabs, p, acc, name_acc):
    """pol: dict of write probabilities. acc: lexicon accuracy per type. returns mean reward."""
    tot_r = 0.0; tot_u = 0.0; tot_c = 0.0
    for t in tabs:
        n = len(t['ents'])
        a = acc
        # office
        tot_r += pol['hOFF'] * a['OFFICE'] + (1 - pol['hOFF']) * p['ctx_off']; tot_u += 1; tot_c += pol['hOFF']
        # action
        wA = pol['hACT']; we = pol['eACT']
        tot_r += wA * a['ACTION'] + (1 - wA) * (we * a['ACTION'] + (1 - we) * p['ctx_act']); tot_u += 1
        tot_c += wA + (1 - wA) * we * n
        if t['place'] is not None:
            tot_r += pol['hPLACE'] * a['PLACE'] + (1 - pol['hPLACE']) * p['ctx_place']; tot_u += 1; tot_c += pol['hPLACE']
        hc = pol['hCOMM']; tot_c += hc
        for (c, q, pers) in t['ents']:
            if c == t['main']:
                w = pol['eCOMMmain']
                tot_r += w * a['COMMODITY'] + (1 - w) * (hc * a['COMMODITY'] + (1 - hc) * p['ctx_comm'])
            else:
                w = pol['eCOMMother']
                tot_r += w * a['COMMODITY'] + (1 - w) * p['ctx_comm'] * 0.3
            tot_u += 1; tot_c += w
            if q is not None:
                tot_r += pol['eQUAL'] * a['QUALIFIER']; tot_u += 1; tot_c += pol['eQUAL']
            if pers is not None:
                tot_r += pol['eNAME'] * name_acc; tot_u += 1; tot_c += pol['eNAME'] * name_acc_len[0]
        if t['total']:
            tot_r += pol['tMARK'] + (1 - pol['tMARK']) * p['ctx_tot']; tot_u += 1
            tot_c += pol['tMARK'] + pol['tCOMM']
            tot_r += 0.5 * (pol['tCOMM'] * a['COMMODITY'] + (1 - pol['tCOMM']) * hc * a['COMMODITY']); tot_u += 0.5
    return tot_r / tot_u - p['cost'] * tot_c / tot_u


name_acc_len = [1]


class Population:
    def __init__(self, p, seed):
        self.p = p; self.rng = np.random.default_rng(seed)
        self.econ = Economy(p, self.rng)
        self.learn()

    def learn(self):
        p, rng, E = self.p, self.rng, self.econ
        sample = [E.tablet() for _ in range(300)]
        # usage frequency of each item in the sample
        use = np.zeros(len(E.items)) + 0.2
        for t in sample:
            use[E.idx[('OFFICE', t['off'])]] += 1
            use[E.idx[('ACTION', t['act'])]] += 1
            if t['place'] is not None: use[E.idx[('PLACE', t['place'])]] += 1
            for c, q, pers in t['ents']:
                use[E.idx[('COMMODITY', c)]] += 1
                if q is not None: use[E.idx[('QUALIFIER', q)]] += 1
            if t['total']: use[E.idx[('TOTAL', 0)]] += 1
        self.use = use
        # sign budget split: name pool vs lexicon
        K = p['K']
        any_names = sum(1 for t in sample for e in t['ents'] if e[2] is not None) > 0
        Kn = max(3, int(round(p['f_name'] * K))) if any_names else 0
        Kl = max(4, K - Kn)
        self.Kl, self.Kn = Kl, Kn
        # Lewis signalling game (Roth-Erev) for the lexicon
        nI = len(E.items)
        W = np.ones((nI, Kl)) + rng.random((nI, Kl)) * 0.1
        R = np.ones((Kl, nI)) + rng.random((Kl, nI)) * 0.1
        types = np.array([L2I[it[0]] for it in E.items])
        pu = use / use.sum()
        nround = 40 * nI
        its = rng.choice(nI, size=nround, p=pu)
        slotk = rng.random(nround) < p['p_slot']
        u1 = rng.random(nround); u2 = rng.random(nround)
        for r in range(nround):
            i = its[r]
            cw = np.cumsum(W[i]); s = int(np.searchsorted(cw, u1[r] * cw[-1]))
            if s >= Kl: s = Kl - 1
            row = R[s]
            if slotk[r]:
                row = row * (types == types[i])
            cr = np.cumsum(row); j = int(np.searchsorted(cr, u2[r] * cr[-1]))
            if j >= nI: j = nI - 1
            if j == i:
                W[i, s] += 1.0; R[s, i] += 1.0
        self.W = W
        # accuracy per type (reader argmax with slot knowledge mix)
        Wp = W ** p['sharp']; Wp /= Wp.sum(1, keepdims=True)
        acc = {}
        for tname in LABELS:
            if tname == 'NAME':
                continue
            ids = np.where(types == L2I[tname])[0]
            if len(ids) == 0:
                acc[tname] = 1.0; continue
            a = 0.0; ws = 0.0
            for i in ids:
                for s in np.argsort(-Wp[i])[:3]:
                    row = R[s]
                    j1 = np.argmax(row); j2 = np.argmax(row * (types == types[i]))
                    a += pu[i] * Wp[i, s] * (p['p_slot'] * (j2 == i) + (1 - p['p_slot']) * (j1 == i))
                ws += pu[i]
            acc[tname] = a / ws if ws else 1.0
        self.Wp = Wp
        # names: choose length L from cost/collision trade-off
        self.L = 0; name_acc = 0.0
        if Kn:
            neff = Kn / (1 + p['name_alpha'])
            best = None
            for L in range(1, 6):
                coll = 1 - math.exp(-p['n_pers'] / max(neff ** L, 1e-9))
                rew = (1 - coll) - p['cost'] * L + rng.normal(0, 0.03)
                if best is None or rew > best[0]:
                    best = (rew, L, 1 - coll)
            _, self.L, name_acc = best
        name_acc_len[0] = max(self.L, 1)
        # cultural hill-climbing of the writing policy
        logit = rng.normal(0, 1, len(POL))
        sub = sample[:120]

        def pol_of(lg):
            return {k: 1 / (1 + math.exp(-v)) for k, v in zip(POL, lg)}
        cur = expected_reward(pol_of(logit), sub, p, acc, name_acc)
        for g in range(60):
            cand = logit + rng.normal(0, 0.6, len(POL))
            r = expected_reward(pol_of(cand), sub, p, acc, name_acc)
            if r >= cur:
                logit, cur = cand, r
        self.pol = pol_of(logit)
        self.acc = acc; self.name_acc = name_acc
        # orders (conventions): header and entry
        self.hord = list(rng.permutation(['OFFICE', 'ACTION', 'PLACE', 'COMMODITY']))
        self.eord = list(rng.permutation(['NAME', 'QUALIFIER', 'COMMODITY', 'ACTION']))
        # sign ids: lexicon 0..Kl-1, names Kl..Kl+Kn-1; rebus overlap
        self.name_signs = np.arange(Kl, Kl + Kn)
        if Kn and p['rebus'] > 0:
            nre = int(round(p['rebus'] * Kn))
            if nre:
                self.name_signs[:nre] = rng.choice(Kl, size=nre, replace=False)
        nw = 1.0 / np.arange(1, Kn + 1) ** p['name_alpha'] if Kn else np.zeros(0)
        self.nw = nw / nw.sum() if Kn else nw
        self.names = {}

    def name(self, pers):
        if pers not in self.names:
            self.names[pers] = [int(x) for x in self.rng.choice(self.name_signs, size=self.L, p=self.nw)]
        return self.names[pers]

    def write_item(self, t, i):
        k = self.econ.idx[(t, i)]
        cw = np.cumsum(self.Wp[k]); s = int(np.searchsorted(cw, self.rng.random() * cw[-1]))
        return min(s, self.Kl - 1)

    def write_corpus(self, n_tab=None):
        p, rng, E, pol = self.p, self.rng, self.econ, self.pol
        n_tab = n_tab or p['n_tab']
        tablets = []
        label_tok = {}   # sign -> Counter of labels (by usage)
        nxt = [self.Kl + self.Kn + 10]
        variants = {}

        def emit(s, lab):
            if rng.random() < p['innov']:
                vs = variants.setdefault(s, [])
                if not vs or rng.random() < 0.5:
                    vs.append(nxt[0]); nxt[0] += 1
                s = vs[int(rng.integers(len(vs)))]
            d = label_tok.setdefault(s, np.zeros(len(LABELS)))
            d[L2I[lab]] += 1
            return s

        def maybe_shuffle(seq):
            if len(seq) > 1 and rng.random() < p['order_noise']:
                seq = list(seq); rng.shuffle(seq)
            return seq
        for _ in range(n_tab):
            t = E.tablet()
            lines = []
            h = []
            for typ in self.hord:
                if typ == 'OFFICE' and rng.random() < pol['hOFF']:
                    h.append(('OFFICE', t['off']))
                elif typ == 'ACTION' and rng.random() < pol['hACT']:
                    h.append(('ACTION', t['act']))
                elif typ == 'PLACE' and t['place'] is not None and rng.random() < pol['hPLACE']:
                    h.append(('PLACE', t['place']))
                elif typ == 'COMMODITY' and rng.random() < pol['hCOMM']:
                    h.append(('COMMODITY', t['main']))
            hc = any(x[0] == 'COMMODITY' for x in h)
            h = maybe_shuffle(h)
            if h:
                lines.append(dict(role='header', signs=[emit(self.write_item(a, b), a) for a, b in h], ncls=None))
            mainunit = E.comm[t['main']][1]
            eact = (not any(x[0] == 'ACTION' for x in h)) and rng.random() < pol['eACT']
            for (c, q, pers) in t['ents']:
                parts = []
                for typ in self.eord:
                    if typ == 'COMMODITY':
                        w = pol['eCOMMmain'] if c == t['main'] else pol['eCOMMother']
                        if rng.random() < w:
                            parts.append([('COMMODITY', c)])
                    elif typ == 'QUALIFIER' and q is not None and rng.random() < pol['eQUAL']:
                        parts.append([('QUALIFIER', q)])
                    elif typ == 'NAME' and pers is not None and self.L and rng.random() < pol['eNAME']:
                        parts.append([('NAME', s) for s in self.name(pers)])
                    elif typ == 'ACTION' and eact:
                        parts.append([('ACTION', t['act'])])
                parts = maybe_shuffle(parts)
                signs = []
                for part in parts:
                    for a, b in part:
                        if a == 'NAME':
                            signs.append(emit(b, 'NAME'))
                        else:
                            signs.append(emit(self.write_item(a, b), a))
                unit = E.comm[c][1]
                if rng.random() < p['num_follows_tablet']:
                    unit = mainunit
                if unit == 'C' and rng.random() < p['p_frac'] and t['tt'] == 0:
                    unit = 'F'
                lines.append(dict(role='entry', signs=signs, ncls=unit))
            if t['total']:
                ts = []
                if rng.random() < pol['tMARK']:
                    ts.append(emit(self.write_item('TOTAL', 0), 'TOTAL'))
                if rng.random() < pol['tCOMM']:
                    ts.append(emit(self.write_item('COMMODITY', t['main']), 'COMMODITY'))
                lines.append(dict(role='total', signs=ts, ncls=mainunit))
            tablets.append(lines)
        labels = {s: int(np.argmax(v)) for s, v in label_tok.items()}
        return tablets, labels

# ----------------------------------------------------------------------------- real corpora

CAP = ('N39', 'N24', 'N25', 'N26', 'N28', 'N29', 'N30', 'N36')
FRAC = ('N08', 'N8', 'N02')


def ncls_of(nums):
    codes = [n for _, n in nums]
    if not codes:
        return None
    if any(c.startswith(CAP) for c in codes):
        return 'C'
    if any(c.startswith(FRAC) for c in codes):
        return 'F'
    return 'S'


BAD = {'x', 'n', 'X', 'N'}


def load_real(name):
    f = {'PE': 'pe_corpus.json', 'PC': 'pe2_pc_corpus.json'}[name]
    d = json.load(open(os.path.join(DATA, f)))
    out = []; ids = []
    for t in d:
        lines = []; seen_num = False
        for l in t['lines']:
            sg = [s for s in l['signs'] if s not in BAD and not s.startswith('N')]
            nc = ncls_of(l['numerals'])
            if nc is None:
                if sg and not seen_num and l['surface'] == 'obverse':
                    lines.append(dict(role='header', signs=sg, ncls=None))
                continue
            seen_num = True
            role = 'total' if l['surface'] == 'reverse' else 'entry'
            lines.append(dict(role=role, signs=sg, ncls=nc))
        if lines:
            out.append(lines); ids.append(t['id'])
    return out, ids


def shuffle_corpus(tabs, rng):
    toks = [s for t in tabs for l in t for s in l['signs']]
    perm = rng.permutation(len(toks)); toks = [toks[i] for i in perm]
    k = 0; out = []
    for t in tabs:
        nt = []
        for l in t:
            n = len(l['signs']); nt.append(dict(role=l['role'], signs=toks[k:k + n], ncls=l['ncls'])); k += n
        out.append(nt)
    return out

# ----------------------------------------------------------------------------- features

FEAT = ['logfreq', 'header', 'total', 'init', 'final', 'solo', 'cap', 'frac', 'disp', 'relpos',
        'partners', 'linelen', 'nb_header', 'nb_solo', 'nb_cap', 'nb_final']


def features(tabs, fmin=8, maxsigns=None, shrink=0.0):
    from collections import defaultdict, Counter
    cnt = Counter(s for t in tabs for l in t for s in l['signs'])
    keep = [s for s, c in cnt.most_common() if c >= fmin]
    if maxsigns:
        keep = keep[:maxsigns]
    ix = {s: i for i, s in enumerate(keep)}
    n = len(keep)
    F = np.zeros((n, 12)); tabsets = [set() for _ in range(n)]
    entry_tok = np.zeros(n); multi_tok = np.zeros(n)
    A = np.zeros((n, n))
    partners = [set() for _ in range(n)]
    for ti, t in enumerate(tabs):
        nl = len(t)
        for li, l in enumerate(t):
            sg = l['signs']; L = len(sg)
            for k, s in enumerate(sg):
                i = ix.get(s)
                if i is None:
                    continue
                F[i, 0] += 1
                tabsets[i].add(ti)
                F[i, 9] += li / max(nl - 1, 1)
                F[i, 11] += L
                if l['role'] == 'header': F[i, 1] += 1
                elif l['role'] == 'total': F[i, 2] += 1
                if l['role'] == 'entry':
                    entry_tok[i] += 1
                    if L == 1: F[i, 5] += 1
                    else:
                        multi_tok[i] += 1
                        if k == 0: F[i, 3] += 1
                        if k == L - 1: F[i, 4] += 1
                    if l['ncls'] == 'C': F[i, 6] += 1
                    if l['ncls'] == 'F': F[i, 7] += 1
                for k2, s2 in enumerate(sg):
                    j = ix.get(s2)
                    if k2 != k and j is not None:
                        A[i, j] += 1
                    if k2 != k:
                        partners[i].add(s2)
    tok = F[:, 0].copy()
    G = np.zeros((n, len(FEAT)))
    G[:, 0] = np.log(tok)
    G[:, 1] = F[:, 1] / tok; G[:, 2] = F[:, 2] / tok
    G[:, 3] = F[:, 3] / np.maximum(multi_tok, 1); G[:, 4] = F[:, 4] / np.maximum(multi_tok, 1)
    G[:, 5] = F[:, 5] / np.maximum(entry_tok, 1)
    G[:, 6] = F[:, 6] / np.maximum(entry_tok, 1); G[:, 7] = F[:, 7] / np.maximum(entry_tok, 1)
    G[:, 8] = np.array([len(x) for x in tabsets]) / tok
    G[:, 9] = F[:, 9] / tok
    G[:, 10] = np.array([len(x) for x in partners]) / tok
    G[:, 11] = F[:, 11] / tok
    if shrink > 0:
        # empirical-Bayes shrinkage of each share toward the corpus-wide token rate
        for j, den in [(1, tok), (2, tok), (3, multi_tok), (4, multi_tok), (5, entry_tok), (6, entry_tok), (7, entry_tok)]:
            num = G[:, j] * np.maximum(den, 1) if j not in (1, 2) else G[:, j] * tok
            mu = num.sum() / max(den.sum(), 1)
            G[:, j] = (num + shrink * mu) / (den + shrink)
    An = A / np.maximum(A.sum(1, keepdims=True), 1)
    G[:, 12] = An @ G[:, 1]; G[:, 13] = An @ G[:, 5]; G[:, 14] = An @ G[:, 6]; G[:, 15] = An @ G[:, 4]
    return keep, G, An


def rankq(G):
    from scipy.stats import rankdata
    R = np.zeros_like(G)
    for j in range(G.shape[1]):
        R[:, j] = (rankdata(G[:, j]) - 0.5) / G.shape[0]
    return R


def align(Ga, Aa, Gb, Ab, lam=1.0, iters=2):
    """assign rows of a (target) to rows of b (invented). returns match array (len a, -1 if none) and cost."""
    D = np.sqrt(((Ga[:, None, :] - Gb[None, :, :]) ** 2).sum(-1))
    r, c = linear_sum_assignment(D)
    for _ in range(iters):
        # neighbour consistency: S[i,j] = sum_k Aa[i,k] * Ab[j, pi(k)]
        P = np.zeros((Ga.shape[0], Gb.shape[0])); P[r, c] = 1
        S = Aa @ P @ Ab.T
        S = S / (S.max() + 1e-12)
        r, c = linear_sum_assignment(D - lam * S)
    m = -np.ones(Ga.shape[0], dtype=int); m[r] = c
    return m, float(D[r, c].mean())


def corpus_stats(tabs):
    """global shape stats to compare an invented corpus with PE."""
    from collections import Counter
    cnt = Counter(s for t in tabs for l in t for s in l['signs'])
    tot = sum(cnt.values()); ntypes = len(cnt)
    hap = sum(1 for c in cnt.values() if c == 1) / max(ntypes, 1)
    ents = [l for t in tabs for l in t if l['role'] == 'entry']
    lens = np.array([len(l['signs']) for l in ents]) if ents else np.zeros(1)
    hdr = np.mean([any(l['role'] == 'header' for l in t) for t in tabs])
    totl = np.mean([any(l['role'] == 'total' for l in t) for t in tabs])
    epert = len(ents) / len(tabs)
    cap = np.mean([l['ncls'] == 'C' for l in ents]) if ents else 0
    top10 = sum(c for _, c in cnt.most_common(10)) / max(tot, 1)
    return np.array([math.log(max(ntypes, 1)), hap, np.mean(lens == 0), np.mean(lens == 1), np.mean(lens == 2),
                     np.mean(lens >= 3), hdr, totl, math.log(epert), cap, top10, tot / len(tabs)])

STATN = ['log_types', 'hapax', 'len0', 'len1', 'len2', 'len3+', 'header_tab', 'total_tab', 'log_entries_per_tab', 'cap_share', 'top10_share', 'signs_per_tab']


def prep(G):
    """cycle 2 normalisation: frequency as rank quantile, behaviour shares kept on their raw 0-1 scale
    (rank-normalising shares turned shuffle noise into full-range features in cycle 1)."""
    from scipy.stats import rankdata
    H = G.copy()
    H[:, 0] = (rankdata(G[:, 0]) - 0.5) / G.shape[0]
    H[:, 10] = np.minimum(G[:, 10], 1.0)
    H[:, 11] = np.minimum(G[:, 11] / 6.0, 1.0)
    return H
