#!/usr/bin/env python3
"""LA-45 shared code: TWO MACHINES ARGUE UNTIL THE MEANINGS SETTLE.

Adversarial self-play over word meanings.
  Proposer: assigns every word / sign type (opaque id; no sound values) one of up to Kmax abstract
            meanings. A meaning is defined only by what it entails: a class-bigram of meanings
            (what may follow what), which types it emits, and occurrence frames (what comes next:
            end / small number / large number / fraction / word; whether the adjacent number equals
            the running sum of the section; document length). Trained by collapsed Gibbs sweeps on
            a training part with an MDL cost per meaning (lambda nats + assignment code length).
  Critic:   holds a vault of documents the Proposer never sees and evolves the subset of the vault
            that hurts the Proposers most. It scores held-out bits gained over a one-meaning model
            and counts CONTRADICTIONS: held-out occurrences that the meanings call (near-)impossible
            (any frame or transition with predictive probability < 0.01 in a row seen >= 30 times).
  Population-based training: 12 Proposers x 6 Critics per population; worst Proposers copy the best
  and perturb lambda / Kmax; the least damaging Critic copies the most damaging and swaps documents.
  Independent populations (different seeds and splits) are compared at the end.
The abstract vocabulary names (TOTAL, COMMODITY, ENTRY, HEADER, TERM, RARE) are attached to meanings by
fixed profile rules (name_meaning), set on the controls and frozen before Linear A is read.
"""
import json, os, re, sys, math, random, hashlib, collections
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la45_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
PE_TOOLS = os.path.join(HERE, '..', '..', 'proto-elamite', 'tools')


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# =========================================================== corpora -> docs
# doc = {'id', 'site', 'toks': [('T', typestr) | ('N', value, hasfrac)], 'pub'}

def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        toks = []
        for t in ins['tokens']:
            if t['t'] == 'word':
                toks.append(('T', '-'.join(t['s'])))
            elif t['t'] == 'logo':
                toks.append(('T', 'L:' + t['v']))
            elif t['t'] == 'num':
                toks.append(('N', float(t['v']), bool(t['frac'])))
        if toks:
            out.append({'id': ins['id'], 'site': ins['site'], 'support': ins['support'], 'toks': toks})
    return out


def lb_docs_all():
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(fn))]
    from la41_common import lb_docs
    L = lb_docs()
    out = []
    for k, d in L.items():
        toks = []
        for it in d['items']:
            if it['w']:
                toks.append(('T', '-'.join(it['w'])))
            for c in it.get('ctx', []):
                toks.append(('T', '-'.join(c)))
            if it['logo']:
                toks.append(('T', 'L:' + it['logo']))
            if it['num'] is not None:
                v = it['val']
                toks.append(('N', float(v), abs(v - round(v)) > 1e-9))
        if toks:
            out.append({'id': k, 'site': d['site'], 'series': d['support'], 'toks': toks})
    json.dump(out, open(fn, 'w'))
    return out


def ur3_docs_all():
    fn = os.path.join(CK, 'ur3_docs_small.json')
    if os.path.exists(fn):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(fn))]
    src = json.load(open(os.path.join(PE_TOOLS, '..', 'data', 'pe38_ckpt', 'ur3_docs.json')))
    rng = random.Random(seed('la45-ur3-pool'))
    rng.shuffle(src)
    out = []
    for d in src[:6000]:
        toks = []
        for l in d['lines']:
            if l['val'] is not None and l['sys'] in (1, 2):
                from fractions import Fraction as Fr
                v = float(Fr(l['val']))
                toks.append(('N', v, abs(v - round(v)) > 1e-9))
            for t in l['toks']:
                if t:
                    toks.append(('T', t))
        if toks:
            out.append({'id': d['id'], 'site': d['site'], 'toks': toks})
    json.dump(out, open(fn, 'w'))
    return out


def ntok(docs):
    return sum(len(d['toks']) for d in docs)


def sample_size(docs, n, rng):
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= n:
            break
        out.append(docs[i]); c += len(docs[i]['toks'])
    return out


# =========================================================== truth (controls only; never used to fit)
UR_TRUTH_W = {
    'COMMODITY': 'udu u8 masz2 ud5 sila4 kir11 gu4 ab2 amar ansze masz gukkal udu-nita2 dusu2 szah2 sila4-ga '
                 'amar-ga gu4-niga u8-sig5 udu-niga sze ziz2 kasz ninda i3 zu2-lum tug2 siki ku3-babbar urudu '
                 'gesz ga numun i3-gesz i3-nun sze-ba gig kasz-saga dabin esza zi3 munu4 sum mun gurusz geme2 '
                 'masz2-gal kusz',
    'UNIT': 'sila3 gur ban2 barig gin2 ma-na dug gu2 sar iku sa',
    'VERB': 'ba-ti ba-zi zi-ga i3-dab5 ba-ug7 ba-an-ti mu-du sa2-du11 szu-ba-ti ba-ab-dab5 mu-kux(du) ba-an-zi '
            'i3-gal2 ba-na-zi ba-usz2 mu-kux(DU)',
    'TOTAL': 'szu-nigin2 szunigin szu-nigin',
    'DEFICIT': 'la2-ia3',
    'DATE': 'iti mu u4',
    'OFFICE': 'ugula nu-banda3 szabra sanga ensi2 kiszib3 giri3 ki',
}
UR_TRUTH = {w: r for r, ws in UR_TRUTH_W.items() for w in ws.split()}


def ur_truth(docs):
    """Lexical truth; PLACE = token with {ki}; PERSON = token after giri3 / kiszib3, or after ki ending -ta
    (majority of the type's occurrences)."""
    tr = dict(UR_TRUTH)
    pn = collections.Counter(); tot = collections.Counter()
    for d in docs:
        ts = d['toks']
        for i, x in enumerate(ts):
            if x[0] != 'T':
                continue
            w = x[1]
            tot[w] += 1
            if i > 0 and ts[i - 1][0] == 'T':
                p = ts[i - 1][1]
                if p in ('giri3', 'kiszib3') or (p == 'ki' and w.endswith('-ta')):
                    pn[w] += 1
    out = {}
    for w, c in tot.items():
        if w in tr:
            out[w] = tr[w]
        elif '{ki}' in w:
            out[w] = 'PLACE'
        elif pn[w] >= max(1, 0.5 * c) and not w.startswith('{d}'):
            out[w] = 'PERSON'
    return out


LB_TRUTH_W = {
    'TOTAL': 'to-so to-sa to-so-de to-sa-de to-so-pa',
    'DEFICIT': 'o-pe-ro o-pe-ro-si o-pe-ra',
    'VERB': 'a-pu-do-si de-ka-sa-to do-so-mo e-ko-si e-ke e-ko-te a-ke-re a-ke-re-se di-do-si o-u-di-do-si '
            'di-do-ke a-pe-do-ke a-pe-e-si a-pe-o-te pa-ro o-da-a2 o-da-a2 e-e-si me-ta-pe-mo da-ma-te',
    'PLACE': 'pa-i-to ku-do-ni-ja a-mi-ni-so ko-no-so tu-ri-so ru-ki-to da-wo e-ra su-ri-mo pu-ro pa-ki-ja-ne '
             'ro-u-so ka-ra-do-ro ri-jo ti-mi-to-a-ko a-pu2-we a-ke-re-wa e-ra-to pe-to-no me-ta-pa za-ma-e-wi-ja '
             'ri-jo-no ku-ta-to qa-mo si-ja-du-we ra-to e-ko-me-no ka-ru-no u-ta-no a-ka-wo-ne ti-ri-to '
             'ko-tu-we e-ki-no-jo pa-ra-ja da-*22-to do-ti-ja ra-su-to tu-ni-ja a-pa-ta-wa e-ra-jo '
             'ka-to-ro u-pa-ta-ro qa-ra pi-ja-se-ra a-ra-ka-te-ja ku-ko-no-jo',
}
LB_TRUTH = {w: r for r, ws in LB_TRUTH_W.items() for w in ws.split()}


def lb_truth(docs):
    out = {}
    first = collections.Counter(); tot = collections.Counter()
    for d in docs:
        ts = d['toks']
        for i, x in enumerate(ts):
            if x[0] != 'T':
                continue
            w = x[1]; tot[w] += 1
            if d.get('series', '').startswith('D') and d['site'] == 'KN' and i == 0:
                first[w] += 1
    for w, c in tot.items():
        if w.startswith('L:'):
            out[w] = 'COMMODITY'
        elif w in LB_TRUTH:
            out[w] = LB_TRUTH[w]
        elif first[w] >= max(1, 0.5 * c):
            out[w] = 'PERSON'
    return out


# =========================================================== planted meaning system
PL_ROLES = ['PERSON', 'PLACE', 'COMMODITY', 'VERB', 'TOTAL', 'DEFICIT', 'SEALWORD']


def planted_docs(n_tok, rng):
    """A made-up administration with known meanings (never Linear A words). Roles:
    PLACE heads a list; VERB (transaction) follows it; entries PERSON [COMMODITY] number; a TOTAL word followed
    by the correct sum (70%; else a slip of +-1..10); a DEFICIT word followed by a number smaller than the last
    entry; single-word sealings with SEALWORD or PERSON; noise: 10% of words replaced by random types."""
    syl = ['%s%s' % (c, v) for c in 'PTKDMNRSWJQZ' for v in 'AEIOU']
    def word(k):
        return '-'.join(rng.choice(syl) for _ in range(k))
    persons = [word(rng.choice([2, 3, 3, 4])) for _ in range(700)]
    pw = [1.0 / (i + 1) ** 0.7 for i in range(700)]
    places = [word(3) for _ in range(25)]
    comms = ['L:C%d' % i for i in range(9)]
    cw = [1.0 / (i + 1) for i in range(9)]
    verbs = [word(2) for _ in range(5)]
    tot_w, def_w = word(2), word(2)
    seals = [word(rng.choice([1, 2])) for _ in range(20)]
    truth = {}
    for w in persons: truth[w] = 'PERSON'
    for w in places: truth[w] = 'PLACE'
    for w in comms: truth[w] = 'COMMODITY'
    for w in verbs: truth[w] = 'VERB'
    truth[tot_w] = 'TOTAL'; truth[def_w] = 'DEFICIT'
    for w in seals: truth.setdefault(w, 'SEALWORD')
    docs, c, k = [], 0, 0
    while c < n_tok:
        k += 1
        if rng.random() < 0.45:
            t = [('T', rng.choice(seals) if rng.random() < 0.6 else rng.choices(persons, pw)[0])]
        else:
            t = []
            if rng.random() < 0.6: t.append(('T', rng.choice(places)))
            if rng.random() < 0.5: t.append(('T', rng.choice(verbs)))
            com = rng.choices(comms, cw)[0] if rng.random() < 0.6 else None
            vals = []
            for _ in range(rng.randint(2, 10)):
                t.append(('T', rng.choices(persons, pw)[0]))
                if com and rng.random() < 0.5: t.append(('T', com))
                v = rng.choice([1, 1, 1, 2, 3, 5, 10, 20, 30, 50])
                fr = rng.random() < (0.3 if com in ('L:C1', 'L:C3') else 0.03)
                vals.append(v); t.append(('N', float(v), fr))
            if rng.random() < 0.5:
                s = sum(vals)
                if rng.random() > 0.7: s += rng.choice([-10, -1, 1, 10])
                t.append(('T', tot_w)); t.append(('N', float(max(1, s)), False))
            elif rng.random() < 0.2:
                t.append(('T', def_w)); t.append(('N', float(max(1, vals[-1] // 2)), False))
        # noise
        t = [('T', word(rng.choice([2, 3]))) if x[0] == 'T' and rng.random() < 0.1 else x for x in t]
        docs.append({'id': 'PL%d' % k, 'site': 'P', 'toks': t}); c += len(t)
    return docs, truth


# =========================================================== shuffles
def shuffle_types(docs, rng):
    """S1: type identities shuffled across all word slots (slots, numbers, documents kept)."""
    ws = [x for d in docs for x in d['toks'] if x[0] == 'T']
    rng.shuffle(ws)
    it = iter(ws)
    return [dict(d, toks=[next(it) if x[0] == 'T' else x for x in d['toks']]) for d in docs]


def shuffle_order(docs, rng):
    """S2: tokens shuffled inside each document."""
    out = []
    for d in docs:
        t = list(d['toks']); rng.shuffle(t)
        out.append(dict(d, toks=t))
    return out


# =========================================================== build arrays
NFRAME = 3
CARD = np.array([5, 3, 3], np.int64)   # next kind; sum relation; doc length


def numkind(x):
    v, fr = x[1], x[2]
    if fr or abs(v - round(v)) > 1e-9: return 3
    return 1 if v < 10 else 2


class Built:
    pass


def build(docs):
    """Occurrence arrays over all docs. Split masks are passed separately (per doc)."""
    types = sorted({x[1] for d in docs for x in d['toks'] if x[0] == 'T'})
    ti = {w: i for i, w in enumerate(types)}
    otype, oprev, octx, odoc, F = [], [], [], [], []
    for di, d in enumerate(docs):
        ts = d['toks']
        # running sum relation per number position
        sumrel = {}
        run, nrun = 0.0, 0
        for i, x in enumerate(ts):
            if x[0] == 'N':
                v = x[1]
                if nrun >= 2 and v >= 2 and abs(v - run) <= 1.0:
                    sumrel[i] = 2; run, nrun = 0.0, 0
                else:
                    sumrel[i] = 1; run += v; nrun += 1
        L = len(ts)
        lb = 0 if L <= 2 else 1 if L <= 8 else 2
        last_occ = -1
        for i, x in enumerate(ts):
            if x[0] != 'T':
                last_occ = -1
                continue
            o = len(otype)
            otype.append(ti[x[1]]); odoc.append(di)
            if i == 0: octx.append(0); oprev.append(-1)
            elif ts[i - 1][0] == 'N': octx.append(1); oprev.append(-1)
            else: octx.append(2); oprev.append(last_occ)
            nx = ts[i + 1] if i + 1 < L else None
            fn = 0 if nx is None else (4 if nx[0] == 'T' else numkind(nx))
            if nx is not None and nx[0] == 'N': fs = sumrel[i + 1]
            elif i > 0 and ts[i - 1][0] == 'N': fs = sumrel[i - 1]
            else: fs = 0
            F.append((fn, fs, lb))
            last_occ = o
    B = Built()
    B.types = types; B.ti = ti; B.T = len(types)
    B.otype = np.array(otype, np.int64); B.oprev = np.array(oprev, np.int64)
    B.octx = np.array(octx, np.int64); B.odoc = np.array(odoc, np.int64)
    B.F = np.array(F, np.int64).reshape(-1, NFRAME)
    B.ndoc = len(docs)
    return B


# =========================================================== numba core
@njit(cache=True)
def _lg(x):
    return math.lgamma(x)


@njit(cache=True)
def objective(assign, Kmax, otype, oprev, octx, F, card, omask, tcount, et, lam, alpha, beta, gamma, LH):
    """(alpha = beta = gamma = 0.5 assumed: LH[k] = lgamma(k/2).) Collapsed log marginal likelihood of the training occurrences (omask) under the meanings, minus
    lam nats per used meaning, minus assignment code length (n_types * log K_used)."""
    K = Kmax
    Tr = np.zeros((K + 2, K))
    Nm = np.zeros(K)
    Vm = np.zeros(K)
    Em = np.zeros(K)
    nf = F.shape[1]
    Fc = np.zeros((nf, K, 6))
    n = otype.shape[0]
    for o in range(n):
        if not omask[o]:
            continue
        m = assign[otype[o]]
        Nm[m] += 1
        c = octx[o]
        if c == 0: r = K
        elif c == 1: r = K + 1
        else: r = assign[otype[oprev[o]]]
        Tr[r, m] += 1
        for f in range(nf):
            Fc[f, m, F[o, f]] += 1
    ntyp = 0
    for t in range(assign.shape[0]):
        if tcount[t] > 0:
            m = assign[t]
            Vm[m] += 1
            Em[m] += et[t]
            ntyp += 1
    ll = 0.0
    used = 0
    for m in range(K):
        if Nm[m] > 0:
            used += 1
            ll += LH[int(2.0 * (Vm[m] * beta) + 0.5)] - LH[int(2.0 * (Nm[m] + Vm[m] * beta) + 0.5)] + Em[m]
            for f in range(nf):
                cf = card[f]
                ll += LH[int(2.0 * (cf * alpha) + 0.5)] - LH[int(2.0 * (Nm[m] + cf * alpha) + 0.5)]
                for v in range(cf):
                    ll += LH[int(2.0 * (Fc[f, m, v] + alpha) + 0.5)] - LH[int(2.0 * (alpha) + 0.5)]
    for r in range(K + 2):
        tot = 0.0
        for m in range(K):
            tot += Tr[r, m]
        if tot > 0:
            ll += LH[int(2.0 * (K * gamma) + 0.5)] - LH[int(2.0 * (tot + K * gamma) + 0.5)]
            for m in range(K):
                ll += LH[int(2.0 * (Tr[r, m] + gamma) + 0.5)] - LH[int(2.0 * (gamma) + 0.5)]
    if used > 1:
        ll -= ntyp * math.log(used)
    ll -= lam * used
    return ll


@njit(cache=True)
def gibbs_sweep(assign, Kmax, order, otype, oprev, octx, F, card, omask, tcount, et, lam, temp, alpha, beta, gamma,
                rs, LH):
    """One sweep: for each train type in order, sample a meaning from exp(obj/temp). rs = uniform randoms.
    Returns number of objective evaluations."""
    sc = np.zeros(Kmax)
    nev = 0
    for k in range(order.shape[0]):
        t = order[k]
        old = assign[t]
        best = -1e300
        # only allow one empty meaning as candidate (the lowest-index empty)
        usedm = np.zeros(Kmax, np.bool_)
        for u in range(assign.shape[0]):
            if tcount[u] > 0 and u != t:
                usedm[assign[u]] = True
        empty_taken = False
        for m in range(Kmax):
            if not usedm[m]:
                if empty_taken:
                    sc[m] = -1e300
                    continue
                empty_taken = True
            assign[t] = m
            sc[m] = objective(assign, Kmax, otype, oprev, octx, F, card, omask, tcount, et, lam, alpha, beta, gamma, LH)
            nev += 1
            if sc[m] > best:
                best = sc[m]
        tot = 0.0
        for m in range(Kmax):
            if sc[m] > -1e299:
                sc[m] = math.exp((sc[m] - best) / temp)
            else:
                sc[m] = 0.0
            tot += sc[m]
        u = rs[k] * tot
        acc = 0.0
        pick = old
        for m in range(Kmax):
            acc += sc[m]
            if u <= acc and sc[m] > 0:
                pick = m
                break
        assign[t] = pick
    return nev


@njit(cache=True)
def heldout(assign, Kmax, otype, oprev, octx, F, card, trmask, temask, tcount, alpha, beta, gamma, eps, minrow,
            out_occ, minc):
    """Predictive scoring of test occurrences (temask) from train counts. Returns (n_scored, model bits,
    baseline bits, n_contradictions); out_occ[o] = per-occurrence gain bits (nan if not scored),
    and sets out_occ for contradictions via negative sentinel in a second array is avoided."""
    K = Kmax
    Tr = np.zeros((K + 2, K)); Nm = np.zeros(K); Vm = np.zeros(K)
    nf = F.shape[1]
    Fc = np.zeros((nf, K, 6)); Fp = np.zeros((nf, 6))
    Ntot = 0.0; Vtot = 0.0
    n = otype.shape[0]
    for o in range(n):
        if not trmask[o]:
            continue
        m = assign[otype[o]]
        Nm[m] += 1; Ntot += 1
        c = octx[o]
        r = K if c == 0 else (K + 1 if c == 1 else assign[otype[oprev[o]]])
        Tr[r, m] += 1
        for f in range(nf):
            Fc[f, m, F[o, f]] += 1; Fp[f, F[o, f]] += 1
    for t in range(assign.shape[0]):
        if tcount[t] > 0:
            Vm[assign[t]] += 1; Vtot += 1
    Trow = np.zeros(K + 2); Tpool = np.zeros(K)
    for r in range(K + 2):
        for m in range(K):
            Trow[r] += Tr[r, m]
            if r < K: Tpool[m] += Tr[r, m]
    Tpt = 0.0
    for m in range(K): Tpt += Tpool[m]
    ns = 0; bm = 0.0; bb = 0.0; nc = 0
    l2 = math.log(2.0)
    for o in range(n):
        out_occ[o] = np.nan
        if not temask[o]:
            continue
        t = otype[o]
        if tcount[t] < minc:
            continue
        m = assign[t]
        c = octx[o]
        contra = False
        if c == 0: r = K
        elif c == 1: r = K + 1
        else:
            pt = otype[oprev[o]]
            r = assign[pt] if tcount[pt] > 0 else -1
        if r >= 0:
            pr = (Tr[r, m] + gamma) / (Trow[r] + K * gamma)
            if Trow[r] >= minrow and pr < eps: contra = True
        else:
            pr = (Tpool[m] + gamma) / (Tpt + K * gamma)
        lm = math.log(pr) + math.log((tcount[t] + beta) / (Nm[m] + Vm[m] * beta))
        lb = math.log((tcount[t] + beta) / (Ntot + Vtot * beta))
        for f in range(nf):
            cf = card[f]; v = F[o, f]
            pf = (Fc[f, m, v] + alpha) / (Nm[m] + cf * alpha)
            if Nm[m] >= minrow and pf < eps: contra = True
            lm += math.log(pf)
            lb += math.log((Fp[f, v] + alpha) / (Ntot + cf * alpha))
        ns += 1; bm += lm / l2; bb += lb / l2
        if contra: nc += 1
        out_occ[o] = (lm - lb) / l2 - (1.0 if contra else 0.0)
    return ns, bm, bb, nc


# =========================================================== the game
class Game:
    def __init__(self, docs, B, split_seed, P=12, C=6, rounds=40, vault=0.2, final=0.2, critic_frac=0.5,
                 lam0=(0.0, 30.0), K0=(3, 12), eps=0.01, minrow=30, train_docs=None, vault_docs=None,
                 final_docs=None, minc=5):
        self.B = B; self.rng = np.random.default_rng(split_seed)
        nd = B.ndoc
        if train_docs is None:
            perm = self.rng.permutation(nd)
            nf, nv = int(final * nd), int(vault * nd)
            final_docs = perm[:nf]; vault_docs = perm[nf:nf + nv]; train_docs = perm[nf + nv:]
        self.dtr = np.zeros(nd, np.bool_); self.dtr[train_docs] = True
        self.vault = np.array(vault_docs); self.final = np.array(final_docs)
        self.trmask = self.dtr[B.odoc]
        self.tcount = np.bincount(B.otype[self.trmask], minlength=B.T).astype(np.float64)
        self.et = np.array([math.lgamma(c + 0.5) - math.lgamma(0.5) for c in self.tcount])
        self.LH = np.array([0.0] + [math.lgamma(k / 2.0) for k in range(1, 2 * (len(B.otype) + B.T) + 200)])
        self.P, self.C, self.rounds = P, C, rounds
        self.KMAX = 12
        self.eps, self.minrow, self.minc = eps, minrow, minc
        self.alpha, self.beta, self.gamma = 0.5, 0.5, 0.5
        self.props = []
        for p in range(P):
            K = int(self.rng.integers(K0[0], K0[1] + 1))
            a = self.rng.integers(0, K, B.T).astype(np.int64)
            self.props.append({'assign': a, 'K': K, 'lam': float(self.rng.uniform(*lam0)), 'temp': 1.0,
                               'fit': -1e9, 'hist': []})
        self.critics = []
        for c in range(C):
            pick = self.rng.random(len(self.vault)) < critic_frac
            self.critics.append({'docs': self.vault[pick], 'dmg': 0.0})
        self.nev = 0
        self.out = np.zeros(len(B.otype))

    def docmask(self, docs):
        m = np.zeros(self.B.ndoc, np.bool_); m[docs] = True
        return m[self.B.odoc]

    def score(self, assign, K, docs):
        B = self.B
        ns, bm, bb, nc = heldout(assign, self.KMAX, B.otype, B.oprev, B.octx, B.F, CARD, self.trmask,
                                 self.docmask(docs), self.tcount, self.alpha, self.beta, self.gamma, self.eps,
                                 self.minrow, self.out, self.minc)
        if ns == 0:
            return 0.0, 0.0, 0
        gain = (bm - bb) / ns
        return gain - nc / ns, nc / ns, ns

    def sweep(self, p):
        B = self.B
        types = np.nonzero(self.tcount > 0)[0]
        order = self.rng.permutation(types)
        rs = self.rng.random(len(order))
        if not hasattr(self, 'csr'):
            self.csr = csr_lists(B, self.trmask)
        self.nev += gibbs_sweep_inc(p['assign'], p['K'], order, B.otype, B.oprev, B.octx, B.F, CARD, self.trmask,
                                    self.tcount, self.et, p['lam'], p['temp'], rs, self.LH, *self.csr)

    def play(self, log=None):
        for rnd in range(self.rounds):
            frac = rnd / max(1, self.rounds - 1)
            for p in self.props:
                p['temp'] = max(0.05, 1.0 - frac)
                self.sweep(p)
            # critics evaluate
            S = np.zeros((self.P, self.C))
            for i, p in enumerate(self.props):
                for j, c in enumerate(self.critics):
                    S[i, j] = self.score(p['assign'], p['K'], c['docs'])[0]
            for i, p in enumerate(self.props):
                p['fit'] = float(S[i].mean())
            for j, c in enumerate(self.critics):
                c['dmg'] = float(-S[:, j].mean())
            if log:
                log('round %d fit max %.4f mean %.4f K %s lam %s' % (
                    rnd, S.mean(1).max(), S.mean(), [nused(p['assign'], self.tcount) for p in self.props],
                    ['%.0f' % p['lam'] for p in self.props]))
            if rnd % 5 == 4 and rnd < self.rounds - 1:
                order = np.argsort([p['fit'] for p in self.props])
                nrep = max(1, self.P // 4)
                for w, b in zip(order[:nrep], order[::-1][:nrep]):
                    src = self.props[b]
                    self.props[w] = {'assign': src['assign'].copy(), 'K': int(np.clip(src['K'] + self.rng.integers(-1, 2), 2, self.KMAX)),
                                     'lam': float(np.clip(src['lam'] * math.exp(self.rng.normal(0, 0.4)) + self.rng.normal(0, 1), 0, 80)),
                                     'temp': src['temp'], 'fit': src['fit'], 'hist': []}
                    a = self.props[w]['assign']; a[a >= self.props[w]['K']] = self.rng.integers(0, self.props[w]['K'])
                corder = np.argsort([c['dmg'] for c in self.critics])
                weak, strong = corder[0], corder[-1]
                d = self.critics[strong]['docs'].copy()
                nsw = max(1, len(d) // 10)
                outside = np.setdiff1d(self.vault, d)
                if len(outside) and len(d) > nsw:
                    drop = self.rng.choice(len(d), nsw, replace=False)
                    d = np.delete(d, drop)
                    d = np.concatenate([d, self.rng.choice(outside, min(nsw, len(outside)), replace=False)])
                self.critics[weak] = {'docs': d, 'dmg': 0.0}
        fits = np.array([p['fit'] for p in self.props])
        ks = np.array([nused(p['assign'], self.tcount) for p in self.props])
        ok = np.nonzero(fits >= fits.max() - 0.002)[0]
        best = ok[np.argmin(ks[ok])]
        return self.props[best]


def nused(a, tcount):
    return len(set(a[tcount > 0].tolist()))


# =========================================================== naming (fixed rules)
VOCAB = ['TOTAL', 'COMMODITY', 'ENTRY', 'HEADER', 'TERM', 'RARE']


def profiles(B, assign, trmask, tcount):
    prof = {}
    ms = sorted(set(assign[tcount > 0].tolist()))
    for m in ms:
        sel = trmask & (assign[B.otype] == m)
        n = sel.sum()
        if n == 0:
            continue
        Fm = B.F[sel]; cm = B.octx[sel]
        adj = ((Fm[:, 0] >= 1) & (Fm[:, 0] <= 3)) | (cm == 1)
        sr = (Fm[:, 1] == 2).sum() / max(1, (Fm[:, 1] > 0).sum())
        tys = np.nonzero((assign == m) & (tcount > 0))[0]
        prof[m] = {'n': int(n), 'ntypes': len(tys), 'adjnum': float(adj.mean()), 'sumrate': float(sr),
                   'nsum': int((Fm[:, 1] == 2).sum()), 'start': float((cm == 0).mean()),
                   'end': float((Fm[:, 0] == 0).mean()), 'frac': float((Fm[:, 0] == 3).mean()),
                   'medfreq': float(np.median(tcount[tys])),
                   'wfreq': float(np.median(np.repeat(tcount[tys], tcount[tys].astype(int)))),
                   'top': [B.types[t] for t in tys[np.argsort(-tcount[tys])][:8]]}
    return prof


def name_meaning(p):
    if p['sumrate'] >= 0.25 and p['nsum'] >= 3:
        return 'TOTAL'
    if p['adjnum'] >= 0.5 and p['wfreq'] >= 8:
        return 'COMMODITY'
    if p['adjnum'] >= 0.35:
        return 'ENTRY'
    if p['start'] >= 0.4:
        return 'HEADER'
    if p['wfreq'] >= 8:
        return 'TERM'
    return 'RARE'


# =========================================================== stability
def ari(a, b):
    from collections import Counter
    n = len(a)
    if n < 2: return 0.0
    cont = Counter(zip(a, b)); ca = Counter(a); cb = Counter(b)
    c2 = lambda x: x * (x - 1) / 2
    s = sum(c2(v) for v in cont.values()); sa = sum(c2(v) for v in ca.values()); sb = sum(c2(v) for v in cb.values())
    e = sa * sb / c2(n); mx = (sa + sb) / 2
    return (s - e) / (mx - e) if mx != e else 0.0


def nmi(a, b):
    from collections import Counter
    n = len(a)
    if n == 0: return 0.0
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    H = lambda c: -sum(v / n * math.log(v / n) for v in c.values())
    I = sum(v / n * math.log(v * n / (ca[x] * cb[y])) for (x, y), v in cab.items())
    ha, hb = H(ca), H(cb)
    return 2 * I / (ha + hb) if ha + hb > 0 else 0.0


def run_population(docs, B, pop_seed, rounds=40, log=None, **kw):
    g = Game(docs, B, pop_seed, rounds=rounds, **kw)
    best = g.play(log)
    a = best['assign']
    prof = profiles(B, a, g.trmask, g.tcount)
    names = {m: name_meaning(p) for m, p in prof.items()}
    fs, fc, fn = g.score(a, best['K'], g.final)
    gain_only = fs + fc
    tnames = {B.types[t]: names.get(int(a[t]), 'NA') for t in range(B.T) if g.tcount[t] > 0}
    tclus = {B.types[t]: int(a[t]) for t in range(B.T) if g.tcount[t] > 0}
    tcnt = {B.types[t]: int(g.tcount[t]) for t in range(B.T) if g.tcount[t] > 0}
    return {'seed': pop_seed, 'K': nused(a, g.tcount), 'lam': best['lam'], 'final_score': fs,
            'final_gain': gain_only, 'final_contra': fc, 'final_n': fn, 'nev': g.nev,
            'prof': {str(m): dict(p, name=names[m]) for m, p in prof.items()},
            'tnames': tnames, 'tclus': tclus, 'tcnt': tcnt}


def stability(runs, minc=3, truth=None):
    """Across populations: mean pairwise ARI of clusterings over types seen >= minc times in every run;
    per-type modal name and agreement; truth recovery if given."""
    common = None
    for r in runs:
        s = {w for w, c in r['tcnt'].items() if c >= minc}
        common = s if common is None else common & s
    common = sorted(common or [])
    aris = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            aris.append(ari([runs[i]['tclus'][w] for w in common], [runs[j]['tclus'][w] for w in common]))
    allw = set().union(*[set(r['tnames']) for r in runs])
    modal = {}
    for w in allw:
        ns = [r['tnames'][w] for r in runs if w in r['tnames']]
        c = collections.Counter(ns).most_common(1)[0]
        modal[w] = (c[0], c[1] / len(ns), len(ns))
    out = {'n_common': len(common), 'ari_mean': float(np.mean(aris)) if aris else float('nan'),
           'ari_min': float(np.min(aris)) if aris else float('nan'),
           'name_agree_common': float(np.mean([modal[w][1] for w in common])) if common else float('nan'),
           'K': [r['K'] for r in runs], 'final_gain': [round(r['final_gain'], 4) for r in runs],
           'final_contra': [round(r['final_contra'], 4) for r in runs]}
    if truth:
        lab = [w for w in common if w in truth]
        # NMI between truth and each run's clusters (labelled common types), vs label-permutation null
        rng = random.Random(7)
        vals, nulls = [], []
        for r in runs:
            a = [truth[w] for w in lab]; b = [r['tclus'][w] for w in lab]
            vals.append(nmi(a, b))
            for _ in range(50):
                aa = a[:]; rng.shuffle(aa); nulls.append(nmi(aa, b))
        out['truth_n'] = len(lab)
        out['truth_cats'] = dict(collections.Counter(truth[w] for w in lab))
        out['truth_nmi'] = float(np.mean(vals)); out['truth_nmi_null'] = float(np.mean(nulls))
        out['truth_nmi_null_sd'] = float(np.std(nulls))
        # name accuracy for TOTAL and COMMODITY
        for cat in ('TOTAL', 'COMMODITY'):
            ws = [w for w in truth if truth[w] == cat and w in modal]
            out['acc_' + cat] = (sum(modal[w][0] == cat for w in ws), len(ws))
        # separation: P(same cluster | same truth) vs different truth, mean over runs
        ss, sd = [], []
        for r in runs:
            for i in range(len(lab)):
                for j in range(i + 1, len(lab)):
                    same = r['tclus'][lab[i]] == r['tclus'][lab[j]]
                    (ss if truth[lab[i]] == truth[lab[j]] else sd).append(same)
        out['p_same_given_same'] = float(np.mean(ss)) if ss else float('nan')
        out['p_same_given_diff'] = float(np.mean(sd)) if sd else float('nan')
        # where truth categories land (modal names)
        land = collections.defaultdict(collections.Counter)
        for w in lab:
            land[truth[w]][modal[w][0]] += 1
        out['truth_to_name'] = {k: dict(v) for k, v in land.items()}
    out['modal'] = {w: list(v) for w, v in modal.items()}
    return out


# =========================================================== cycle 2/3 helpers
def la_docs_dedup():
    """la_docs without joins whose components are also listed (as in la15), plus publication source."""
    src = None
    try:
        import la15_common as L15
        src = L15._pub_source()
    except Exception:
        src = {}
    docs = la_docs()
    ids = {d['id'] for d in docs}
    out = []
    for d in docs:
        if '+' in d['id']:
            parts = re.split(r'\+', re.sub(r'[ab]$', '', d['id']))
            pre = re.match(r'[A-Z]+[A-Za-z]*?(?=\d)', parts[0])
            pre = pre.group(0) if pre else ''
            comp = [parts[0]] + [p if not p[0].isdigit() else pre + p for p in parts[1:]]
            if all(any(i.startswith(c) for i in ids if i != d['id']) for c in comp):
                continue
        out.append(dict(d, pub=src.get(d['id'], 'blank')))
    return out


def targeted_critique(g, assign, docs, margin=0.0):
    """Critic attacks single words: for each committed type (train count >= g.minc), move it to every other used
    meaning, rescore on `docs` (train counts refitted implicitly by heldout()). A type is REFUTED if some move
    raises the held-out score by more than `margin` bits/occ. Returns {type index: (refuted, best delta)}."""
    base = g.score(assign, None, docs)[0]
    used = sorted(set(assign[g.tcount > 0].tolist()))
    res = {}
    for t in np.nonzero(g.tcount >= g.minc)[0]:
        old = assign[t]; best = -1e9
        for m in used:
            if m == old: continue
            assign[t] = m
            s = g.score(assign, None, docs)[0]
            best = max(best, s - base)
        assign[t] = old
        res[int(t)] = (bool(best > margin), float(best))
    return res


def run_population_split(docs, B, pop_seed, train_idx, rest_idx, rounds=40, critique=True, log=None,
                         fixed=None, **kw):
    """Train on train_idx docs; the vault and the final set are random halves of rest_idx (out-of-site),
    or fixed = (vault_idx, final_idx)."""
    rng = np.random.default_rng(pop_seed)
    if fixed is None:
        rest = rng.permutation(np.array(rest_idx))
        h = len(rest) // 2
        vd, fd = rest[:h], rest[h:]
    else:
        vd, fd = np.array(fixed[0]), np.array(fixed[1])
    g = Game(docs, B, pop_seed, rounds=rounds, train_docs=np.array(train_idx), vault_docs=vd,
             final_docs=fd, **kw)
    best = g.play(log)
    a = best['assign']
    prof = profiles(B, a, g.trmask, g.tcount)
    names = {m: name_meaning(p) for m, p in prof.items()}
    fs, fc, fn = g.score(a, best['K'], g.final)
    r = {'seed': pop_seed, 'K': nused(a, g.tcount), 'lam': best['lam'], 'final_score': fs, 'final_gain': fs + fc,
         'final_contra': fc, 'final_n': fn, 'nev': g.nev,
         'prof': {str(m): dict(p, name=names[m]) for m, p in prof.items()},
         'tnames': {B.types[t]: names.get(int(a[t]), 'NA') for t in range(B.T) if g.tcount[t] > 0},
         'tclus': {B.types[t]: int(a[t]) for t in range(B.T) if g.tcount[t] > 0},
         'tcnt': {B.types[t]: int(g.tcount[t]) for t in range(B.T) if g.tcount[t] > 0}}
    if critique:
        tc = targeted_critique(g, a, g.final)
        r['critique'] = {B.types[t]: v for t, v in tc.items()}
        r['refuted_frac'] = float(np.mean([v[0] for v in tc.values()])) if tc else float('nan')
    # one-meaning reference on the same final set
    one = np.zeros_like(a)
    r['final_one'] = g.score(one, 1, g.final)[0]
    return r


def lb_docs_site(site):
    from la41_common import lb_docs
    L = lb_docs(sites=(site,))
    out = []
    for k, d in L.items():
        toks = []
        for it in d['items']:
            if it['w']: toks.append(('T', '-'.join(it['w'])))
            for c in it.get('ctx', []): toks.append(('T', '-'.join(c)))
            if it['logo']: toks.append(('T', 'L:' + it['logo']))
            if it['num'] is not None:
                v = it['val']; toks.append(('N', float(v), abs(v - round(v)) > 1e-9))
        if toks:
            out.append({'id': k, 'site': d['site'], 'series': d['support'], 'toks': toks})
    return out


# =========================================================== incremental Gibbs (same objective, faster)
@njit(cache=True)
def obj_counts(Nm, Vm, Em, Fc, Tr, K, card, ntyp, lam, LH):
    ll = 0.0
    used = 0
    nf = Fc.shape[0]
    for m in range(K):
        if Vm[m] > 0:
            used += 1
            ll += LH[int(Vm[m] + 0.5)] - LH[int(2 * Nm[m] + Vm[m] + 0.5)] + Em[m]
            for f in range(nf):
                cf = card[f]
                ll += LH[int(cf + 0.5)] - LH[int(2 * Nm[m] + cf + 0.5)]
                for v in range(cf):
                    ll += LH[int(2 * Fc[f, m, v] + 1.5)] - LH[1]
    for r in range(K + 2):
        tot = 0.0
        for m in range(K):
            tot += Tr[r, m]
        if tot > 0:
            ll += LH[K] - LH[int(2 * tot + K + 0.5)]
            for m in range(K):
                ll += LH[int(2 * Tr[r, m] + 1.5)] - LH[1]
    if used > 1:
        ll -= ntyp * math.log(used)
    ll -= lam * used
    return ll


@njit(cache=True)
def _row(o, assign, otype, oprev, octx, K):
    c = octx[o]
    if c == 0: return K
    if c == 1: return K + 1
    return assign[otype[oprev[o]]]


@njit(cache=True)
def _move(t, sign, assign, K, otype, oprev, octx, F, occ_ptr, occ_idx, suc_ptr, suc_idx, Nm, Vm, Em, Fc, Tr, et):
    m = assign[t]
    nf = F.shape[1]
    Vm[m] += sign; Em[m] += sign * et[t]
    for k in range(occ_ptr[t], occ_ptr[t + 1]):
        o = occ_idx[k]
        Nm[m] += sign
        for f in range(nf):
            Fc[f, m, F[o, f]] += sign
        Tr[_row(o, assign, otype, oprev, octx, K), m] += sign
    for k in range(suc_ptr[t], suc_ptr[t + 1]):
        o = suc_idx[k]
        Tr[m, assign[otype[o]]] += sign


@njit(cache=True)
def gibbs_sweep_inc(assign, Kmax, order, otype, oprev, octx, F, card, omask, tcount, et, lam, temp, rs, LH,
                    occ_ptr, occ_idx, suc_ptr, suc_idx):
    K = Kmax
    nf = F.shape[1]
    Nm = np.zeros(K); Vm = np.zeros(K); Em = np.zeros(K)
    Fc = np.zeros((nf, K, 6)); Tr = np.zeros((K + 2, K))
    ntyp = 0
    for t in range(assign.shape[0]):
        if tcount[t] > 0:
            ntyp += 1
            m = assign[t]
            Vm[m] += 1; Em[m] += et[t]
    n = otype.shape[0]
    for o in range(n):
        if omask[o]:
            m = assign[otype[o]]
            Nm[m] += 1
            for f in range(nf):
                Fc[f, m, F[o, f]] += 1
            Tr[_row(o, assign, otype, oprev, octx, K), m] += 1
    sc = np.zeros(K)
    nev = 0
    for k in range(order.shape[0]):
        t = order[k]
        old = assign[t]
        _move(t, -1.0, assign, K, otype, oprev, octx, F, occ_ptr, occ_idx, suc_ptr, suc_idx, Nm, Vm, Em, Fc, Tr, et)
        best = -1e300
        empty_taken = False
        for m in range(K):
            if Vm[m] == 0:
                if empty_taken:
                    sc[m] = -1e300
                    continue
                empty_taken = True
            assign[t] = m
            _move(t, 1.0, assign, K, otype, oprev, octx, F, occ_ptr, occ_idx, suc_ptr, suc_idx, Nm, Vm, Em, Fc, Tr, et)
            sc[m] = obj_counts(Nm, Vm, Em, Fc, Tr, K, card, ntyp, lam, LH)
            _move(t, -1.0, assign, K, otype, oprev, octx, F, occ_ptr, occ_idx, suc_ptr, suc_idx, Nm, Vm, Em, Fc, Tr, et)
            nev += 1
            if sc[m] > best:
                best = sc[m]
        tot = 0.0
        for m in range(K):
            if sc[m] > -1e299:
                sc[m] = math.exp((sc[m] - best) / temp)
            else:
                sc[m] = 0.0
            tot += sc[m]
        u = rs[k] * tot
        acc = 0.0
        pick = old
        for m in range(K):
            acc += sc[m]
            if u <= acc and sc[m] > 0:
                pick = m
                break
        assign[t] = pick
        _move(t, 1.0, assign, K, otype, oprev, octx, F, occ_ptr, occ_idx, suc_ptr, suc_idx, Nm, Vm, Em, Fc, Tr, et)
    return nev


def csr_lists(B, trmask):
    occ = [[] for _ in range(B.T)]
    suc = [[] for _ in range(B.T)]
    for o in np.nonzero(trmask)[0]:
        t = B.otype[o]
        occ[t].append(o)
        if B.octx[o] == 2:
            pt = B.otype[B.oprev[o]]
            if pt != t:
                suc[pt].append(o)
    def csr(L):
        ptr = np.zeros(len(L) + 1, np.int64)
        ptr[1:] = np.cumsum([len(x) for x in L])
        idx = np.array([x for l in L for x in l], np.int64)
        return ptr, idx
    return csr(occ) + csr(suc)
