"""Loop 42: the grammar as an authenticity scorer.

Rules (all parameters fitted on the FIT set = Mohenjo-daro + Harappa objects with an excavation register number):
  R1 REPEAT    a sign written twice, not adjacent (S88, S-DARK-15); signs that repeat in >= 3 fit texts are exempt
  R2 EXCL      two signs of a mutual-exclusion pair in one text (S360 cut: both >= 25 texts, E >= 4, O <= 0.1 E)
  R3 W2        W2/W60 twice, W2 text-initial, or W2 with a marked jar 741/742/745 (S-DARK-15.2 R1/R3)
  R4 BIGRAM    an adjacent ordered pair never written in the fit set although expected >= 8 times (S-DARK-33.3, sign level)
  R5 ORDER     a sign pair written in the direction the fit set never uses (>= 10 co-occurrences, 0 reversals; S-DARK-19)
  R6 EDGE      a complete text that ends with a sign never final in the fit set, or starts with a sign never initial (>= 20 tokens)
  R7 FROZEN    both members of a frozen pair present but not adjacent in the fixed order (S229 + pairs adjacent >= 90% in fit)
  R8 FISH      fish qualifiers out of the hat -> whiskers -> bar -> stroke order (S296)
Score = number of rules broken.  Texts are read first-read-first (the CSV display string reversed, '000' dropped).

Usage: python3 tools/dark_loop42.py <cycle> [seq_raw|seq_strong|seq_all]
  cycle 1: build classes and rules; calibrate (5-fold CV false alarm, held-out sites, shuffles, chain forgers)
  cycle 2: score weak-provenance objects vs matched excavated ones (permutation null); top offenders
  cycle 3: loop 36 museum pieces, Umma tag, foreign pieces; mixture estimate with bootstrap CI
Sources: data/raw/inscriptions.csv (5,680 rows; S-DARK-23 caution), data/derived/sign_allographs_levels.json,
         data/derived/dark/loop36_newfinds.csv.
"""
import csv, json, sys, os, random, collections, math, bisect
ROOT = '/home/user/Indus-/'
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '1'
LEVEL = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
random.seed(42)

# ---------------------------------------------------------------- data
LEV = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
def merge_map(level):
    ok = {'seq_raw': set(), 'seq_strong': {'strong'}, 'seq_all': {'strong', 'probable'}}[level]
    m = {int(x['form']): int(x['into']) for x in LEV['merges'] if x['level'] in ok}
    def f(s):
        seen = 0
        while s in m and seen < 5: s = m[s]; seen += 1
        return s
    return f
MM = merge_map(LEVEL)

def parse_text(tx):
    """CSV display string -> reading order (first-read-first), unknown signs dropped; returns (seq, n_unknown)."""
    tx = (tx or '').strip()
    body = tx.strip('+[]')
    out = []; unk = 0
    for tok in body.split('-'):
        tok = tok.strip()
        if not tok: continue
        if '/' in tok: tok = tok.split('/')[0]
        try: v = int(tok)
        except ValueError: continue
        if v == 0: unk += 1; continue
        out.append(MM(v))
    out.reverse()
    return out, unk

FOREIGN = {'Ur', 'Tell Umma', 'Kish', 'Susa', 'Nippur', 'Tello', 'Girsu', 'Failaka', "Qala'at al-Bahrain", 'Saar',
           'Janabiyah', 'Karzakan', 'Hajar', 'Dilmun', 'Salut', "Ra's al-Junayz", 'Kalba', 'Luristan', 'Tepe Yahya',
           'Altyn Depe', 'Gonur Depe', 'Shortughai', 'Murda Sang', 'Hama'}
def prov_class(r):
    if r['site'] == 'Unknown': return 'UNKNOWN'
    if r['site'] in FOREIGN: return 'FOREIGN'
    if 'surface' in (r['depth'] or ''): return 'SURFACE'
    if r['excavation-idno'] != '-': return 'FULL'
    if r['area-section'] == '--' and r['depth'] == '- -': return 'NOCTX'
    return 'PART'
def tclass(ty):
    return 'SEAL' if ty.startswith('SEAL') else 'TAB' if ty.startswith('TAB') else 'POT' if ty.startswith('POT') else 'OTHER'

def load():
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    T = []
    for r in rows:
        seq, unk = parse_text(r['text'])
        if not seq: continue
        tx = (r['text'] or '').strip()
        complete = (r['complete'] == 'Y') and unk == 0 and tx.startswith('+') and tx.endswith('+')
        T.append(dict(id=r['id'], cisi=r['cisi'], site=r['site'], type=r['type'], tc=tclass(r['type']), seq=seq,
                      unk=unk, complete=complete, prov=prov_class(r), idno=r['excavation-idno'], area=r['area-section'],
                      depth=r['depth'], raw=tx))
    return T

# ---------------------------------------------------------------- rules
OPENERS = {817, 861, 820, 920, 692}
FISH = [235, 240, 233, 231]
S229 = [(590, 390), (590, 405), (435, 690), (255, 435), (840, 32), (17, 585), (3, 156), (33, 520), (32, 226)]

class Rules:
    def __init__(self, fit):
        seqs = [t['seq'] for t in fit]
        self.n = len(seqs)
        # R1: repeatable signs
        rep = collections.Counter()
        for s in seqs:
            for x in set(s):
                pos = [i for i, y in enumerate(s) if y == x]
                if len(pos) >= 2 and any(b - a > 1 for a, b in zip(pos, pos[1:])): rep[x] += 1
        self.repeatable = {x for x, c in rep.items() if c >= 3}
        # unigram presence counts
        pres = collections.Counter(x for s in seqs for x in set(s))
        self.pres = pres
        co = collections.Counter()
        for s in seqs:
            u = sorted(set(s))
            for i in range(len(u)):
                for j in range(i + 1, len(u)): co[(u[i], u[j])] += 1
        # R2: exclusion pairs
        big = [x for x, c in pres.items() if c >= 25]
        self.excl = set()
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                a, b = sorted((big[i], big[j]))
                E = pres[a] * pres[b] / self.n
                if E >= 4 and co[(a, b)] <= 0.1 * E: self.excl.add((a, b))
        # R4: forbidden adjacent bigrams
        left = collections.Counter(); right = collections.Counter(); bg = collections.Counter(); nb = 0
        for s in seqs:
            for a, b in zip(s, s[1:]): left[a] += 1; right[b] += 1; bg[(a, b)] += 1; nb += 1
        self.bg = bg
        self.forbid = set()
        for a in left:
            for b in right:
                E = left[a] * right[b] / nb
                if E >= 8 and bg[(a, b)] == 0: self.forbid.add((a, b))
        # R5: fixed-direction pairs (any distance), >= 10 co-occurring texts, 0 reversals
        dirc = collections.Counter()
        for s in seqs:
            first = {}
            for i, x in enumerate(s): first.setdefault(x, i)
            last = {}
            for i, x in enumerate(s): last[x] = i
            u = sorted(set(s))
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    a, b = u[i], u[j]
                    if last[a] < first[b]: dirc[(a, b)] += 1
                    elif last[b] < first[a]: dirc[(b, a)] += 1
        self.fixed = set()
        for (a, b), c in dirc.items():
            if c >= 10 and dirc[(b, a)] == 0: self.fixed.add((a, b))
        # R6: edge sets
        tok = collections.Counter(x for s in seqs for x in s)
        fin = collections.Counter(s[-1] for t, s in zip(fit, seqs) if t['complete'])
        ini = collections.Counter(s[0] for t, s in zip(fit, seqs) if t['complete'])
        self.never_final = {x for x, c in tok.items() if c >= 20 and fin[x] == 0}
        self.never_initial = {x for x, c in tok.items() if c >= 20 and ini[x] == 0}
        # R7: frozen pairs: adjacent a-b in >= 90% of the texts holding both, >= 10 such texts
        self.frozen = set(S229)
        for (a, b), c in dirc.items():
            tot = co[(min(a, b), max(a, b))]
            if tot >= 10 and bg[(a, b)] >= 0.9 * tot: self.frozen.add((a, b))
        self.frozen = {(MM(a), MM(b)) for a, b in self.frozen if MM(a) != MM(b)}

    def violations(self, s, complete=True):
        V = {}
        # R1
        bad = []
        for x in set(s):
            pos = [i for i, y in enumerate(s) if y == x]
            if len(pos) >= 2 and any(b - a > 1 for a, b in zip(pos, pos[1:])) and x not in self.repeatable: bad.append(x)
        if bad: V['R1_REPEAT'] = bad
        u = sorted(set(s))
        # R2
        e = [(a, b) for i, a in enumerate(u) for b in u[i + 1:] if (a, b) in self.excl]
        if e: V['R2_EXCL'] = e
        # R3
        w3 = []
        if s.count(2) >= 2: w3.append('W2x2')
        if s.count(60) >= 2: w3.append('W60x2')
        if s[0] == 2 and len(s) >= 2: w3.append('W2-initial')
        if 2 in s and ({741, 742, 745} & set(s)): w3.append('W2+marked-jar')
        if w3: V['R3_W2'] = w3
        # R4
        f = [(a, b) for a, b in zip(s, s[1:]) if (a, b) in self.forbid]
        if f: V['R4_BIGRAM'] = f
        # R5
        first = {}; last = {}
        for i, x in enumerate(s): first.setdefault(x, i); last[x] = i
        o = []
        for i, a in enumerate(u):
            for b in u[i + 1:]:
                if (a, b) in self.fixed and last[b] < first[a]: o.append((b, a))
                if (b, a) in self.fixed and last[a] < first[b]: o.append((a, b))
        if o: V['R5_ORDER'] = o
        # R6
        if complete and len(s) >= 2:
            ed = []
            if s[-1] in self.never_final: ed.append(('final', s[-1]))
            if s[0] in self.never_initial: ed.append(('initial', s[0]))
            if ed: V['R6_EDGE'] = ed
        # R7
        fr = []
        for a, b in self.frozen:
            if a in s and b in s and not any(x == a and y == b for x, y in zip(s, s[1:])): fr.append((a, b))
        if fr: V['R7_FROZEN'] = fr
        # R8
        fp = [(FISH.index(x), i) for i, x in enumerate(s) if x in FISH]
        fb = [(fp[a], fp[b]) for a in range(len(fp)) for b in range(a + 1, len(fp)) if fp[a][0] > fp[b][0]]
        if fb: V['R8_FISH'] = [(FISH[a[0]], FISH[b[0]]) for a, b in fb]
        return V

RULES = ['R1_REPEAT', 'R2_EXCL', 'R3_W2', 'R4_BIGRAM', 'R5_ORDER', 'R6_EDGE', 'R7_FROZEN', 'R8_FISH']

# ---------------------------------------------------------------- forger models
S_, E_ = -1, -2
class KN2:
    """order-2 chain over whole texts, absolute discounting with KN continuation backoff (loop15_engine)."""
    def __init__(self, seqs, D=0.75, order=2):
        self.D = D; self.order = order
        self.c2 = collections.defaultdict(collections.Counter); self.c1 = collections.defaultdict(collections.Counter)
        self.cont = collections.Counter()
        for s in seqs:
            s = [S_, S_] + list(s) + [E_]
            for i in range(2, len(s)):
                self.c2[(s[i-2], s[i-1])][s[i]] += 1; self.c1[(s[i-1],)][s[i]] += 1
        for h, cnt in self.c1.items():
            for v in cnt: self.cont[v] += 1
        self.vocab = sorted(self.cont); self.V = len(self.vocab); self.cache = {}
    def dist(self, h):
        ck = h if (self.order == 2 and h in self.c2) else ('u', h[1])
        if ck in self.cache: return self.cache[ck]
        tot = sum(self.cont.values())
        p = {v: (self.cont[v] + 0.5) / (tot + 0.5 * self.V) for v in self.vocab}
        ctxs = [(h[1],), h] if self.order == 2 else [(h[1],)]
        for ctx in ctxs:
            cnt = self.c1.get(ctx) if len(ctx) == 1 else self.c2.get(ctx)
            if not cnt: continue
            n = sum(cnt.values()); back = self.D * len(cnt) / n
            p = {v: max(cnt.get(v, 0) - self.D, 0) / n + back * p[v] for v in self.vocab}
        keys = list(p); acc = 0.0; cw = []
        for k in keys: acc += p[k]; cw.append(acc)
        self.cache[ck] = (keys, cw, acc); return self.cache[ck]
    def gen(self, rng, length=None, tries=0):
        out = []; h = (S_, S_)
        while len(out) < 16:
            k, cw, tot = self.dist(h)
            v = k[min(bisect.bisect_left(cw, rng.random() * tot), len(k) - 1)]
            if v == E_:
                if length is None or len(out) == length: break
                if len(out) < length: continue
                break
            if length is not None and len(out) >= length: break
            out.append(v); h = (h[1], v)
        if not out: return self.gen(rng, length, tries + 1)
        return out

def unigram_forger(seqs):
    toks = [x for s in seqs for x in s]
    def gen(rng, length): return [rng.choice(toks) for _ in range(length)]
    return gen

# ---------------------------------------------------------------- helpers
def score_set(R, texts):
    out = []
    for t in texts:
        V = R.violations(t['seq'], t['complete'])
        out.append((t, V))
    return out
def lbin(n): return '2' if n == 2 else '3' if n == 3 else '4' if n == 4 else '5' if n == 5 else '6+'
def rate_by_len(scored, thr=1):
    d = collections.defaultdict(lambda: [0, 0])
    for t, V in scored:
        b = lbin(len(t['seq'])); d[b][1] += 1; d[b][0] += (len(V) >= thr)
    return d
def fmt_rates(d):
    return '  '.join(f"{b}: {d[b][0]}/{d[b][1]} ({d[b][0]/max(1,d[b][1]):.3f})" for b in ['2', '3', '4', '5', '6+'] if b in d)
def shuffle_texts(texts, rng):
    out = []
    for t in texts:
        s = list(t['seq']); rng.shuffle(s)
        if len(s) >= 3 and s == t['seq']: rng.shuffle(s)
        out.append(dict(t, seq=s))
    return out
def forged_like(texts, gen, rng):
    return [dict(t, seq=gen(rng, len(t['seq'])), complete=True) for t in texts]
def rule_counts(scored):
    c = collections.Counter()
    for t, V in scored:
        for k in V: c[k] += 1
    return c
def out(path, lines):
    open(ROOT + 'data/derived/dark/' + path, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

T = load()
FIT = [t for t in T if t['site'] in ('Mohenjo-daro', 'Harappa') and t['prov'] == 'FULL']
HELD = [t for t in T if t['site'] not in ('Mohenjo-daro', 'Harappa') and t['prov'] == 'FULL']
def ge3(ts): return [t for t in ts if len(t['seq']) >= 3]

if __name__ == '__main__':
    if CYCLE == '1':
        L = [f'# loop 42 cycle 1  level={LEVEL}  (source: data/raw/inscriptions.csv, 5,680 rows; texts with >= 1 known sign: {len(T)})']
        c = collections.Counter((t['prov']) for t in T)
        L.append('provenance classes (texts): ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
        c3 = collections.Counter((t['prov']) for t in ge3(T))
        L.append('  texts >= 3 signs: ' + ', '.join(f'{k} {v}' for k, v in sorted(c3.items())))
        L.append('  how provenance is recorded: excavation-idno (register number; "-" = none), area-section ("--" = none), depth ("surface -" = surface find; "- -" = none), site "Unknown" = museum/dealer object (CISI "Un-" / "?-" numbers). No purchase/dealer field exists in the CSV; the Umma tag carries the Ashmolean number 1931.120 as its idno.')
        L.append(f'FIT set (MD+H with register number): {len(FIT)} texts, {len(ge3(FIT))} with >= 3 signs; held-out excavated sites with register number: {len(HELD)} / {len(ge3(HELD))}')
        R = Rules(FIT)
        L.append(f'rules fitted: repeatable signs {sorted(R.repeatable)}; exclusion pairs {len(R.excl)}; forbidden bigrams {len(R.forbid)}; fixed-direction pairs {len(R.fixed)}; never-final {sorted(R.never_final)}; never-initial {sorted(R.never_initial)}; frozen pairs {len(R.frozen)}')
        L.append('  forbidden bigrams: ' + ' '.join(f'{a}-{b}' for a, b in sorted(R.forbid)))
        L.append('  exclusion pairs: ' + ' '.join(f'{a}|{b}' for a, b in sorted(R.excl)))
        # in-sample rates (for reference)
        sc = score_set(R, FIT)
        L.append('in-sample FIT violation rate (>=1 rule) by length: ' + fmt_rates(rate_by_len(sc)))
        L.append('  rules broken in FIT (in-sample): ' + str(dict(rule_counts(sc))))
        # 5-fold CV
        rng = random.Random(1)
        idx = list(range(len(FIT))); rng.shuffle(idx)
        cv = []
        cvs = []
        for k in range(5):
            test = [FIT[i] for i in idx[k::5]]; train = [FIT[i] for j in range(5) if j != k for i in idx[j::5]]
            Rk = Rules(train)
            cv += score_set(Rk, test)
            cvs += score_set(Rk, shuffle_texts(test, rng))
        L.append('5-fold CV false alarm on excavated MD+H (>=1 rule): ' + fmt_rates(rate_by_len(cv)))
        L.append('   >=2 rules: ' + fmt_rates(rate_by_len(cv, 2)))
        L.append('   rules broken (CV): ' + str(dict(rule_counts(cv))))
        L.append('5-fold CV power on within-text SHUFFLES of the same texts (>=1): ' + fmt_rates(rate_by_len(cvs)))
        L.append('   >=2 rules: ' + fmt_rates(rate_by_len(cvs, 2)))
        # held-out sites
        sh = score_set(R, HELD)
        L.append('held-out excavated sites false alarm (>=1): ' + fmt_rates(rate_by_len(sh)))
        L.append('   >=2 rules: ' + fmt_rates(rate_by_len(sh, 2)))
        L.append('   rules broken (held-out): ' + str(dict(rule_counts(sh))))
        bysite = collections.defaultdict(lambda: [0, 0])
        for t, V in sh:
            if len(t['seq']) >= 3: bysite[t['site']][1] += 1; bysite[t['site']][0] += (len(V) >= 1)
        L.append('   by site (>= 3 signs): ' + ', '.join(f'{s} {a}/{b}' for s, (a, b) in sorted(bysite.items(), key=lambda x: -x[1][1]) if b >= 10))
        shs = score_set(R, shuffle_texts(HELD, rng))
        L.append('held-out SHUFFLE power (>=1): ' + fmt_rates(rate_by_len(shs)))
        # forgers, length-matched to held-out >= 3
        fitseqs = [t['seq'] for t in FIT]
        k2 = KN2(fitseqs, order=2); k1 = KN2(fitseqs, order=1); ug = unigram_forger(fitseqs)
        for name, g in [('order-2 chain (skilled forger)', k2.gen), ('order-1 chain', k1.gen), ('unigram sign-copier', ug)]:
            acc = []
            for rep in range(5):
                acc += score_set(R, forged_like(HELD, g, rng))
            L.append(f'{name} power (>=1): ' + fmt_rates(rate_by_len(acc)))
            L.append(f'   >=2 rules: ' + fmt_rates(rate_by_len(acc, 2)))
            L.append('   rules broken: ' + str(dict(rule_counts(acc))))
        # pooled >=3 summary
        def pooled(scored, thr=1):
            s3 = [(t, V) for t, V in scored if len(t['seq']) >= 3]
            return sum(len(V) >= thr for _, V in s3), len(s3)
        a, n = pooled(cv); b, m = pooled(sh); c1, n1 = pooled(cvs); d, n2 = pooled(shs)
        L.append(f'SUMMARY (>= 3 signs, >=1 rule): false alarm CV {a}/{n} = {a/n:.3f}; held-out sites {b}/{m} = {b/m:.3f}; shuffle sensitivity CV {c1}/{n1} = {c1/n1:.3f}, held-out {d}/{n2} = {d/n2:.3f}')
        f2 = score_set(R, forged_like(ge3(HELD) * 5, k2.gen, rng)); f1 = score_set(R, forged_like(ge3(HELD) * 5, k1.gen, rng)); f0 = score_set(R, forged_like(ge3(HELD) * 5, ug, rng))
        L.append(f'   forger sensitivity (>= 3 signs, >=1 rule): order-2 {sum(len(V)>=1 for _,V in f2)/len(f2):.3f}; order-1 {sum(len(V)>=1 for _,V in f1)/len(f1):.3f}; unigram {sum(len(V)>=1 for _,V in f0)/len(f0):.3f}')
        json.dump(dict(forbid=sorted(R.forbid), excl=sorted(R.excl), fixed=sorted(R.fixed), frozen=sorted(R.frozen),
                       never_final=sorted(R.never_final), never_initial=sorted(R.never_initial), repeatable=sorted(R.repeatable)),
                  open(ROOT + f'data/derived/dark/loop42_rules_{LEVEL}.json', 'w'))
        out(f'loop42_cycle1_{LEVEL}.txt', L)

    elif CYCLE == '2':
        L = [f'# loop 42 cycle 2  level={LEVEL}: weak-provenance objects vs matched excavated ones']
        R = Rules(FIT)
        rng = random.Random(2)
        ALL = score_set(R, T)
        byid = {t['id']: V for t, V in ALL}
        # population comparison, texts >= 3 signs
        def viol(t): return len(byid[t['id']]) >= 1
        pools = collections.defaultdict(list)
        for t in ge3(T): pools[(t['site'], t['tc'], len(t['seq']))].append(t)
        pools_notype = collections.defaultdict(list)
        for t in ge3(T):
            if t['prov'] == 'FULL': pools_notype[(t['tc'], min(len(t['seq']), 8))].append(t)
        def matched_test(group, label):
            """for each test text pick an excavated FULL text of the same site, object class and length (fallback: MD+H pooled, same class and length); permutation null on the pooled labels"""
            test = [t for t in group if len(t['seq']) >= 3]
            if not test: return f'{label}: no texts >= 3 signs'
            ctrl = []
            for t in test:
                cand = [c for c in pools[(t['site'], t['tc'], len(t['seq']))] if c['prov'] == 'FULL']
                if len(cand) < 3: cand = pools_notype[(t['tc'], min(len(t['seq']), 8))]
                if not cand: cand = [c for c in ge3(FIT)]
                ctrl.append(cand)
            obs_t = sum(viol(t) for t in test) / len(test)
            # control rate: average over 200 matched draws
            draws = []
            for _ in range(200):
                draws.append(sum(viol(rng.choice(c)) for c in ctrl) / len(test))
            ctrl_rate = sum(draws) / len(draws)
            # permutation: for each pair (test_i, matched_i) swap labels at random
            diffs = []
            for _ in range(2000):
                m = [rng.choice(c) for c in ctrl]
                d = 0
                for t, c in zip(test, m):
                    a, b = viol(t), viol(c)
                    if rng.random() < 0.5: a, b = b, a
                    d += a - b
                diffs.append(d / len(test))
            dobs = obs_t - ctrl_rate
            p = sum(1 for d in diffs if d >= dobs) / len(diffs)
            rc = rule_counts([(t, byid[t['id']]) for t in test])
            return f'{label}: n={len(test)} texts >= 3 signs; violation rate {obs_t:.3f} vs matched excavated {ctrl_rate:.3f} (diff {dobs:+.3f}, permutation P(>=) = {p:.3f}); rules: {dict(rc)}'
        for cls in ['UNKNOWN', 'SURFACE', 'NOCTX', 'PART', 'FOREIGN']:
            L.append(matched_test([t for t in T if t['prov'] == cls], cls))
        L.append(matched_test([t for t in T if t['prov'] == 'NOCTX' and t['site'] in ('Mohenjo-daro', 'Harappa')], 'NOCTX at MD+H only'))
        L.append(matched_test([t for t in T if t['prov'] == 'NOCTX' and t['tc'] == 'SEAL'], 'NOCTX seals'))
        L.append(matched_test([t for t in T if t['prov'] == 'UNKNOWN' and t['tc'] == 'SEAL' and t['type'] == 'SEAL:S'], 'UNKNOWN square seals'))
        L.append(matched_test([t for t in T if t['prov'] == 'FULL' and t['site'] not in ('Mohenjo-daro', 'Harappa')], 'held-out excavated (reference, matched to itself/MD+H)'))
        # by site for NOCTX
        for site in ['Dholavira', 'Mohenjo-daro', 'Harappa', 'Lakhanjo-daro', 'Rakhigarhi']:
            g = [t for t in T if t['prov'] == 'NOCTX' and t['site'] == site]
            if len(ge3(g)) >= 5: L.append('  ' + matched_test(g, f'NOCTX {site}'))
        # top offenders
        L.append('')
        L.append('objects with weak or no provenance, ranked by score (>= 1 rule), with the rule broken:')
        weak = [(t, byid[t['id']]) for t in T if t['prov'] in ('UNKNOWN', 'SURFACE', 'NOCTX', 'FOREIGN') and byid[t['id']]]
        weak.sort(key=lambda x: (-len(x[1]), -len(x[0]['seq'])))
        for t, V in weak[:60]:
            L.append(f"  {t['prov']:8s} {t['site']:14s} {t['cisi'] or t['id']:10s} {t['type']:8s} {'-'.join(map(str,t['seq'])):40s} score {len(V)}  " + '; '.join(f'{k}:{v}' for k, v in V.items()))
        L.append('')
        L.append('for comparison, the 20 excavated FIT texts with the highest score (in-sample):')
        fs = sorted([(t, byid[t['id']]) for t in FIT if byid[t['id']]], key=lambda x: (-len(x[1]), -len(x[0]['seq'])))
        for t, V in fs[:20]:
            L.append(f"  {t['prov']:8s} {t['site']:14s} {t['cisi'] or t['id']:10s} {t['type']:8s} {'-'.join(map(str,t['seq'])):40s} score {len(V)}  " + '; '.join(f'{k}:{v}' for k, v in V.items()))
        # the 28 Unknown texts in full
        L.append('')
        L.append('all site="Unknown" (museum/dealer) texts:')
        for t in T:
            if t['prov'] == 'UNKNOWN':
                V = byid[t['id']]
                L.append(f"  {t['id']:7s} {t['cisi']:7s} {t['type']:8s} {'-'.join(map(str,t['seq'])):40s} complete={t['complete']} score {len(V)}  " + '; '.join(f'{k}:{v}' for k, v in V.items()))
        out(f'loop42_cycle2_{LEVEL}.txt', L)

    elif CYCLE == '3':
        L = [f'# loop 42 cycle 3  level={LEVEL}: loop 36 museum pieces, historically doubtful objects, mixture estimate']
        R = Rules(FIT)
        rng = random.Random(3)
        # false alarm reference at each length from CV (recompute quickly) and held-out
        idx = list(range(len(FIT))); rng.shuffle(idx); cv = []
        for k in range(5):
            test = [FIT[i] for i in idx[k::5]]; train = [FIT[i] for j in range(5) if j != k for i in idx[j::5]]
            cv += score_set(Rules(train), test)
        fa = rate_by_len(cv); fah = rate_by_len(score_set(R, HELD))
        L.append('false-alarm reference (>=1 rule): CV ' + fmt_rates(fa) + ' | held-out sites ' + fmt_rates(fah))
        # loop 36 pieces
        rows = list(csv.DictReader(open(ROOT + 'data/derived/dark/loop36_newfinds.csv')))
        L.append('')
        L.append('loop 36 out-of-corpus pieces (strict = A/B-grade signs only; lenient = all read signs; ? = unread sign, splits the text for adjacency rules):')
        def parse36(row, strict):
            signs = row['signs_W'].split('-'); conf = (row['confidence_per_sign'] or '').split('-'); o = []
            for i, s in enumerate(signs):
                c = conf[i] if i < len(conf) else '?'
                if s in ('?', '') or '/' in s or not s.strip().isdigit(): o.append(None); continue
                if strict and c not in ('A', 'B'): o.append(None); continue
                o.append(MM(int(s)))
            return o
        def viol_gapped(seq_with_none):
            """rules on a text with unread positions: adjacency rules only across read neighbours; order/co-occurrence on all read signs"""
            conc = [x for x in seq_with_none if x is not None]
            if len(conc) < 2: return {}
            V = R.violations(conc, complete=False)
            # drop adjacency-based hits that straddle an unread sign
            adj = set(zip(conc, conc[1:]))
            real_adj = set((a, b) for a, b in zip(seq_with_none, seq_with_none[1:]) if a is not None and b is not None)
            if 'R4_BIGRAM' in V:
                V['R4_BIGRAM'] = [p for p in V['R4_BIGRAM'] if p in real_adj]
                if not V['R4_BIGRAM']: del V['R4_BIGRAM']
            if 'R7_FROZEN' in V and None in seq_with_none: del V['R7_FROZEN']   # cannot judge adjacency
            return V
        n36 = collections.Counter()
        for row in rows:
            if 'not transcribed' in row['signs_W'] or not row['signs_W']: continue
            for mode in ('strict', 'lenient'):
                s = parse36(row, mode == 'strict'); conc = [x for x in s if x is not None]
                if len(conc) < 2: continue
                V = viol_gapped(s)
                n36[(row['tier'][0], mode, len(V) >= 1)] += 1
                L.append(f"  {row['tier'][0]} {row['id']:16s} {mode:7s} {'-'.join('?' if x is None else str(x) for x in s):32s} n={len(conc)} score {len(V)}  " + '; '.join(f'{k}:{v}' for k, v in V.items()))
        L.append('  tier x mode flagged/total: ' + ', '.join(f'{k[0]}/{k[1]}: {n36[(k[0],k[1],True)]}/{n36[(k[0],k[1],True)]+n36[(k[0],k[1],False)]}' for k in sorted(set((a, b) for a, b, _ in n36))))
        # per-rule calibrated false-alarm rates (CV on MD+H, held-out sites), texts >= 3 signs
        def per_rule(scored):
            s3 = [(t, V) for t, V in scored if len(t['seq']) >= 3]
            return {r: sum(r in V for _, V in s3) / len(s3) for r in RULES}, len(s3)
        prc, nc = per_rule(cv); prh, nh = per_rule(score_set(R, HELD))
        L.append(f'  per-rule false alarm, texts >= 3 signs: CV (n={nc}) ' + ' '.join(f'{r}={prc[r]:.3f}' for r in RULES) + f' | held-out (n={nh}) ' + ' '.join(f'{r}={prh[r]:.3f}' for r in RULES))
        # group tests on the tier-B museum pieces: any rule, and R1 repeat alone, binomial against the calibrated rates
        def binom_ge(k, n, p):
            return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
        for mode in ('strict', 'lenient'):
            texts = []
            for row in rows:
                if not row['tier'].startswith('B') or 'not transcribed' in row['signs_W'] or not row['signs_W']: continue
                s = parse36(row, mode == 'strict'); conc = [x for x in s if x is not None]
                if len(conc) < 3: continue
                texts.append((row['id'], s, viol_gapped(s), len(conc)))
            if not texts: continue
            k_any = sum(len(V) >= 1 for _, _, V, _ in texts); k_r1 = sum('R1_REPEAT' in V for _, _, V, _ in texts)
            # matched-length false alarm from CV
            fa_any = sum(fa[lbin(n)][0] / fa[lbin(n)][1] for _, _, _, n in texts) / len(texts)
            cvl = collections.defaultdict(lambda: [0, 0])
            for t, V in cv:
                b = lbin(len(t['seq'])); cvl[b][1] += 1; cvl[b][0] += ('R1_REPEAT' in V)
            fa_r1 = sum(cvl[lbin(n)][0] / cvl[lbin(n)][1] for _, _, _, n in texts) / len(texts)
            unk = [t for t in T if t['prov'] == 'UNKNOWN' and len(t['seq']) >= 3]
            unk_r1 = sum('R1_REPEAT' in R.violations(t['seq'], t['complete']) for t in unk)
            L.append(f'  tier-B museum pieces ({mode}, >= 3 read signs): any rule {k_any}/{len(texts)} vs matched false alarm {fa_any:.3f} (binomial P(>=k) = {binom_ge(k_any, len(texts), fa_any):.3f}); R1 repeat {k_r1}/{len(texts)} vs CV R1 rate {fa_r1:.3f} (P = {binom_ge(k_r1, len(texts), fa_r1):.4f}); corpus Unknown-site texts R1 {unk_r1}/{len(unk)}')
        # historically doubtful corpus objects
        L.append('')
        L.append('historically doubtful corpus objects:')
        for t in T:
            if t['site'] in FOREIGN or t['prov'] == 'UNKNOWN':
                V = R.violations(t['seq'], t['complete'])
                L.append(f"  {t['prov']:8s} {t['site']:18s} {t['cisi'] or t['id']:10s} idno={t['idno']:12s} {t['type']:8s} {'-'.join(map(str,t['seq'])):36s} complete={t['complete']} score {len(V)}  " + '; '.join(f'{k}:{v}' for k, v in V.items()))
        # mixture model: p_u = (1-f) p_e + f p_f ; p_f from the order-2 forger at matched lengths; bootstrap CI
        L.append('')
        L.append('mixture estimate of the forgery share f (texts >= 3 signs): p_weak = (1-f) p_exc + f p_forger')
        fitseqs = [t['seq'] for t in FIT]; k2 = KN2(fitseqs, order=2)
        heldsc = {t['id']: len(R.violations(t['seq'], t['complete'])) >= 1 for t in HELD}
        byl = collections.defaultdict(list)
        for t in HELD:
            if len(t['seq']) >= 3: byl[min(len(t['seq']), 8)].append(heldsc[t['id']])
        cvl = collections.defaultdict(list)
        for t, V in cv:
            if len(t['seq']) >= 3: cvl[min(len(t['seq']), 8)].append(len(V) >= 1)
        def p_exc_at(lengths, ref):
            """matched-length false alarm: ref='held' = held-out excavated sites (CV where thin), ref='cv' = 5-fold CV on MD+H"""
            vals = []
            for n in lengths:
                b = min(n, 8); pool = byl[b] if (ref == 'held' and len(byl[b]) >= 20) else cvl[b]
                vals.append(sum(pool) / len(pool))
            return sum(vals) / len(vals)
        def p_forg_at(lengths, reps=20):
            tot = 0; n = 0
            for _ in range(reps):
                for ln in lengths:
                    tot += len(R.violations(k2.gen(rng, ln), True)) >= 1; n += 1
            return tot / n
        def clip(x): return min(1.0, max(0.0, x))
        groups = {cls: [t for t in T if t['prov'] == cls and len(t['seq']) >= 3] for cls in ['UNKNOWN', 'SURFACE', 'NOCTX', 'FOREIGN']}
        groups['ALL WEAK (Unknown+surface+no-context+foreign)'] = [t for t in T if t['prov'] in ('UNKNOWN', 'SURFACE', 'NOCTX', 'FOREIGN') and len(t['seq']) >= 3]
        for cls, g in groups.items():
            if len(g) < 5: L.append(f'  {cls}: n={len(g)} too few'); continue
            v = [len(R.violations(t['seq'], t['complete'])) >= 1 for t in g]; lens = [len(t['seq']) for t in g]
            pf = p_forg_at(lens); pu = sum(v) / len(v)
            for ref, nref in (('cv', 2400), ('held', 300)):
                pe = p_exc_at(lens, ref)
                f = (pu - pe) / (pf - pe) if pf > pe else float('nan')
                boots = []
                for _ in range(2000):
                    idxs = [rng.randrange(len(g)) for _ in g]
                    pub = sum(v[i] for i in idxs) / len(g)
                    peb = pe + rng.gauss(0, math.sqrt(pe * (1 - pe) / nref))   # binomial uncertainty of the reference
                    boots.append((pub - peb) / (pf - peb) if pf > peb else float('nan'))
                boots = sorted(b for b in boots if b == b)
                lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
                L.append(f'  {cls}: n={len(g)}, p_weak {pu:.3f}, p_exc({"held-out sites" if ref == "held" else "CV MD+H"}, matched length) {pe:.3f}, p_forger(order-2, matched length) {pf:.3f} -> f = {f:+.2f}, clipped {clip(f):.2f}  [95% bootstrap {clip(lo):.2f}, {clip(hi):.2f}] (raw interval {lo:+.2f}, {hi:+.2f})')
        out(f'loop42_cycle3_{LEVEL}.txt', L)
