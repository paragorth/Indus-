"""Loop 59 cycle 1: the predictive-model ladder. Fit on Mohenjo-daro + Harappa, score per-sign cross-entropy (bits)
and per-text perplexity on (a) held-out sites, (b) the 324 IM77-only texts, with bootstrap CIs, per slot.
Models: unigram (per object class), Markov-1 KN, Markov-2 KN, Markov-3 KN, the structural model (frame slots x shared
pool + rules + site cache + global cache + unigram, EM-interpolated), structural + KN2 (combined), the S366 generator
(Monte-Carlo estimate: KN-3 fitted on 60 synthetic corpora), the loop 33 automaton (class level + sign | class).
Usage: python3 tools/dark_loop59_c1.py <seq_raw|seq_strong|seq_all> [regime die|rows]
Outputs: data/derived/dark/loop59_c1_<level>[_rows].txt / .json
"""
import sys, json, math, random, collections, time, os
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop59_common import *

LV = sys.argv[1]; REG = sys.argv[2] if len(sys.argv) > 2 else 'die'
TAG = LV + ('' if REG == 'die' else '_' + REG)
OUT = DARK + f'loop59_c1_{TAG}'
logf = open(OUT + '.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); logf.write(s + '\n'); logf.flush()
t0 = time.time()

ALL = load_corpus(LV, REG)
FIT = [o for o in ALL if o['site'] in BIG]; HELD = [o for o in ALL if o['site'] not in BIG]
IM, unb, tot = load_im77_new(LV, FIT)
V = set(x for o in FIT for x in o['seq'])
QUAL = learn_qual([o['seq'] for o in FIT])
log(f'# loop 59 cycle 1, level {LV}, regime {REG} ({time.strftime("%Y-%m-%d %H:%M")}): fit {len(FIT)} MD+H texts '
    f'({sum(len(o["seq"]) for o in FIT)} signs, {len(V)} types); held-out sites {len(HELD)} texts '
    f'({sum(len(o["seq"]) for o in HELD)} signs); IM77-only {len(IM)} texts ({tot} signs, {unb} unbridged = UNK)')
oov_h = sum(1 for o in HELD for x in o['seq'] if x not in V); oov_i = sum(1 for o in IM for x in o['seq'] if x not in V)
log(f'OOV signs (scored as UNK): held-out {oov_h} ({oov_h/sum(len(o["seq"]) for o in HELD):.3f}), IM77 {oov_i} ({oov_i/tot:.3f})')

# ---------------------------------------------------------------- fit
uni_fn = lambda T: Unigram(T, V)
UNI = uni_fn(FIT)
KN1 = KN(FIT, 1, UNI); KN2 = KN(FIT, 2, UNI); KN3 = KN(FIT, 3, UNI)
log('fitting structural weights (5-fold deleted interpolation) ...')
W, obs = fit_weights(FIT, V, uni_fn)
STR = Structural(FIT, V, UNI, weights=W)
log(f'rule pairs in the fit set: {STR.rules.npairs} over {len(STR.rules.R)} left signs; '
    f'top: {sorted(((a, b, c) for a, d in STR.rules.R.items() for b, c in d.items()), key=lambda x: -x[2])[:15]}')
log('mixture weights by zone (frame / rules / cache_site / cache_all / unigram; all-active pattern):')
for z in ZONES:
    for key, w in sorted(W.items()):
        if key[0] == z and len(key[1]) == 5:
            log(f'  {z:5s} ' + ' '.join(f'{k}={w[k]:.2f}' for k in Structural.names))
for key, w in sorted(W.items()):
    if key[0] == '*': log(f'  pooled {key[1]}: ' + ' '.join(f'{k}={v:.2f}' for k, v in w.items()))

# structural ablations (components removed; weights refitted)
ABL = {}
for name, use in [('frame_only', ['frame', 'unigram']), ('frame+rules', ['frame', 'rules', 'unigram']),
                  ('frame+rules+cache', ['frame', 'rules', 'cache_site', 'cache_all', 'unigram'])]:
    w, _ = fit_weights(FIT, V, uni_fn, use=use); ABL[name] = Structural(FIT, V, UNI, weights=w, use=use)

# combined: structural + KN2 (+ KN3) by EM on folds
def fit_combo(folds=5, iters=30):
    rnd = random.Random(7); idx = list(range(len(FIT))); rnd.shuffle(idx); obsL = collections.defaultdict(list)
    for f in range(folds):
        train = [FIT[i] for j, i in enumerate(idx) if j % folds != f]; test = [FIT[i] for j, i in enumerate(idx) if j % folds == f]
        u = uni_fn(train); k2 = KN(train, 2, u); k3 = KN(train, 3, u)
        w, _ = fit_weights(train, V, uni_fn, folds=4); st = Structural(train, V, u, weights=w)
        for o in test:
            s = list(o['seq']) + [END]
            for i in range(len(s)):
                obsL[Frame.zone(s[:i])[0]].append((st.p(s[i], o, s[:i]), k2.p(s[i], o, s[:i]), k3.p(s[i], o, s[:i])))
            st.adapt(o)
    Wc = {}
    for z, L in obsL.items():
        w = [1 / 3] * 3
        for _ in range(iters):
            acc = [0.0] * 3
            for ps in L:
                tot = max(sum(wi * pi for wi, pi in zip(w, ps)), 1e-300)
                for k in range(3): acc[k] += w[k] * ps[k] / tot
            Z = sum(acc); w = [a / Z for a in acc]
        Wc[z] = w
    return Wc
WC = fit_combo()
log('combined weights (structural, KN2, KN3) by zone: ' + '; '.join(f'{z} {tuple(round(x, 2) for x in w)}' for z, w in WC.items()))
def combo_p(x, o, h):
    w = WC.get(Frame.zone(h)[0], [1 / 3] * 3)
    return w[0] * STR.p(x, o, h) + w[1] * KN2.p(x, o, h) + w[2] * KN3.p(x, o, h)

# ---------------------------------------------------------------- S366 generator as a predictive distribution (Monte Carlo)
S366 = None
try:
    import dark_loop54_common as L54
    SA, gen = L54.s366_generator(LV)
    def s366_fit(metas, R=60):
        rnd = random.Random(366); syn = []
        for r in range(R):
            g = gen.generate_corpus(metas, rnd)
            for (site, cls, s) in g: syn.append(dict(site=site, ot=cls, seq=tuple(x if (isinstance(x, int) and x > 0) else UNK for x in s)))
        u = Unigram(syn, V); return KN(syn, 3, u, site_token=True)
    S366 = {'held': s366_fit([(o['site'], o['ot']) for o in HELD]), 'im': s366_fit([(o['site'], o['ot']) for o in IM])}
    log(f'S366 generator sampled (60 corpora per evaluation set) and fitted as KN-3 with site + type tokens ({time.time()-t0:.0f}s)')
except Exception as e:
    log('S366 generator unavailable:', repr(e))

# ---------------------------------------------------------------- loop 33 automaton (class alphabet, all types + type symbol)
class PFA33:
    def __init__(self, path, fit):
        d = json.load(open(path)); self.trans = [{a: tuple(v) for a, v in t.items()} for t in d['trans']]
        self.final = d['final']; self.n = d['n']; self.A = d['alphabet']
        self.wsign = {int(a[1:]): a for a in self.A if a.startswith('w') and a[1:].isdigit()}
        enc = [['T:' + o['ot']] + [self.sym(x) for x in o['seq']] + ['#'] for o in fit]
        c = collections.Counter(x for s in enc for x in s); N = sum(c.values())
        self.uni = {a: c.get(a, 0.5) / N for a in set(self.A) | {'#'}}
        self.sc = collections.defaultdict(collections.Counter)
        for o in fit:
            for x in o['seq']: self.sc[self.sym(x)][x] += 1
    def sym(self, x):
        if x == UNK: return 'OTHER'
        if x in self.wsign: return self.wsign[x]
        c = sign_class(x)
        if c == 'TITLE' and x not in {255, 435, 690, 760, 100, 176, 923}: c = 'OTHER'
        if x in {630, 904, 482}: c = 'TABTITLE'
        if x == 900: c = 'BRACKET'
        if x == 368: c = 'CONN'
        if c in ('OPEN',): c = 'OPEN'
        return c if c in self.A else 'OTHER'
    def psym(self, q, a, beta=0.5):
        if q == -1: return self.uni.get(a, 1e-6)
        c = self.trans[q].get(a, (0, None))[0]
        return (c + beta * self.uni.get(a, 1e-6)) / (self.n[q] + beta)
    def nxt(self, q, a):
        if q == -1: return -1
        t = self.trans[q].get(a); return t[1] if t else -1
    def p(self, x, o, h):
        q = 0; q = self.nxt(q, 'T:' + o['ot'])
        for y in h: q = self.nxt(q, self.sym(y))
        if x == END:
            if q == -1: return self.uni.get('#', 1e-6)
            return (self.final[q] + 0.5 * self.uni.get('#', 1e-6)) / (self.n[q] + 0.5)
        a = self.sym(x); ps = self.psym(q, a)
        cnt = self.sc.get(a)
        if a == 'OTHER':
            d = UNI.dist(o['ot']); m = sum(p for y, p in d.items() if y != END and self.sym(y) == 'OTHER')
            return ps * UNI.p(x, o['ot']) / max(m, 1e-9)
        if not cnt: return ps * 1e-3
        N = sum(cnt.values()); T = len(cnt)
        return ps * (cnt.get(x, 0) + 0.5) / (N + 0.5 * T)
A33 = None
p33 = DARK + f'loop33_automaton_all+type_{LV}.json'
if os.path.exists(p33): A33 = PFA33(p33, FIT); log(f'loop 33 automaton loaded: {len(A33.trans)} states')

# ---------------------------------------------------------------- evaluate
def models_for(setname):
    M = {'uniform': lambda x, o, h: 1.0 / (len(V) + 2), 'unigram': lambda x, o, h: UNI.p(x, o['ot']), 'KN1': KN1.p, 'KN2': KN2.p, 'KN3': KN3.p,
         'frame_only': ABL['frame_only'].p, 'frame+rules': ABL['frame+rules'].p, 'frame+rules+cache': ABL['frame+rules+cache'].p,
         'structural': STR.p, 'combined': combo_p}
    if S366: M['S366gen'] = S366[setname].p
    if A33: M['loop33_PFA'] = A33.p
    return M
RESULTS = {}
for setname, T in [('held', HELD), ('im', IM)]:
    M = models_for(setname)
    adapt = [STR.adapt] + [a.adapt for a in ABL.values()]
    recs = evaluate(M, T, QUAL, adapt=adapt)
    # restore caches for the next set (remove adapted texts): rebuild
    STR.cs = Cache(FIT, True); STR.cg = Cache(FIT, False)
    for a in ABL.values(): a.cs = Cache(FIT, True); a.cg = Cache(FIT, False)
    ntok = len(recs); nsign = sum(1 for r in recs if r['x'] != END)
    log(f'\n## {setname}: {len(T)} texts, {nsign} signs + {len(T)} END tokens')
    log('| model | bits/token (95% CI) | bits/sign (no END) | per-text perplexity | gap closed vs unigram |')
    log('|---|---|---|---|---|')
    R = {}
    uni_ce = ce(recs, 'unigram')[0]
    for name in M:
        c, n = ce(recs, name); lo, hi = bootstrap_ce(recs, name, len(T))
        cs, _ = ce(recs, name, lambda r: r['x'] != END)
        ppl_text = 2 ** (sum(r['bits'][name] for r in recs) / len(T))
        R[name] = dict(bits=c, lo=lo, hi=hi, bits_sign=cs, text_ppl=ppl_text, gap=1 - c / uni_ce)
        log(f'| {name} | {c:.3f} [{lo:.3f}, {hi:.3f}] | {cs:.3f} | {ppl_text:.1f} | {100*(1-c/uni_ce):.1f}% |')
    # per-slot table
    slots = ['OPENER', 'MARKER', 'COUNT', 'NAME', 'TITLE', 'CLOSER', 'SUFFIX', 'END']
    log('\nper-slot bits (gap closed vs unigram in brackets); n tokens')
    log('| slot | n | ' + ' | '.join(M) + ' |'); log('|---|---|' + '---|' * len(M))
    R['slots'] = {}
    for sl in slots:
        sel = lambda r, sl=sl: r['slot'] == sl
        n = sum(1 for r in recs if sel(r))
        if n == 0: continue
        u = ce(recs, 'unigram', sel)[0]; row = []
        R['slots'][sl] = {'n': n}
        for name in M:
            c = ce(recs, name, sel)[0]; R['slots'][sl][name] = c; row.append(f'{c:.2f} [{100*(1-c/u):.0f}%]')
        log(f'| {sl} | {n} | ' + ' | '.join(row) + ' |')
    # share of total bits by slot under the best model
    best = min((k for k in M), key=lambda k: R[k]['bits'])
    R['best'] = best
    tot_bits = sum(r['bits'][best] for r in recs)
    share = collections.Counter()
    for r in recs: share[r['slot']] += r['bits'][best]
    log(f'\nbest model {best}: {tot_bits/len(T):.2f} bits per text; share of remaining bits by slot: ' +
        ', '.join(f'{k} {100*v/tot_bits:.0f}%' for k, v in share.most_common()))
    uni_bits = sum(r['bits']['unigram'] for r in recs)
    log(f'unigram {uni_bits/len(T):.2f} bits per text; explained share (1 - best/unigram) {100*(1-tot_bits/uni_bits):.1f}%')
    # by object type
    for ot in ('SEAL', 'TAB', 'OTHER'):
        sel = lambda r, ot=ot: r['ot'] == ot
        n = sum(1 for o in T if o['ot'] == ot)
        if n < 5: continue
        log(f'  {ot}: {n} texts; bits/token unigram {ce(recs, "unigram", sel)[0]:.3f}, KN2 {ce(recs, "KN2", sel)[0]:.3f}, '
            f'structural {ce(recs, "structural", sel)[0]:.3f}, combined {ce(recs, "combined", sel)[0]:.3f}')
    RESULTS[setname] = R
    RESULTS[setname + '_recs'] = [dict(t=r['t'], i=r['i'], slot=r['slot'], pc=r['pc'], zone=r['zone'], x=str(r['x']), prev=str(r['prev']), ot=r['ot'], site=r['site'], n=r['n'], bits=r['bits']) for r in recs]

# in-sample 5-fold on MD+H for reference (same ladder, cheaper: unigram, KN2, structural)
rnd = random.Random(11); idx = list(range(len(FIT))); rnd.shuffle(idx); cvb = collections.defaultdict(float); cvn = 0
for f in range(5):
    train = [FIT[i] for j, i in enumerate(idx) if j % 5 != f]; test = [FIT[i] for j, i in enumerate(idx) if j % 5 == f]
    u = uni_fn(train); k2 = KN(train, 2, u); w, _ = fit_weights(train, V, uni_fn, folds=4); st = Structural(train, V, u, weights=w)
    M = {'unigram': lambda x, o, h, u=u: u.p(x, o['ot']), 'KN2': k2.p, 'structural': st.p}
    recs = evaluate(M, test, QUAL, adapt=[st.adapt])
    for r in recs:
        for k, b in r['bits'].items(): cvb[k] += b
    cvn += len(recs)
log('\n## MD+H 5-fold cross-validation (bits/token): ' + ', '.join(f'{k} {v/cvn:.3f}' for k, v in cvb.items()))
RESULTS['cv_mdh'] = {k: v / cvn for k, v in cvb.items()}
RESULTS['meta'] = dict(level=LV, regime=REG, fit=len(FIT), held=len(HELD), im=len(IM), V=len(V), rules=STR.rules.npairs, weights={str(k): v for k, v in W.items()}, combo=WC)
json.dump(RESULTS, open(OUT + '.json', 'w'))
log(f'\ndone in {time.time()-t0:.0f}s')
