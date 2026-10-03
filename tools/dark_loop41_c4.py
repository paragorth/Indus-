"""Loop 41 cycle 4 (e) leakage: every fitted component (closer list, closer qualifier sets, fixed-pair directions,
W2 rules, closer-from-middle rule, name bigram model, W->M bridge, allograph merges) refitted on Mohenjo-daro + Harappa
ONLY and tested on (i) held-out sites absent from IM77 and (ii) the IM77 small sites (Lothal, Kalibangan, Chanhu-daro),
each against a null fitted the same way. Output: data/derived/dark/loop41_cycle4.txt"""
import sys, random, json, collections, statistics as st
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
from scipy.stats import binomtest
out = open(DARK + 'loop41_cycle4.txt', 'w')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical'); rnd = random.Random(414)
P('# Loop 41 cycle 4: leakage (refit on Mohenjo-daro + Harappa, test elsewhere)')

def split3(T):
    A, H = heldout_split(T)
    fit = [(r, s) for r, s in A if r['site'] in BIG]; im_small = [(r, s) for r, s in A if r['site'] not in BIG]
    return fit, im_small, H

def closer_fit(T, minn=15):
    texts = [s for _, s in T if len(s) >= 2]
    tok = collections.Counter(x for s in texts for x in s); fin = collections.Counter(); jw = collections.Counter(); je = collections.defaultdict(float)
    bylen = collections.defaultdict(list)
    for s in texts: bylen[len(s)].append(740 in s)
    jr = {L: sum(v) / len(v) for L, v in bylen.items()}
    for s in texts:
        t = list(s)
        while len(t) > 1 and t[-1] in SUF: t.pop()
        fin[t[-1]] += 1
        for x in set(s):
            if 740 in s: jw[x] += 1
            je[x] += jr[len(s)]
    return sorted(x for x, n in tok.items() if n >= minn and x != 740 and x not in SUF and fin[x] / n >= 0.4 and (jw[x] / je[x] if je[x] else 9) <= 0.5)

def last_core(s):
    t = list(s)
    while len(t) > 1 and t[-1] in SUF: t.pop()
    return t[-1]

def test_closers(cl, T, rnd, nn=200):
    """on test set: (1) share of >=2-sign texts whose final core sign is in cl, vs within-text shuffle; (2) finality of each
    fitted closer on the test set (tokens final / tokens); (3) exclusivity: texts with two cl signs vs tokens permuted within site x type."""
    tx = [s for _, s in T if len(s) >= 2]
    o = sum(last_core(s) in cl for s in tx) / len(tx)
    null = sorted(sum(last_core(tuple(rnd.sample(s, len(s)))) in cl for s in tx) / len(tx) for _ in range(nn))
    tok = sum(s.count(x) for s in tx for x in cl); fin = sum(last_core(s) in cl for s in tx)
    two = sum(1 for s in tx if len(set(s) & set(cl)) >= 2)
    (tw, mj), (etw, emj) = (two, 0), (0, 0)
    groups = collections.defaultdict(list)
    for i, (r, s) in enumerate(T): groups[(r['site'], otype(r['type']))].append(i)
    twos = []
    for _ in range(nn):
        TT = list(T)
        for idx in groups.values():
            pool = [x for i in idx for x in T[i][1]]; rnd.shuffle(pool); k = 0
            for i in idx:
                L = len(T[i][1]); TT[i] = (T[i][0], tuple(pool[k:k + L])); k += L
        twos.append(sum(1 for _, s in TT if len(s) >= 2 and len(set(s) & set(cl)) >= 2))
    return dict(n=len(tx), closer_last=o, null_med=null[nn // 2], null_max=null[-1], finality=fin / tok if tok else float('nan'), tokens=tok, two=two, two_E=st.mean(twos), two_P=(sum(1 for x in twos if x <= two) + 1) / (nn + 1))

for lvl in ('seq_raw', 'seq_all'):
    T = dedup(C, lvl, 'die'); fit, ims, H = split3(T)
    P(f'\n## {lvl}: fit MD+H n={len(fit)}; IM77 small sites n={len(ims)}; held-out (non-IM77) n={len(H)}')
    cl_fit = closer_fit(fit); cl_pub = CL
    cl_ims = closer_fit(ims, minn=8); cl_H = closer_fit(H, minn=8)
    P(f'  S289 closer list fitted on MD+H: {cl_fit}  (published {sorted(cl_pub)}; symmetric difference {sorted(set(cl_fit) ^ set(cl_pub))})')
    P(f'  closers that pass the same cuts (minn 8) fitted on IM77 small sites alone: {cl_ims}; on held-out alone: {cl_H}')
    for tname, TT in (('IM77-small', ims), ('held-out', H)):
        for cname, cl in (('MD+H-fitted', cl_fit), ('published', cl_pub)):
            d = test_closers(cl, TT, rnd); P(f'    {tname:10s} {cname:12s}: closer-last {d["closer_last"]:.3f} vs shuffle {d["null_med"]:.3f} (max {d["null_max"]:.3f}); finality of listed closers {d["finality"]:.2f} ({d["tokens"]} tokens); two-closer texts {d["two"]} vs {d["two_E"]:.1f} permuted (P {d["two_P"]:.3f})')
        # S303 qualifier overlap on the test set with the MD+H-fitted closers (minimum 8 texts per closer there)
        o, nl, p, k, nr = qualifier_overlap(TT, rnd, nnull=300, mint=8, closers=cl_fit)
        P(f'    {tname:10s} S303 qualifier overlap with MD+H-fitted closers: {o:.3f} vs permuted {nl:.3f} (P {p:.3f}; {k} closers with >= 8 texts, {nr} texts)')
    # ---- S-DARK-19 partial order: fit directions on MD+H, test on test sets; Markov-1 baseline also fitted on MD+H
    def fit_dirs(T, minco=5, alpha=0.05):
        ab = collections.Counter()
        for _, s in T:
            seen = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    if s[i] != s[j]: seen.add((s[i], s[j]))
            for p_ in seen: ab[p_] += 1
        done = set(); tests = []
        for (a, b), n in ab.items():
            if (b, a) in done: continue
            done.add((a, b)); m = ab[(b, a)]
            if n + m < minco: continue
            k = max(n, m); tests.append((binomtest(k, n + m, 0.5, alternative='greater').pvalue, (a, b) if n >= m else (b, a)))
        tests.sort(); M = len(tests); cut = 0
        for i, (p_, _) in enumerate(tests):
            if p_ <= alpha * (i + 1) / M: cut = i + 1
        return {d for _, d in tests[:cut]}
    dirs = fit_dirs(fit)
    def test_dirs(dirs, T):
        ok = bad = 0
        for _, s in T:
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    if (s[i], s[j]) in dirs: ok += 1
                    elif (s[j], s[i]) in dirs: bad += 1
        return ok, bad
    P(f'  S-DARK-19 fixed directions fitted on MD+H: {len(dirs)} pairs')
    m1 = markov_fit([s for _, s in fit], 1); uni = collections.Counter(x for _, s in fit for x in s)
    for tname, TT in (('IM77-small', ims), ('held-out', H)):
        ok, bad = test_dirs(dirs, TT)
        # Markov-1 baseline: generate test-set-shaped texts from the MD+H chain; how often do they obey the fitted directions?
        base = []
        for _ in range(40):
            G = [(r, markov_gen(m1, len(s), rnd, 1, uni)) for r, s in TT]; o2, b2 = test_dirs(dirs, G); base.append(o2 / (o2 + b2) if o2 + b2 else float('nan'))
        # and a frame-only baseline: within-text shuffle of the test set (destroys all order)
        P(f'    {tname:10s}: co-occurrences of fitted pairs {ok + bad}; obey fitted direction {ok / (ok + bad):.3f} (shuffle 0.50; Markov-1 fitted on MD+H {st.mean(base):.3f} [{min(base):.3f},{max(base):.3f}])')
    # ---- S-DARK-15 W2 rules on test sets (rules need no fitting; expectation from the test set's own strata)
    for tname, TT in (('IM77-small', ims), ('held-out', H)):
        (tw, mj), (etw, emj) = w2_rules(TT, rnd, nnull=200)
        n2 = sum(s.count(2) for _, s in TT)
        P(f'    {tname:10s} W2 tokens {n2}: W2 twice {tw} (E {etw:.1f}); W2 with marked jar {mj} (E {emj:.1f})')
    # ---- S366 closer depends on the middle's last sign: fit majority closer per last-middle-sign on MD+H, test accuracy elsewhere
    def recs(T):
        out = []
        for _, s in T:
            t = list(s)
            while len(t) > 1 and t[-1] in SUF: t.pop()
            if len(t) >= 2 and t[-1] in CL + [740]:
                out.append((t[-2], t[-1]))
        return out
    rf = recs(fit); maj = collections.defaultdict(collections.Counter)
    for a, c in rf: maj[a][c] += 1
    rule = {a: cnt.most_common(1)[0][0] for a, cnt in maj.items() if sum(cnt.values()) >= 3}
    base_closer = collections.Counter(c for _, c in rf).most_common(1)[0][0]
    for tname, TT in (('IM77-small', ims), ('held-out', H)):
        rt = [(a, c) for a, c in recs(TT) if a in rule]
        acc = sum(rule[a] == c for a, c in rt) / max(1, len(rt)); base = sum(c == base_closer for a, c in rt) / max(1, len(rt))
        # null: rule with closer labels permuted in the fit set
        nulls = []
        for _ in range(100):
            cs = [c for _, c in rf]; rnd.shuffle(cs); m2 = collections.defaultdict(collections.Counter)
            for (a, _), c in zip(rf, cs): m2[a][c] += 1
            r2 = {a: cnt.most_common(1)[0][0] for a, cnt in m2.items() if sum(cnt.values()) >= 3}
            nulls.append(sum(r2.get(a) == c for a, c in rt) / max(1, len(rt)))
        P(f'    {tname:10s} S366 closer-from-last-middle-sign: {len(rt)} test texts with a known left sign; accuracy {acc:.3f} vs majority-closer {base:.3f} vs permuted-fit rule {st.mean(nulls):.3f} [{min(nulls):.3f},{max(nulls):.3f}]')
    # ---- S321 name model fitted on MD+H middles only, tested on small-site seal middles
    msf = [m for m in (name_middle(s) for r, s in fit if r['type'].startswith('SEAL')) if m]
    for tname, TT in (('IM77-small', ims), ('held-out', H), ('all non-MD+H', ims + H)):
        mt = [m for m in (name_middle(s) for r, s in TT if r['type'].startswith('SEAL')) if m]
        if len(mt) < 40: continue
        b = bigram(msf); gens = [uniq([gen(b, len(m), rnd) for m in mt]) for _ in range(40)]
        bt = bigram(mt); gens_t = [uniq([gen(bt, len(m), rnd) for m in mt]) for _ in range(40)]
        u = uniq(mt)
        # name-like signature: share of test middles already seen in the fit set (a shared name stock would show this)
        shared = sum(1 for m in mt if m in set(msf)) / len(mt)
        gshared = st.mean(sum(1 for g in [gen(b, len(m), rnd) for m in mt] if g in set(msf)) / len(mt) for _ in range(40))
        P(f'    {tname:12s} S321 seal middles n={len(mt)}: unique {u:.3f}; bigram fitted on MD+H {st.mean(gens):.3f} (ratio {u/st.mean(gens):.2f}); bigram fitted on test {st.mean(gens_t):.3f} (ratio {u/st.mean(gens_t):.2f}); share of test middles also on an MD+H seal {shared:.3f} vs MD+H-bigram strings {gshared:.3f}')

# ---------------------------------------------------------------- bridge leakage (W->M from aligned pairs)
P('\n## Bridge: W->M mapping learned from MD+H aligned pairs only (loop 24 accepted pairs, substitution positions), tested on Lothal/Kalibangan/Chanhu-daro/Banawali pairs')
pairs = [p for p in json.load(open(DARK + 'loop24_pairs.json')) if p['accepted']]
bridge = {int(k): set(v) for k, v in json.load(open(ROOT + 'data/derived/bridge_extended.json')).items()}
props = {p['W']: p['M'] for p in json.load(open(DARK + 'bridge_proposals.json'))['proposals']}
def subs(ps):
    c = collections.defaultdict(collections.Counter)
    for p in ps:
        for op in p['ops']:
            if op[0] == 'S' and op[1] and op[2]: c[op[1]][op[2]] += 1
    return c
fitp = [p for p in pairs if p['wells']['site'] in BIG]; testp = [p for p in pairs if p['wells']['site'] not in BIG]
P(f'  pairs: MD+H {len(fitp)}, other sites {len(testp)} ({collections.Counter(p["wells"]["site"] for p in testp).most_common()})')
cf = subs(fitp); ct = subs(testp)
for kmin in (2, 3, 5):
    learned = {w: cnt.most_common(1)[0][0] for w, cnt in cf.items() if sum(cnt.values()) >= kmin and cnt.most_common(1)[0][1] / sum(cnt.values()) >= 0.6}
    tot = hit = 0; perW = []
    for w, cnt in ct.items():
        if w in learned:
            n = sum(cnt.values()); h = cnt[learned[w]]; tot += n; hit += h; perW.append((w, h, n))
    bad = sorted([x for x in perW if x[1] < x[2]], key=lambda x: x[2] - x[1], reverse=True)[:8]
    P(f'  learned on MD+H (k>={kmin}, consistency>=0.6): {len(learned)} W signs; on other-site aligned positions {hit}/{tot} = {hit / max(1, tot):.3f} agree; worst: {bad}')
# how do the published proposals fare outside MD+H?
tot = hit = 0; cov = 0; mism = []
for w, cnt in ct.items():
    if w in props:
        n = sum(cnt.values()); h = cnt[props[w]]; tot += n; hit += h; cov += 1
        if h < n: mism.append((w, props[w], dict(cnt)))
P(f'  S-DARK-27 proposals seen at other sites: {cov} W signs, {hit}/{tot} = {hit / max(1, tot):.3f} positions agree; mismatches {mism[:10]}')
tot = hit = 0
for w, cnt in ct.items():
    if w in bridge and bridge[w]:
        n = sum(cnt.values()); h = sum(v for m, v in cnt.items() if m in bridge[w]); tot += n; hit += h
P(f'  existing bridge_extended on other-site positions: {hit}/{tot} = {hit / max(1, tot):.3f} agree')
# the other direction: mappings learnable from other sites alone that MD+H contradicts
learned_t = {w: cnt.most_common(1)[0][0] for w, cnt in ct.items() if sum(cnt.values()) >= 3 and cnt.most_common(1)[0][1] / sum(cnt.values()) >= 0.6}
learned_f = {w: cnt.most_common(1)[0][0] for w, cnt in cf.items() if sum(cnt.values()) >= 3 and cnt.most_common(1)[0][1] / sum(cnt.values()) >= 0.6}
both = [w for w in learned_t if w in learned_f]; dis = [(w, learned_f[w], learned_t[w]) for w in both if learned_f[w] != learned_t[w]]
P(f'  W signs mappable from both halves: {len(both)}; disagreeing: {len(dis)} {dis}')

# ---------------------------------------------------------------- allograph merges: support outside MD+H
P('\n## Allograph merges (sign_allographs_levels.json): token counts and positional profile of form vs target outside MD+H')
lev = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
T = dedup(C, 'seq_raw', 'die'); fit, ims, H = split3(T); other = ims + H
def prof(x, T):
    tok = fin = ini = 0
    for _, s in T:
        for i, y in enumerate(s):
            if y == x: tok += 1; fin += (i == len(s) - 1); ini += (i == 0)
    return tok, fin / tok if tok else float('nan'), ini / tok if tok else float('nan')
for m in lev['merges']:
    if m['level'] not in ('strong', 'probable'): continue
    a, b = m['form'], m['into']
    tf, ff, i_f = prof(a, fit); tb, fb, ib = prof(b, fit); to, fo, io = prof(a, other); tbo, fbo, ibo = prof(b, other)
    P(f"  {m['level']:8s} {a}->{b}: MD+H tokens {tf}/{tb} final {fmt(ff)}/{fmt(fb)}; outside MD+H tokens {to}/{tbo} final {fmt(fo)}/{fmt(fbo)} initial {fmt(io)}/{fmt(ibo)}")
