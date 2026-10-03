"""S-DARK-66 cycle 1: class x class attraction matrix of NON-ADJACENT middle-element pairs under four classifications,
against (A) elements permuted among texts within site x type x middle-length (1,000x) and (B) Markov-2 chains of the
middle (fitted per site x type, lengths and positions kept, 300x); and where the beyond-chain attraction pairs of S366 /
S-DARK-41 fall in class space (concentrated in a few cells = semantic compatibility; scattered = idiosyncratic).
Usage: python3 tools/dark_loop66_c1.py <seq_raw|seq_strong|seq_all> [nperm] [nmarkov]
Outputs: data/derived/dark/loop66_c1_LEVEL.txt, loop66_classes_LEVEL.json, loop66_c1_LEVEL.json
"""
import sys, json, collections, random, math, time
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop66_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
NM = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rnd = random.Random(66); T0 = time.time()
out = [f'# S-DARK-66 cycle 1 ({LV}, perm {NP}, markov {NM}) {time.strftime("%Y-%m-%dT%H:%M")}']
def P(s): out.append(s); print(s, flush=True)

T = parse_all(load_wells(LV))
midtok = collections.Counter(x for t in T for x in t['mid'])
ELEM = sorted(w for w, c in midtok.items() if c >= 5)
P(f'texts {len(T)} (collapsed, complete, >= 2 signs); middle tokens {sum(midtok.values())}, types {len(midtok)}, elements >= 5 tokens {len(ELEM)} '
  f'covering {sum(midtok[w] for w in ELEM) / sum(midtok.values()):.2f} of tokens')

# ---------------------------------------------------------------- classifications
DESC, DTXT = desc_classes()
CLS = {'DESC': {w: c for w, c in DESC.items() if w in midtok},
       'WBLOCK': wblock_classes(ELEM),
       'GLYPH': glyph_classes(ELEM, 10),
       'DIST': dist_classes(T, ELEM, 10)}
json.dump({k: {str(w): c for w, c in v.items()} for k, v in CLS.items()} | {'desc_text': {str(w): DTXT[w] for w in DESC}},
          open(DARK + f'loop66_classes_{LV}.json', 'w'), indent=0)
for name, cls in CLS.items():
    cnt = collections.Counter(cls.values()); cov = sum(midtok[w] for w in cls) / sum(midtok.values())
    P(f'\n## {name}: {len(cls)} signs classed, {len(cnt)} classes, middle-token coverage {cov:.2f}')
    for c, n in cnt.most_common():
        mem = sorted([w for w in cls if cls[w] == c], key=lambda w: -midtok[w])
        P(f'   {c:10s} {n:3d} signs: ' + ' '.join(f'W{w}' for w in mem[:14]) + (' ...' if len(mem) > 14 else ''))
# agreement between classifications (adjusted MI over shared signs)
from sklearn.metrics import adjusted_mutual_info_score as ami
names = list(CLS)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        sh = [w for w in CLS[names[i]] if w in CLS[names[j]]]
        if len(sh) > 10:
            P(f'   AMI({names[i]}, {names[j]}) over {len(sh)} shared signs = {ami([CLS[names[i]][w] for w in sh], [CLS[names[j]][w] for w in sh]):.3f}')

# ---------------------------------------------------------------- observed pair sets
pairs_obs = [nonadj_pairs(t) for t in T]
npairs = sum(len(p) for p in pairs_obs)
P(f'\nnon-adjacent middle pairs (distance >= 2, unordered, per text): {npairs} in {sum(1 for p in pairs_obs if p)} texts')

def tables(pairs_per_text):
    return {name: class_table(pairs_per_text, cls, sorted(set(cls.values())))[0] for name, cls in CLS.items()}

def pairs_from_seqs(seqs):
    return [nonadj_pairs(dict(seq=s, midpos=t['midpos'])) for s, t in zip(seqs, T)]

OBS = tables(pairs_obs)
# null A: permutation within site x type x midlen
NULLA = {n: [] for n in CLS}
for r in range(NP):
    seqs = permute_middles(T, rnd)
    tb = tables(pairs_from_seqs(seqs))
    for n in CLS: NULLA[n].append(tb[n])
    if r == 0: P(f'   permutation 1 done at {time.time() - T0:.0f}s')
# null B: Markov-2 on the middle sequences (per site x type), lengths kept
def markov_mid(T, rnd, order=2):
    groups = collections.defaultdict(list)
    for t in T: groups[(t['site'], t['ot'])].append(t['mid'])
    models = {}
    for g, seqs in groups.items():
        models[g] = ({o: markov_fit(seqs, o) for o in range(1, order + 1)}, collections.Counter(x for s in seqs for x in s))
    new = []
    for t in T:
        backoff, uni = models[(t['site'], t['ot'])]
        s = list(t['seq'])
        if t['mid']:
            gen = markov_gen(None, len(t['mid']), rnd, order, uni, backoff)
            for pos, x in zip(t['midpos'], gen): s[pos] = x
        new.append(s)
    return new
NULLB = {n: [] for n in CLS}; NULLM1 = {n: [] for n in CLS}
for r in range(NM):
    seqs = markov_mid(T, rnd, 2)
    tb = tables(pairs_from_seqs(seqs))
    for n in CLS: NULLB[n].append(tb[n])
    seqs = markov_mid(T, rnd, 1)
    tb = tables(pairs_from_seqs(seqs))
    for n in CLS: NULLM1[n].append(tb[n])
P(f'   nulls done at {time.time() - T0:.0f}s')
NUMCLASS = {'DESC': 'numeral', 'WBLOCK': 'num'}

RES = {}
for name, cls in CLS.items():
    classes = sorted(set(cls.values())); k = len(classes)
    O = OBS[name]; A = np.array(NULLA[name]); B = np.array(NULLB[name])
    EA = A.mean(0); EB = B.mean(0); sdA = A.std(0) + 1e-9; sdB = B.std(0) + 1e-9
    iu = np.triu_indices(k)
    def chi(Ot, E):
        m = E >= 2
        return float((((Ot - E) ** 2 / np.where(m, E, 1))[m]).sum())
    chiA = chi(O, EA); chiA_null = [chi(a, EA) for a in A]
    chiB = chi(O, EB); chiB_null = [chi(b, EB) for b in B]
    M1 = np.array(NULLM1[name]); EM1 = M1.mean(0); chiM1 = chi(O, EM1); chiM1_null = [chi(m, EM1) for m in M1]
    # G statistic of class table vs expected (information in class pairing)
    def G(Ot, E):
        m = (E >= 2) & (Ot > 0)
        return float(2 * (Ot[m] * np.log(Ot[m] / E[m])).sum())
    GA = G(O, EA); GA_null = [G(a, EA) for a in A]
    diag = float(np.trace(O)); diagA = [float(np.trace(a)) for a in A]; diagB = [float(np.trace(b)) for b in B]
    tot = O[iu].sum()
    P(f'\n## {name}: {int(tot)} classed pairs in {k} classes')
    P(f'   chi2 vs permutation: {chiA:.0f} vs null {np.mean(chiA_null):.0f} [{np.percentile(chiA_null, 2.5):.0f}, {np.percentile(chiA_null, 97.5):.0f}] P = {pval(chiA, chiA_null):.3f}; '
      f'G {GA:.0f} vs {np.mean(GA_null):.0f} P = {pval(GA, GA_null):.3f}')
    P(f'   chi2 vs Markov-1 middle: {chiM1:.0f} vs null {np.mean(chiM1_null):.0f} [{np.percentile(chiM1_null, 2.5):.0f}, {np.percentile(chiM1_null, 97.5):.0f}] P = {pval(chiM1, chiM1_null):.3f}')
    P(f'   chi2 vs Markov-2 middle: {chiB:.0f} vs null {np.mean(chiB_null):.0f} [{np.percentile(chiB_null, 2.5):.0f}, {np.percentile(chiB_null, 97.5):.0f}] P = {pval(chiB, chiB_null):.3f}')
    if name in NUMCLASS and NUMCLASS[name] in classes:
        ni = classes.index(NUMCLASS[name]); keep = [i for i in range(k) if i != ni]
        d2 = float(sum(O[i, i] for i in keep)); t2 = float(sum(O[i, j] for i in keep for j in keep if i <= j))
        d2A = [float(sum(a[i, i] for i in keep)) for a in A]; d2B = [float(sum(b[i, i] for i in keep)) for b in B]
        P(f'   same-class pairs WITHOUT the numeral class: {d2:.0f} ({d2 / t2:.3f}) vs permutation {np.mean(d2A):.1f} ({np.mean(d2A) / t2:.3f}) P(hi) = {pval(d2, d2A):.3f} P(lo) = {pval(d2, d2A, "lo"):.3f}; vs Markov-2 {np.mean(d2B):.1f} P(lo) = {pval(d2, d2B, "lo"):.3f}')
    P(f'   same-class pairs: {diag:.0f} ({diag / tot:.3f}) vs permutation {np.mean(diagA):.1f} ({np.mean(diagA) / tot:.3f}) P(hi) = {pval(diag, diagA):.3f} P(lo) = {pval(diag, diagA, "lo"):.3f}; '
      f'vs Markov-2 {np.mean(diagB):.1f} P(hi) = {pval(diag, diagB):.3f} P(lo) = {pval(diag, diagB, "lo"):.3f}')
    # cell-level
    zA = (O - EA) / sdA; zB = (O - EB) / sdB
    cells = []
    for i, j in zip(*iu):
        if EA[i, j] >= 2 or O[i, j] >= 5:
            cells.append((classes[i], classes[j], int(O[i, j]), float(EA[i, j]), float(zA[i, j]), float(EB[i, j]), float(zB[i, j])))
    ncell = len(cells); nsig = sum(1 for c in cells if abs(c[4]) > 3); nsigB = sum(1 for c in cells if abs(c[6]) > 3)
    # how many cells would be |z|>3 under the null itself (false alarm)
    fa = []
    for a in A[:200]:
        za = (a - EA) / sdA; fa.append(sum(1 for i, j in zip(*iu) if (EA[i, j] >= 2 or a[i, j] >= 5) and abs(za[i, j]) > 3))
    P(f'   cells tested {ncell}; |z| > 3 vs permutation {nsig} (null itself {np.mean(fa):.1f}), vs Markov-2 {nsigB}')
    cells.sort(key=lambda c: -abs(c[4]))
    for c in cells[:12]:
        P(f'      {c[0]:>10s} x {c[1]:<10s} O {c[2]:4d}  E_perm {c[3]:6.1f} z {c[4]:+5.1f}   E_mk2 {c[5]:6.1f} z {c[6]:+5.1f}  log2 O/E {math.log2((c[2] + 0.5) / (c[3] + 0.5)):+.2f}')
    RES[name] = dict(classes=classes, O=O.tolist(), EA=EA.tolist(), sdA=sdA.tolist(), EB=EB.tolist(), sdB=sdB.tolist(),
                     chiA=chiA, chiA_P=pval(chiA, chiA_null), chiM1=chiM1, chiM1_P=pval(chiM1, chiM1_null), chiB=chiB, chiB_P=pval(chiB, chiB_null), GA_P=pval(GA, GA_null),
                     diag=diag, diag_share=diag / tot, diagA=float(np.mean(diagA)), diagB=float(np.mean(diagB)),
                     diag_Phi=pval(diag, diagA), diag_Plo=pval(diag, diagA, 'lo'), ncell=ncell, nsig=nsig, nsigB=nsigB, fa=float(np.mean(fa)))

# ---------------------------------------------------------------- beyond-chain pairs (S366 / S-DARK-41 statistic)
P('\n## Beyond-chain attraction pairs (S366 statistic: distance >= 2, count >= 5, O/E >= 3) at Mohenjo-daro + Harappa, whole texts')
TD = load_wells_die(LV); TB = [t for t in TD if t['site'] in BIG]
P(f'   die regime (S-DARK-41): {len(TD)} texts, MD+H {len(TB)} (collapsed MD+H would be {sum(1 for t in T if t["site"] in BIG)})')
att, both, has = attraction_pairs([t['seq'] for t in TB], all_nonadj_pairs)
attC, _, _ = attraction_pairs([t['seq'] for t in T if t['site'] in BIG], all_nonadj_pairs)
P(f'   attraction pairs on the collapsed corpus: {len(attC)}')
# Markov-2 whole-text null: count distribution of each pair
mk_counts = collections.defaultdict(list)
for r in range(NM):
    seqs = markov_corpus(TB, rnd, 2)
    _, b2, _ = attraction_pairs(seqs, all_nonadj_pairs)
    for p in att: mk_counts[p].append(b2.get(p, 0))
natt_mk = []
for r in range(min(NM, 100)):
    seqs = markov_corpus(TB, rnd, 2); a2, _, _ = attraction_pairs(seqs, all_nonadj_pairs); natt_mk.append(len(a2))
beyond = {p: v for p, v in att.items() if v[0] > np.percentile(mk_counts[p], 97.5)}
P(f'   attraction pairs {len(att)} (Markov-2 corpora give {np.mean(natt_mk):.1f} [{min(natt_mk)}, {max(natt_mk)}]); beyond Markov-2 (count > 97.5th pct of {NM} chains): {len(beyond)}')
midset = set(ELEM)
fr_beyond = {p: v for p, v in beyond.items() if p[0] in FRAME or p[1] in FRAME}
mid_beyond = {p: v for p, v in beyond.items() if p not in fr_beyond and p[0] in midset and p[1] in midset}
P(f'   of these, involving a frame sign (opener/marker/closer/suffix): {len(fr_beyond)}; both signs middle elements: {len(mid_beyond)}; other (rare signs): {len(beyond) - len(mid_beyond) - len(fr_beyond)}')
P('   beyond-chain pairs: ' + ', '.join(f'{a}-{b} ({c}:{e:.1f})' for (a, b), (c, e) in sorted(beyond.items(), key=lambda kv: -kv[1][0])))
# concentration in class cells
def concentration(pairset, cls, nullpool, nullw, nrep=1000):
    """pairs mapped to class cells; share in top-3 cells and normalised entropy; null = same number of pairs drawn from
    the tested-pair pool with probability proportional to expected co-occurrence (frequency-matched scattering)"""
    def cellstats(ps):
        cnt = collections.Counter()
        for a, b in ps:
            ca, cb = cls.get(a), cls.get(b)
            if ca is None or cb is None: continue
            cnt[tuple(sorted((ca, cb)))] += 1
        n = sum(cnt.values())
        if n == 0: return float('nan'), float('nan'), 0, cnt
        v = sorted(cnt.values(), reverse=True)
        top3 = sum(v[:3]) / n
        H = -sum(x / n * math.log2(x / n) for x in v); Hn = H / math.log2(len(v)) if len(v) > 1 else 0.0
        return top3, Hn, n, cnt
    top3, Hn, n, cnt = cellstats(pairset)
    if n < 3: return None
    nulls_t = []; nulls_h = []
    w = np.array(nullw, float); w /= w.sum()
    for _ in range(nrep):
        pick = [nullpool[i] for i in np.random.default_rng(_).choice(len(nullpool), size=len(pairset), replace=False, p=w)]
        t3, h, m, _c = cellstats(pick)
        if m >= 3: nulls_t.append(t3); nulls_h.append(h)
    return dict(n=n, top3=top3, top3_null=float(np.mean(nulls_t)), top3_P=pval(top3, nulls_t), Hn=Hn, Hn_null=float(np.mean(nulls_h)), Hn_P=pval(Hn, nulls_h, 'lo'),
                cells=cnt.most_common(6))
# null pool: all tested pairs (both >= 5 tokens) with expected co-occurrence as weight
pool = [p for p in both if has[p[0]] >= 5 and has[p[1]] >= 5]
poolw = [has[p[0]] * has[p[1]] / len(TB) for p in pool]
CONC = {}
for name, cls in CLS.items():
    for label, ps in (('all attraction', list(att)), ('beyond Markov-2', list(beyond)), ('beyond, middle only', list(mid_beyond))):
        pp = [p for p in pool if cls.get(p[0]) is not None and cls.get(p[1]) is not None]
        pw = [has[p[0]] * has[p[1]] / len(TB) for p in pp]
        r = concentration(ps, cls, pp, pw)
        if r is None: P(f'   {name:7s} {label:20s}: too few classed pairs'); continue
        CONC[(name, label)] = r
        P(f'   {name:7s} {label:20s}: n {r["n"]:3d} classed; top-3 cells {r["top3"]:.2f} vs scattered {r["top3_null"]:.2f} (P {r["top3_P"]:.3f}); '
          f'H/Hmax {r["Hn"]:.2f} vs {r["Hn_null"]:.2f} (P {r["Hn_P"]:.3f}); cells ' + ', '.join(f'{a}x{b} {n}' for (a, b), n in r['cells']))

json.dump(dict(level=LV, nperm=NP, nmarkov=NM, n_texts=len(T), n_pairs=npairs, res=RES,
               attraction=len(att), attraction_mk=[float(np.mean(natt_mk)), min(natt_mk), max(natt_mk)], beyond=[[list(p), v] for p, v in beyond.items()],
               conc={f'{a}|{b}': v for (a, b), v in CONC.items()}), open(DARK + f'loop66_c1_{LV}.json', 'w'))
open(DARK + f'loop66_c1_{LV}.txt', 'w').write('\n'.join(out) + '\n')
P(f'done {time.time() - T0:.0f}s')
