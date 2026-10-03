"""Loop 46 cycle 1: do hands exist? Variant co-selection across sign pairs within texts.

For every pair of merged signs (classes with >= 2 variant forms of >= 20 tokens) that co-occur in >= MINPAIR texts,
the 2x2 table of (form of A is non-head, form of B is non-head) gives MI and a log-odds ratio. Global statistics:
G = sum over pairs of N_pair * MI (bits), and the number of pairs whose |log-odds| exceeds its own null 97.5th pct.
Nulls (1,000x): variant labels of each class permuted among texts within
  N1 site x type  (city dialect + medium)        N2 site x type x period   N3 site x type x area-section
Also: within-text consistency (texts carrying two tokens of one class: same form or not) vs the same null.
Run at merge levels strong / all / ext, whole set and the transcription-robust subset {390, 803, 156}.
Usage: python3 tools/dark_loop46_c1.py [nperm]
"""
import sys, json, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop46_common import *

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DEDUP = len(sys.argv) > 2 and sys.argv[2] == 'dedup'
SUFFIX = '_dedup' if DEDUP else ''
MINPAIR = 20
MINPAIR_ROBUST = 10
rng = np.random.default_rng(46)
corpus = load_corpus()
if DEDUP:
    seen = set(); kept = []
    for r in corpus:
        key = (tuple(r['seq_all']), r['site'])
        if key not in seen:
            seen.add(key); kept.append(r)
    corpus = kept
N = len(corpus)
lines = [f'# loop 46 cycle 1: variant co-selection across sign pairs (hands?)  nperm={NPERM}  dedup={DEDUP}  texts={N}  file=merged-corpus-canonical.json (S-DARK-23 caution)']
summary = {}

NULLS = {'N1 site x type': ['site', 'type'],
         'N2 site x type x period': ['site', 'type', 'period'],
         'N3 site x type x area': ['site', 'type', 'area-section']}


def run(level, heads, minpair, tag):
    classes = load_classes(level)
    use = usable_classes(corpus, classes, heads=heads)
    labels, multi, incons = text_labels(corpus, classes, use)
    heads_used = sorted(use)
    lines.append(f'\n## level={level} subset={tag}  usable classes: ' + ', '.join(
        f"W{h}[{'/'.join(str(f)+'x'+str(use[h]['counts'][f]) for f in use[h]['forms'])}]" for h in heads_used))
    # per-class arrays
    cls_idx = {h: np.array([i for i in range(N) if h in labels[i]]) for h in heads_used}
    cls_lab = {h: np.array([labels[i][h] for i in cls_idx[h]]) for h in heads_used}
    # pairs
    pairs = []
    for a, b in itertools.combinations(heads_used, 2):
        sa = set(cls_idx[a].tolist()); shared = sorted(sa & set(cls_idx[b].tolist()))
        if len(shared) >= minpair:
            pa = {t: k for k, t in enumerate(cls_idx[a].tolist())}
            pb = {t: k for k, t in enumerate(cls_idx[b].tolist())}
            pairs.append((a, b, np.array([pa[t] for t in shared]), np.array([pb[t] for t in shared])))
    lines.append(f'pairs with >= {minpair} shared texts: {len(pairs)}  (texts with >= 2 usable classes: '
                 f'{sum(1 for l in labels if len(l) >= 2)})')
    if not pairs:
        lines.append('no testable pairs')
        return
    # observed
    def stats(lab):
        g = 0.0; lo = []; mis = []
        for a, b, ia, ib in pairs:
            x = lab[a][ia]; y = lab[b][ib]
            m = mi_bits(x, y); g += len(x) * m; lo.append(log_odds(x, y)); mis.append(m)
        return g, np.array(lo), np.array(mis)
    G, LO, MIs = stats(cls_lab)
    res = {}
    for nname, keys in NULLS.items():
        st = strata(corpus, keys)
        st_codes = {h: np.array([hash(st[i]) for i in cls_idx[h]]) for h in heads_used}
        nullG = np.zeros(NPERM); nullLO = np.zeros((NPERM, len(pairs)))
        for p in range(NPERM):
            lab = {h: permute_within(cls_lab[h], st_codes[h], rng) for h in heads_used}
            g, lo, _ = stats(lab)
            nullG[p] = g; nullLO[p] = lo
        pG = (np.sum(nullG >= G) + 1) / (NPERM + 1)
        z = (G - nullG.mean()) / (nullG.std() + 1e-12)
        # per-pair two-sided p
        pp = np.array([(np.sum(np.abs(nullLO[:, k]) >= abs(LO[k])) + 1) / (NPERM + 1) for k in range(len(pairs))])
        nsig = int(np.sum(pp < 0.05)); exp_sig = 0.05 * len(pairs)
        # count of significant pairs vs null count (how many pairs would pass under the null)
        null_nsig = np.array([np.sum(np.abs(nullLO[q]) >= np.quantile(np.abs(nullLO), 0.95, axis=0)) for q in range(NPERM)])
        p_nsig = (np.sum(null_nsig >= nsig) + 1) / (NPERM + 1)
        res[nname] = dict(G=G, nullG_mean=float(nullG.mean()), nullG_sd=float(nullG.std()), z=float(z), P=float(pG),
                          nsig=nsig, exp_sig=exp_sig, p_nsig=float(p_nsig), perpair_p=pp.tolist())
        lines.append(f'{nname}: G={G:.2f} null {nullG.mean():.2f}+/-{nullG.std():.2f} z={z:+.2f} P={pG:.3f}; '
                     f'pairs |log-odds| beyond own null 95%: {nsig}/{len(pairs)} (exp {exp_sig:.1f}, P={p_nsig:.3f})')
    # per pair table
    lines.append('pair table: A x B  n  n11/n10/n01/n00  log-odds  MI  p(N1) p(N2) p(N3)')
    for k, (a, b, ia, ib) in enumerate(pairs):
        x = cls_lab[a][ia]; y = cls_lab[b][ib]
        n11 = int(np.sum((x == 1) & (y == 1))); n10 = int(np.sum((x == 1) & (y == 0)))
        n01 = int(np.sum((x == 0) & (y == 1))); n00 = int(np.sum((x == 0) & (y == 0)))
        ps = ' '.join(f"{res[n]['perpair_p'][k]:.3f}" for n in NULLS)
        lines.append(f'  W{a} x W{b}  n={len(x)}  {n11}/{n10}/{n01}/{n00}  LO={LO[k]:+.2f}  MI={MIs[k]:.3f}  {ps}')
    # within-text consistency
    same = sum(1 for (_, h, fs) in multi if len(set(fs)) == 1); tot = len(multi)
    # null: for each multi-token text, draw forms independently from the class's form distribution within site x type
    st = strata(corpus, ['site', 'type'])
    pool = collections.defaultdict(list)
    vt = variant_tokens(corpus, classes)
    for i, toks in enumerate(vt):
        for h, f in toks:
            if h in use:
                pool[(h, st[i])].append(f)
    exp_same = 0.0
    for (i, h, fs) in multi:
        pl = pool[(h, st[i])]
        freq = collections.Counter(pl)
        # probability two random tokens have the same form (with replacement), extended to len(fs) tokens all same
        tot_n = len(pl); p_all = sum((c / tot_n) ** len(fs) for c in freq.values())
        exp_same += p_all
    lines.append(f'within-text consistency: {tot} texts carry >= 2 tokens of one class; all the same form in {same} '
                 f'(expected {exp_same:.1f} if forms were drawn independently from the site x type form mix); '
                 f'inconsistent texts: ' + '; '.join(f"{corpus[i]['cisi']} W{h} {fs}" for (i, h, fs) in incons[:12]))
    summary[f'{level}/{tag}'] = dict(classes=heads_used, npairs=len(pairs), nulls={k: {kk: vv for kk, vv in v.items() if kk != 'perpair_p'} for k, v in res.items()},
                                     consistency=dict(multi=tot, same=same, expected=exp_same))


for level in ['strong', 'all', 'ext']:
    run(level, None, MINPAIR, 'all-classes')
    run(level, ROBUST_HEADS, MINPAIR_ROBUST, 'robust')

# --- breakdown of the one suggestive pair, W390 x W803, by site and type (seq_raw forms)
classes = load_classes('all')
vt = variant_tokens(corpus, classes)
tab = collections.Counter()
for i, toks in enumerate(vt):
    f390 = [f for h, f in toks if h == 390]; f803 = [f for h, f in toks if h == 803]
    if f390 and f803:
        tab[(corpus[i]['site'], corpus[i]['type'].split(':')[0], f390[0], f803[0])] += 1
lines.append('\nW390 x W803 co-occurring texts by site, type, forms: ' + '; '.join(f'{k[0]}/{k[1]} {k[2]}+{k[3]} x{v}' for k, v in sorted(tab.items())))

# --- power control: plant two hands per site x type stratum and see whether G detects them (level=all, N1 null)
NPLANT = 30; PPERM = 200
classes = load_classes('all'); use = usable_classes(corpus, classes)
labels, _, _ = text_labels(corpus, classes, use)
heads_used = sorted(use)
cls_idx = {h: np.array([i for i in range(N) if h in labels[i]]) for h in heads_used}
cls_lab = {h: np.array([labels[i][h] for i in cls_idx[h]]) for h in heads_used}
pairs = []
for a, b in itertools.combinations(heads_used, 2):
    shared = sorted(set(cls_idx[a].tolist()) & set(cls_idx[b].tolist()))
    if len(shared) >= MINPAIR:
        pa = {t: k for k, t in enumerate(cls_idx[a].tolist())}; pb = {t: k for k, t in enumerate(cls_idx[b].tolist())}
        pairs.append((a, b, np.array([pa[t] for t in shared]), np.array([pb[t] for t in shared])))
st = strata(corpus, ['site', 'type'])
st_codes = {h: np.array([hash(st[i]) for i in cls_idx[h]]) for h in heads_used}
def Gstat(lab):
    return sum(len(lab[a][ia]) * mi_bits(lab[a][ia], lab[b][ib]) for a, b, ia, ib in pairs)
for delta in (0.4, 0.3, 0.2):
    det = 0; zs = []
    for q in range(NPLANT):
        hand = rng.integers(0, 2, N)            # two hands per text, independent of stratum
        lab = {}
        for h in heads_used:
            idx = cls_idx[h]; base = cls_lab[h]
            # stratum rate of non-head form
            rate = {}
            for code in np.unique(st_codes[h]):
                m = st_codes[h] == code; rate[code] = base[m].mean()
            r = np.array([rate[c] for c in st_codes[h]])
            p = np.clip(np.where(hand[idx] == 1, r + delta, r - delta), 0.02, 0.98)
            lab[h] = (rng.random(len(idx)) < p).astype(int)
        g = Gstat(lab)
        nullg = np.array([Gstat({h: permute_within(lab[h], st_codes[h], rng) for h in heads_used}) for _ in range(PPERM)])
        pv = (np.sum(nullg >= g) + 1) / (PPERM + 1); zs.append((g - nullg.mean()) / (nullg.std() + 1e-12))
        det += pv < 0.05
    lines.append(f'planted two hands (non-head rate = stratum rate +/- {delta}, random hand per text): detected (P<0.05, N1) in {det}/{NPLANT} plants, mean z {np.mean(zs):+.2f}')

json.dump(summary, open(OUT + f'loop46_cycle1{SUFFIX}.json', 'w'), indent=1)
open(OUT + f'loop46_cycle1{SUFFIX}_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
