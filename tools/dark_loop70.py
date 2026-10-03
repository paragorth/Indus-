#!/usr/bin/env python3
"""Loop 70: do scribes' line breaks fall at word boundaries? (see dark_loop70_common.py for data/units)

Cycle 1: rate of breaks INSIDE each candidate unit vs three nulls (uniform random interior cut of the same
         text; ink-midpoint cut; ink-balance-matched cut), Wells canonical + order-agnostic at three merge
         levels, IM77 known order + agnostic, home (MD+HP) vs other sites. Controls: Ur III seal legends
         (word-internal pairs; true line breaks vs random re-breaks), planted Indus breaks (random vs
         unit-avoiding, same sizes as the Wells texts) for power.
Usage: python3 tools/dark_loop70.py 1
"""
import sys, json, random, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop70_common import *

rng = random.Random(70)
LOG = []
LOGFILE = [None]
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    if LOGFILE[0]:
        open(LOGFILE[0], 'a').write(s + '\n')


def ur3_legends():
    out = []
    for line in open(OUT + 'loop61_corpora/ur3_legend_fields.jsonl'):
        d = json.loads(line)
        segs, wordpos = [], []
        for ln in d['lines']:
            seg = []; wp = []
            for wi, w in enumerate(ln.split()):
                sg = [x for x in w.split('-') if x]
                for si, s in enumerate(sg):
                    seg.append(s); wp.append(si > 0)   # True = this sign continues the previous sign's word
            if seg:
                segs.append(tuple(seg)); wordpos.append(wp)
        if len(segs) >= 2:
            out.append(dict(segs=segs, wp=wordpos))
    return out


def ur3_sets(legs):
    inw = collections.Counter(); tot = collections.Counter()
    big = collections.Counter(); lc = collections.Counter(); rc = collections.Counter(); tw = collections.Counter()
    for L in legs:
        T = sum(L['segs'], ()); WP = sum(L['wp'], [])
        # within-line adjacency word status (line breaks are not used: cross-line pairs are excluded here
        # only for the WORD label; they are rare and would bias the label toward 'not in word')
        for seg, wp in zip(L['segs'], L['wp']):
            for i in range(1, len(seg)):
                p = (seg[i - 1], seg[i]); tot[p] += 1; inw[p] += wp[i]
        for p in set(zip(T, T[1:])):
            tw[p] += 1
        for a, b in zip(T, T[1:]):
            big[(a, b)] += 1; lc[a] += 1; rc[b] += 1
    word = {p for p in tot if inw[p] / tot[p] >= 0.8}
    frozen = {p for p, n in tw.items() if n >= 5 and big[p] / lc[p[0]] >= 0.4 and big[p] / rc[p[1]] >= 0.4}
    return dict(QH=set(), OPEN=set(), NUM=set(), FISH=word, FROZEN=frozen, UNIT=word | frozen), set()


def planted(one, sets, frame, W, sizes, n, avoid, reps=200):
    """Indus single-line texts of the right length, broken at random (avoid=False) or only at positions
    not inside a UNIT pair (avoid=True); returns O/E (uniform) and P_low for UNIT over reps."""
    bylen = collections.defaultdict(list)
    for t in one:
        bylen[len(t['seq'])].append(t['seq'])
    oes, sig = [], 0
    for _ in range(reps):
        texts = []
        for k in rng.sample(sizes, min(n, len(sizes))):
            pool = bylen.get(sum(k), [])
            if not pool:
                continue
            s = rng.choice(pool)
            cuts = list(range(1, len(s)))
            if avoid:
                ok = [c for c in cuts if (s[c - 1], s[c]) not in sets['UNIT']]
                cuts = ok or cuts
            c = rng.choice(cuts)
            texts.append(dict(segs=[s[:c], s[c:]]))
        r = category_test(texts, 'known', sets, frame, W, cats=['UNIT'])['UNIT']
        oes.append(r['uniform']['OE']); sig += r['uniform']['p_low'] < 0.05
    return float(np.median(oes)), sig / reps


def cycle1():
    LOGFILE[0] = OUT + 'loop70_cycle1_log.txt'; open(LOGFILE[0], 'w').close()
    W = widths()
    br = bridge(True)
    P('LOOP 70 cycle 1: break rate inside candidate units vs random / ink-midpoint / ink-balance nulls')
    res = {}
    for lv in LEVELS:
        one, multi = wells(lv)
        sets, frame = unit_sets(one)
        if lv == 'seq_raw':
            P(f'Wells: {len(one)} single-line complete texts (deduplicated), {len(multi)} multi-segment texts '
              f'({collections.Counter(len(t["segs"]) for t in multi)}), home {sum(is_home(t["site"]) for t in multi)}')
            P('Unit set sizes: ' + ', '.join(f'{c} {len(sets[c])}' for c in ('QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT')))
            P('FROZEN pairs: ' + ' '.join(f'{a}-{b}' for a, b in sorted(sets['FROZEN'])))
            P('Observed Wells junctions (canonical order, reading order: last of segment | first of next) with category:')
            jc = collections.Counter()
            for t in multi:
                for p in junctions_known(t['segs']):
                    jc[(p, '/'.join(pair_cat(p, sets, frame)))] += 1
            P('  ' + '; '.join(f'{a}|{b} [{c}] x{n}' for ((a, b), c), n in jc.most_common()))
            P('Segment sizes: ' + str(collections.Counter(tuple(len(s) for s in t['segs']) for t in multi).most_common()))
        for mode in ('canon', 'agnostic'):
            for sub, f in (('all', lambda t: True), ('home', lambda t: is_home(t['site'])), ('other', lambda t: not is_home(t['site']))):
                tx = [t for t in multi if f(t)]
                r = category_test(tx, mode, sets, frame, W)
                res[f'wells|{lv}|{mode}|{sub}'] = r
                P(f'-- Wells {lv} {mode} {sub} (n={len(tx)})')
                for c in CATS:
                    P('   ' + fmt_row(c, r[c]))
    # IM77
    one, multi = im77()
    smap = lambda w: br.get(w, [])
    sets, frame = unit_sets(one, smap)
    P(f'IM77: {len(one)} one-line sides, {len(multi)} multi-line sides (lines: {collections.Counter(len(t["segs"]) for t in multi)}); '
      f'home {sum(is_home(t["site"]) for t in multi)}. Unit sizes: ' + ', '.join(f'{c} {len(sets[c])}' for c in ('QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT')))
    jc = collections.Counter()
    for t in multi:
        for p in junctions_known(t['segs']):
            jc[(p, '/'.join(pair_cat(p, sets, frame)))] += 1
    P('IM77 junctions (line i last | line i+1 first): ' + '; '.join(f'{a}|{b} [{c}] x{n}' for ((a, b), c), n in jc.most_common(60)))
    for mode in ('known', 'agnostic'):
        for sub, f in (('all', lambda t: True), ('home', lambda t: is_home(t['site'])), ('other', lambda t: not is_home(t['site']))):
            tx = [t for t in multi if f(t)]
            r = category_test(tx, mode, sets, frame, {})
            res[f'im77|{mode}|{sub}'] = r
            P(f'-- IM77 {mode} {sub} (n={len(tx)})  [widths: M signs have no font width, all 0.52 -> midpoint = sign-count midpoint]')
            for c in CATS:
                P('   ' + fmt_row(c, r[c]))
    # Ur III positive control
    legs = ur3_legends()
    usets, uframe = ur3_sets(legs)
    P(f'Ur III control: {len(legs)} distinct legends with >= 2 lines; WORD pairs {len(usets["FISH"])} (label FISH below = WORD), FROZEN {len(usets["FROZEN"])}')
    sample = rng.sample(legs, 400)
    P(f'   (agnostic mode on the {sum(len(L["segs"]) <= 3 for L in sample)} legends of 2-3 lines only; k! orders explode beyond)')
    for mode in ('known', 'agnostic'):
        smp = sample if mode == 'known' else [L for L in sample if len(L['segs']) <= 3]
        r = category_test(smp, mode, usets, uframe, {}, cats=['FISH', 'FROZEN', 'UNIT', 'BOUND'])
        res[f'ur3|{mode}'] = r
        for c in ('FISH', 'FROZEN', 'UNIT', 'BOUND'):
            P(f'   Ur III {mode} ' + fmt_row('WORD' if c == 'FISH' else c, r[c]))
    # Ur III random re-break (negative control)
    rb = []
    for L in sample:
        T = sum(L['segs'], ()); k = len(L['segs'])
        cuts = sorted(rng.sample(range(1, len(T)), k - 1)) if len(T) > k - 1 else None
        if not cuts:
            continue
        b = [0] + cuts + [len(T)]
        rb.append(dict(segs=[T[b[i]:b[i + 1]] for i in range(k)]))
    r = category_test(rb, 'known', usets, uframe, {}, cats=['FISH', 'UNIT'])
    P('   Ur III RANDOM re-break ' + fmt_row('WORD', r['FISH'])); P('   Ur III RANDOM re-break ' + fmt_row('UNIT', r['UNIT']))
    res['ur3|rebreak'] = r
    # Planted Indus power
    one, multi = wells('seq_raw'); sets, frame = unit_sets(one)
    sizes = [tuple(len(s) for s in t['segs']) for t in multi if len(t['segs']) == 2]
    for avoid in (False, True):
        med, power = planted(one, sets, frame, W, sizes, len(sizes), avoid)
        P(f'Planted Indus breaks ({"unit-avoiding" if avoid else "random"}; n={len(sizes)} single-line texts with the Wells size mix): '
          f'median O/E(UNIT, uniform) {med:.2f}; share of 200 replicates with P_low < 0.05: {power:.2f}')
        res[f'planted|avoid={avoid}'] = dict(median_OE=med, power=power)
    json.dump(res, open(OUT + 'loop70_cycle1.json', 'w'), indent=0, default=str)




# ====================================================================== cycle 2
from scipy.optimize import minimize

FEATS = ['UNIT', 'MIDMID', 'AFTERHEAD', 'IMBAL']


def pair_feats(p, sets, frame, heads):
    c = pair_cat(p, sets, frame)
    if 'UNIT' in c:
        return (1, 0, 0)
    if 'MIDMID' in c:
        return (0, 1, 0)
    if p[0] in heads:
        return (0, 0, 1)
    return (0, 0, 0)          # reference: other frame edge


def text_choices(segs, mode, sets, frame, heads, W, junc_ok=None):
    """For one text: list over orders of (index of observed config, feature matrix of all configs)."""
    k = len(segs)
    J = junctions_agnostic if mode == 'agnostic' else junctions_known
    orders = list(itertools.permutations(range(k))) if mode == 'agnostic' else [tuple(range(k))]
    out = []
    for o in orders:
        T = sum((tuple(segs[i]) for i in o), ())
        obs = [tuple(segs[i]) for i in o]
        confs = list(cut_configs(T, k))
        F = []; idx = None
        sc = np.mean([W.get(s, 0.52) for s in T])
        for ci, c in enumerate(confs):
            f = np.zeros(4)
            for p in junctions_known(c):          # features use the TRUE junctions of this order
                f[:3] += pair_feats(p, sets, frame, heads)
            f[3] = imbalance(c, W) / sc
            F.append(f)
            if [tuple(x) for x in c] == obs:
                idx = ci
        out.append((idx, np.array(F)))
    return out


def fit_clogit(data, feats=(0, 1, 2, 3)):
    feats = list(feats)
    if all(len(o) == 1 for o in data):
        Fs = [o[0][1] for o in data]; L = max(len(F) for F in Fs)
        X = np.zeros((len(Fs), L, 4)); M = np.full((len(Fs), L), -1e9)
        for i, F in enumerate(Fs):
            X[i, :len(F)] = F; M[i, :len(F)] = 0.0
        obs = np.array([o[0][0] for o in data])
        def nllv(b):
            beta = np.zeros(4); beta[feats] = b
            u = X @ beta + M; m = u.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(u - m).sum(axis=1))
            return -(u[np.arange(len(obs)), obs] - lse).sum()
        r = minimize(nllv, np.zeros(len(feats)), method='L-BFGS-B', bounds=[(-8, 8)] * len(feats))
        beta = np.zeros(4); beta[feats] = r.x
        return beta, r.fun
    def nll(b):
        beta = np.zeros(4); beta[feats] = b
        tot = 0.0
        for orders in data:
            ls = []
            for idx, F in orders:
                u = F @ beta; m = u.max()
                ls.append(u[idx] - m - np.log(np.exp(u - m).sum()))
            ls = np.array(ls); mm = ls.max()
            tot -= mm + np.log(np.exp(ls - mm).mean())
        return tot
    r = minimize(nll, np.zeros(len(feats)), method='L-BFGS-B', bounds=[(-8, 8)] * len(feats))
    beta = np.zeros(4); beta[feats] = r.x
    return beta, r.fun


def clogit_report(label, data, nboot=200):
    beta, ll = fit_clogit(data)
    # LR test MIDMID = 0 (= other frame edge)
    _, ll0 = fit_clogit(data, feats=(0, 2, 3))
    lr = 2 * (ll0 - ll)
    from scipy.stats import chi2
    pm = float(chi2.sf(max(lr, 0), 1))
    bs = [beta]
    for _ in range(nboot):
        smp = [data[rng.randrange(len(data))] for _ in range(len(data))]
        bs.append(fit_clogit(smp)[0])
    bs = np.array(bs)
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    P(f'   {label} (n={len(data)}): ' + ', '.join(f'{f} {beta[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]' for i, f in enumerate(FEATS))
      + f' | LR MIDMID=ref chi2 {lr:.2f} P {pm:.3f}')
    return dict(beta=beta.tolist(), lo=lo.tolist(), hi=hi.tolist(), lr_midmid=lr, p_midmid=pm, n=len(data))


def pmi_table(one):
    big = collections.Counter(); lc = collections.Counter(); rc = collections.Counter()
    for t in one:
        s = t['seq']
        for a, b in zip(s, s[1:]):
            big[(a, b)] += 1; lc[a] += 1; rc[b] += 1
    N = sum(big.values())
    def pmi(p):
        return math.log2((big[p] + 0.5) * N / ((lc[p[0]] + 1) * (rc[p[1]] + 1)))
    return pmi


def middle_cohesion(texts, mode, sets, frame, pmi):
    """Texts whose (known/canonical) junction is MIDMID and that have >= 2 MIDMID adjacencies in total:
    percentile of the junction PMI among the text's MIDMID adjacencies (0 = least cohesive)."""
    pct, null = [], []
    for t in texts:
        segs = t['segs']
        T = sum((tuple(s) for s in segs), ())
        juncs = set()
        pos = 0
        for s in segs[:-1]:
            pos += len(s); juncs.add(pos)
        mids = [i for i in range(1, len(T)) if 'MIDMID' in pair_cat((T[i - 1], T[i]), sets, frame)]
        jm = [i for i in mids if i in juncs]
        if len(mids) < 2 or not jm:
            continue
        vals = [pmi((T[i - 1], T[i])) for i in mids]
        for j in jm:
            v = pmi((T[j - 1], T[j]))
            below = sum(x < v for x in vals) + 0.5 * (sum(x == v for x in vals) - 1)
            pct.append(below / (len(vals) - 1))
    return pct


def learned_boundaries(texts, one, sets, frame, mode, nnull=200):
    """Sign-level break propensities learned from the breaks alone; AUC on single-line adjacencies:
    UNIT vs other, MIDMID vs BOUND (low score = cohesive). Null: breaks replaced by uniform random cuts."""
    def learn(tx):
        after = collections.Counter(); before = collections.Counter()
        oa = collections.Counter(); ob = collections.Counter()
        nb = 0; no = 0
        for segs in tx:
            T = sum((tuple(s) for s in segs), ())
            js = junctions_known(segs) if mode != 'agnostic' else junctions_agnostic(segs)
            w = 1.0 if mode != 'agnostic' else 1.0 / len(js) * (len(segs) - 1)
            for a, b in js:
                after[a] += w; before[b] += w; nb += w
            for a, b in zip(T, T[1:]):
                oa[a] += 1; ob[b] += 1; no += 1
        base = nb / no; k = 2.0
        ra = lambda x: math.log((after[x] + k * base) / (oa[x] + k))
        rb = lambda y: math.log((before[y] + k * base) / (ob[y] + k))
        return lambda p: ra(p[0]) + rb(p[1])
    def auc(score, pos, neg):
        if not pos or not neg:
            return float('nan')
        sp = np.array([score(p) for p in pos]); sn = np.array([score(p) for p in neg])
        # P(score(pos) < score(neg)): cohesive pairs should have LOWER break score
        allv = np.concatenate([sp, sn]); r = allv.argsort().argsort() + 1.0
        from scipy.stats import rankdata
        r = rankdata(allv)
        u = r[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2
        return 1 - u / (len(sp) * len(sn))
    unit, midm, bound = [], [], []
    for t in one:
        s = t['seq']
        for p in zip(s, s[1:]):
            c = pair_cat(p, sets, frame)
            (unit if 'UNIT' in c else midm if 'MIDMID' in c else bound).append(p)
    segsl = [t['segs'] for t in texts]
    sc = learn(segsl)
    obs = (auc(sc, unit, midm + bound), auc(sc, midm, bound), auc(sc, unit, bound))
    nulls = []
    for _ in range(nnull):
        fake = []
        for segs in segsl:
            T = sum((tuple(s) for s in segs), ()); k = len(segs)
            cuts = sorted(rng.sample(range(1, len(T)), k - 1)); b = [0] + cuts + [len(T)]
            fake.append([T[b[i]:b[i + 1]] for i in range(k)])
        f = learn(fake)
        nulls.append((auc(f, unit, midm + bound), auc(f, midm, bound), auc(f, unit, bound)))
    nulls = np.array(nulls)
    pv = [(np.sum(nulls[:, i] >= obs[i]) + 1) / (nnull + 1) for i in range(3)]
    return obs, nulls.mean(axis=0), np.percentile(nulls, 95, axis=0), pv, sc


def cycle2():
    LOGFILE[0] = OUT + 'loop70_cycle2_log.txt'; open(LOGFILE[0], 'w').close()
    W = widths(); br = bridge(True)
    P('LOOP 70 cycle 2: conditional logit over cut positions (features UNIT-internal, MIDMID, AFTERHEAD = after a closer/suffix,'
      ' IMBAL = ink imbalance / mean sign ink; reference = other frame edge), middle cohesion of MIDMID breaks, and boundary'
      ' propensities learned from the breaks alone')
    res = {}
    corpora = []
    for lv in LEVELS:
        one, multi = wells(lv); sets, frame = unit_sets(one)
        heads = (CLOSERS | SUFFIX)
        corpora.append((f'Wells {lv}', one, multi, sets, frame, heads, W, ('canon', 'agnostic')))
    one, multi = im77(); smap = lambda w: br.get(w, []); sets, frame = unit_sets(one, smap)
    heads = set()
    for w in CLOSERS | SUFFIX:
        heads |= set(smap(w))
    corpora.append(('IM77', one, multi, sets, frame, heads, {}, ('known',)))
    for name, one, multi, sets, frame, heads, Wd, modes in corpora:
        P(f'-- {name}')
        pmi = pmi_table(one)
        for mode in modes:
            for sub, f in (('all', lambda t: True), ('home', lambda t: is_home(t['site'])), ('other', lambda t: not is_home(t['site']))):
                tx = [t for t in multi if f(t)]
                data = [text_choices(t['segs'], mode, sets, frame, heads, Wd) for t in tx]
                data = [d for d in data if all(i is not None for i, _ in d)]
                res[f'{name}|{mode}|{sub}|clogit'] = clogit_report(f'{mode} {sub}', data, nboot=(60 if sub == 'all' else 0) if (name.endswith('raw') or name == 'IM77') else 0)
            pct = middle_cohesion(multi, mode, sets, frame, pmi)
            if pct:
                # null: uniform percentile -> mean 0.5; exact-ish by simulation of discrete uniform positions
                m = float(np.mean(pct))
                sims = [np.mean([rng.random() for _ in pct]) for _ in range(5000)]
                pl = (sum(s <= m for s in sims) + 1) / 5001
                P(f'   middle cohesion ({mode}): {len(pct)} MIDMID breaks in texts with >= 2 MIDMID adjacencies; mean PMI percentile of the '
                  f'broken pair among the text\'s MIDMID pairs {m:.2f} (0 = least cohesive; null 0.50, P_low {pl:.3f})')
                res[f'{name}|{mode}|midcoh'] = dict(n=len(pct), mean=m, p_low=pl)
            obs, nm, n95, pv, sc = learned_boundaries(multi, one, sets, frame, mode)
            P(f'   learned boundaries ({mode}): AUC UNIT-vs-rest {obs[0]:.3f} (null mean {nm[0]:.3f}, 95th {n95[0]:.3f}, P {pv[0]:.3f}); '
              f'MIDMID-vs-BOUND {obs[1]:.3f} (null {nm[1]:.3f}, 95th {n95[1]:.3f}, P {pv[1]:.3f}); UNIT-vs-BOUND {obs[2]:.3f} (null {nm[2]:.3f}, P {pv[2]:.3f})')
            res[f'{name}|{mode}|learned'] = dict(auc=list(obs), null_mean=nm.tolist(), null95=n95.tolist(), p=pv)
    # Ur III positive control for the conditional logit: WORD as UNIT, reference = between-word junction
    legs = ur3_legends(); usets, uframe = ur3_sets(legs)
    smp = [L for L in rng.sample(legs, 600) if len(L['segs']) <= 3]
    data = [text_choices(L['segs'], 'known', usets, uframe, set(), {}) for L in smp]
    P('-- Ur III control (UNIT = within-word sign pair; MIDMID unused; reference = between-word junction)')
    res['ur3|clogit'] = clogit_report('Ur III known', data, nboot=50)
    smp2 = [L for L in smp if len(L['segs']) == 2]
    data = [text_choices(L['segs'], 'agnostic', usets, uframe, set(), {}) for L in smp2]
    res['ur3|clogit_agn'] = clogit_report('Ur III agnostic (2-line)', data, nboot=50)
    # power for MIDMID: simulate breaks on Wells single-line texts from a model with MIDMID = -1.0 / -2.0
    one, multi = wells('seq_raw'); sets, frame = unit_sets(one); heads = CLOSERS | SUFFIX
    sizes = [tuple(len(s) for s in t['segs']) for t in multi if len(t['segs']) == 2]
    bylen = collections.defaultdict(list)
    for t in one:
        bylen[len(t['seq'])].append(t['seq'])
    fitted = res['Wells seq_raw|canon|all|clogit']['beta']
    for bm in (0.0, -1.0, -2.0):
        hits = 0; ests = []
        for rep in range(40):
            data = []
            for sz in sizes:
                pool = bylen.get(sum(sz))
                if not pool:
                    continue
                s = rng.choice(pool)
                ch = text_choices([s[:1], s[1:]], 'known', sets, frame, heads, W)[0][1]
                beta = np.array([fitted[0], bm, fitted[2], fitted[3]])
                u = ch @ beta; pr = np.exp(u - u.max()); pr /= pr.sum()
                c = int(np.searchsorted(np.cumsum(pr), rng.random()))
                c = min(c, len(pr) - 1)
                data.append([(c, ch)])
            b, ll = fit_clogit(data); _, ll0 = fit_clogit(data, feats=(0, 2, 3))
            from scipy.stats import chi2
            hits += chi2.sf(max(2 * (ll0 - ll), 0), 1) < 0.05 and b[1] < 0; ests.append(b[1])
        P(f'   power: planted MIDMID coefficient {bm:+.1f} (other coefficients = fitted Wells canon), n={len(sizes)}: '
          f'median estimate {np.median(ests):+.2f}, detected (P < 0.05, negative) in {hits}/40')
        res[f'power|{bm}'] = dict(median=float(np.median(ests)), hits=hits)
    json.dump(res, open(OUT + 'loop70_cycle2.json', 'w'), indent=0, default=str)


# ====================================================================== cycle 3
def norm_site(s):
    s = s.lower().replace('-', '').replace(' ', '')
    return s


def to_m_options(seq, br, cap=64):
    opts = [br.get(w, []) for w in seq]
    if any(not o for o in opts):
        return []
    out = [()]
    for o in opts:
        out = [x + (m,) for x in out for m in o][:cap]
    return out


def cycle3():
    LOGFILE[0] = OUT + 'loop70_cycle3_log.txt'; open(LOGFILE[0], 'w').close()
    W = widths()
    P('LOOP 70 cycle 3: Wells vs IM77 break agreement on matched objects; true held-out sites; junction anatomy; bridge dependence')
    res = {}
    for use_prop in (False, True):
        br = bridge(use_prop)
        # all Wells texts incl. single-line, canonical order, raw level, with damage-free signs
        one_w, multi_w = wells('seq_raw')
        # need single-line Wells texts regardless of 'complete' flag for matching: rebuild quickly
        wl = []
        for t in one_w:
            wl.append(dict(site=t['site'], segs=[t['seq']]))
        for t in multi_w:
            wl.append(dict(site=t['site'], segs=t['segs']))
        one_m, multi_m = im77()
        ml = [dict(site=t['site'], segs=[t['seq']]) for t in one_m] + [dict(site=t['site'], segs=t['segs']) for t in multi_m]
        idx = collections.defaultdict(list)
        for i, t in enumerate(ml):
            T = sum((tuple(s) for s in t['segs']), ())
            if len(T) >= 3:
                idx[(norm_site(t['site'])[:6], tuple(sorted(T)))].append(i)
        pairs = []
        for t in wl:
            T = sum((tuple(s) for s in t['segs']), ())
            if len(T) < 3:
                continue
            hits = set()
            for mT in to_m_options(T, br):
                for i in idx.get((norm_site(t['site'])[:6], tuple(sorted(mT))), []):
                    hits.add(i)
            if len(hits) >= 1:
                pairs.append((t, [ml[i] for i in hits]))
        wm = [(t, ms) for t, ms in pairs if len(t['segs']) > 1]
        mm_ = [(t, ms) for t, ms in pairs if any(len(m['segs']) > 1 for m in ms)]
        both = [(t, ms) for t, ms in pairs if len(t['segs']) > 1 and any(len(m['segs']) > 1 for m in ms)]
        P(f'-- bridge {"with" if use_prop else "WITHOUT"} S-DARK-27 proposals: Wells texts (>= 3 signs) matched to an IM77 side of the same site '
          f'and M multiset: {len(pairs)}; Wells multi-segment among them {len(wm)}, of which IM77 also multi-line {sum(1 for t, ms in wm if any(len(m["segs"]) > 1 for m in ms))}; '
          f'IM77 multi-line among them {len(mm_)}, of which Wells also has "/" {sum(1 for t, ms in mm_ if len(t["segs"]) > 1)}')
        # junction agreement
        agree = 0; tot = 0; chance = 0.0; order_same = 0; order_rev = 0; ex = []
        for t, ms in both:
            m = [x for x in ms if len(x['segs']) > 1][0]
            if len(t['segs']) != 2 or len(m['segs']) != 2:
                continue
            tot += 1
            wsz = sorted(len(s) for s in t['segs']); msz = sorted(len(s) for s in m['segs'])
            n = sum(wsz)
            same = wsz == msz
            agree += same
            chance += 2.0 / (n - 1) if wsz[0] != wsz[1] else 1.0 / (n - 1)
            if same:
                if [len(s) for s in t['segs']] == [len(s) for s in m['segs']]:
                    order_same += 1
                else:
                    order_rev += 1
            else:
                ex.append((t['site'], t['segs'], m['segs']))
        P(f'   both transcriptions record a break: {tot} two-line pairs; the break falls at the same place (same segment sizes) in {agree} '
          f'(chance if the second transcription broke at a random position: {chance:.1f}); Wells canonical segment order = IM77 line order '
          f'in {order_same}, reversed in {order_rev}; disagreements: ' + '; '.join(f'{s} W{a} M{b}' for s, a, b in ex[:8]))
        res[f'match|prop={use_prop}'] = dict(pairs=len(pairs), wells_multi=len(wm), im77_multi=len(mm_), both=tot, agree=agree,
                                             chance=chance, order_same=order_same, order_rev=order_rev)
        # IM77 unit test with/without proposals
        smap = lambda w: br.get(w, [])
        sets, frame = unit_sets(one_m, smap)
        r = category_test(multi_m, 'known', sets, frame, {}, cats=['QH', 'NUM', 'UNIT', 'MIDMID', 'BOUND'])
        for c in ('QH', 'NUM', 'UNIT', 'MIDMID', 'BOUND'):
            P('   IM77 known ' + fmt_row(c, r[c]))
        res[f'im77|prop={use_prop}'] = r
    # true held-out: Wells multi-line texts at sites absent from IM77 (post-1977 finds etc.)
    one_m, multi_m = im77()
    im_sites = {norm_site(t['site'])[:6] for t in one_m + multi_m}
    for lv in LEVELS:
        one, multi = wells(lv); sets, frame = unit_sets(one)
        ho = [t for t in multi if norm_site(t['site'])[:6] not in im_sites]
        if lv == 'seq_raw':
            P(f'-- held-out sites (not in IM77): {len(ho)} Wells multi-segment texts: ' + str(collections.Counter(t['site'] for t in ho).most_common()))
        for mode in ('canon', 'agnostic'):
            r = category_test(ho, mode, sets, frame, W, cats=['UNIT', 'MIDMID', 'BOUND'])
            for c in ('UNIT', 'MIDMID', 'BOUND'):
                P(f'   held-out {lv} {mode} ' + fmt_row(c, r[c]))
            res[f'heldout|{lv}|{mode}'] = r
        # also Wells home-fit units applied to other sites: units learned on MD+HP only
        one_h = [t for t in one if is_home(t['site'])]
        sets_h, frame_h = unit_sets(one_h)
        oth = [t for t in multi if not is_home(t['site'])]
        r = category_test(oth, 'canon', sets_h, frame_h, W, cats=['UNIT', 'MIDMID', 'BOUND'])
        P(f'   units learned on MD+HP single-line texts only, other-site multi-line texts (n={len(oth)}), {lv} canon: ' +
          ' || '.join(fmt_row(c, r[c]) for c in ('UNIT', 'MIDMID')))
        res[f'homefit_other|{lv}'] = r
    # junction anatomy (Wells canonical, raw; IM77 known)
    br = bridge(True)
    for name, (one, multi), smap, mode, Wd in (('Wells seq_raw canon', wells('seq_raw'), None, 'canon', W),
                                               ('IM77 known', im77(), (lambda w: br.get(w, [])), 'known', {})):
        sets, frame = unit_sets(one, smap)
        S = (lambda ws: set().union(*[set(smap(w)) for w in ws])) if smap else (lambda ws: set(ws))
        heads = S(CLOSERS | SUFFIX); opens = S(OPENERS | CONNECT); nums = S(NUMERALS | MARKERS)
        def kind(p):
            c = pair_cat(p, sets, frame)
            if 'UNIT' in c: return 'inside unit'
            if 'MIDMID' in c: return 'middle|middle'
            if p[0] in heads: return 'after closer/suffix'
            if p[1] in heads: return 'before closer (non-qualifier)'
            if p[0] in opens: return 'after opener/connective'
            if p[0] in nums or p[1] in nums: return 'numeral edge'
            return 'other frame edge'
        obs = collections.Counter(); exp = collections.Counter()
        for t in multi:
            segs = t['segs']; k = len(segs)
            for p in junctions_known(segs):
                obs[kind(p)] += 1
            T = sum((tuple(s) for s in segs), ())
            confs = list(cut_configs(T, k))
            for c in confs:
                for p in junctions_known(c):
                    exp[kind(p)] += 1.0 / len(confs)
        P(f'-- junction anatomy, {name} (uniform-cut expectation in brackets): ' + '; '.join(
            f'{kk} {obs[kk]} ({exp[kk]:.1f}, O/E {obs[kk] / exp[kk] if exp[kk] else float("nan"):.2f})' for kk in sorted(exp, key=lambda x: -exp[x])))
        res[f'anatomy|{name}'] = dict(obs=dict(obs), exp=dict(exp))
    json.dump(res, open(OUT + 'loop70_cycle3.json', 'w'), indent=0, default=str)


if __name__ == '__main__':
    for c in sys.argv[1:] or ['1']:
        globals()['cycle' + c]()
