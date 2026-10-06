"""v74 cycle 1c: score frozen P1 on the measured inter-word gaps (tools/v74_gaps.py).
P1 (frozen, v72_final.txt): at or.aiin, s.aiin, r.aiin, r.ain, ar.al, o.l, ol.chedy the written space is narrower
than the median word space of the same line in >= 60% of instances; at control junctions with the same
first-word ending (before any other word) it is not narrower.
Decision rule fixed before scoring:
  SURVIVES if target share(r < 1) >= 0.60 AND target r is lower than the frozen control (Mann-Whitney one-sided
  p < 0.05); KILLED if share < 0.60 and (target median r >= control median r, or the share's 95% upper bound
  < 0.60); otherwise UNDECIDED. Only valid if the measurement passes its own checks:
  M1 ZL uncertain spaces (',') measure narrower than certain spaces ('.') on the same pages;
  M2 Latin positive control: spaces after monosyllabic prepositions (in de ad a e ex cu/cum ab per/p-bar sub)
     narrower than after other short words of the same letter count (known proclitic writing habit);
  M3 shape-matched control: junctions with the same last glyph before / first glyph after the space as each
     target, other word pairs; M4 the same scores with gapw=0 re-alignment (gap0, r0).
Out: data/v74_ckpt/c1.json"""
import os, sys, json, math, collections
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v74_lib as X

RNG = np.random.default_rng(74)


def share(r):
    r = np.asarray([x for x in r if np.isfinite(x)])
    if len(r) == 0: return dict(n=0)
    k = int((r < 1).sum()); n = len(r)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95)
    return dict(n=n, k=k, share=k / n, ci=[lo, hi], med=float(np.median(r)), mean_log=float(np.mean(np.log(np.maximum(r, 0.05)))))


def mw(a, b):
    a = [x for x in a if np.isfinite(x)]; b = [x for x in b if np.isfinite(x)]
    if len(a) < 3 or len(b) < 3: return None
    return float(stats.mannwhitneyu(a, b, alternative='less').pvalue)


def load(names):
    R = []
    for nm in names:
        d = X.jload(nm)
        if d: R += d
    # quality rule fixed before scoring: drop pages whose line-pitch estimate sits at the search floor (28 px = failed)
    return [r for r in R if np.isfinite(r['gap']) and np.isfinite(r['lmed']) and r['lmed'] > 0 and r['pitch'] > 28.5]


def seps_for(R):
    Z = X.ivtff(os.path.join(X.ROOT, 'data', 'ZL3b-n.txt'))
    out = []
    for r in R:
        z = Z.get((r['folio'], r['n']))
        s = None
        if z and len(z['words']) == r['nw'] and z['words'][r['k']] == r['a'] and z['words'][r['k'] + 1] == r['b']:
            s = z['seps'][r['k']] if r['k'] < len(z['seps']) else None
        out.append(s)
    return out


def last_g(w): g = X.vglyphs(w); return g[-1] if g else ''
def first_g(w): g = X.vglyphs(w); return g[0] if g else ''


def score(R, key='r'):
    tgt = [r for r in R if (r['a'], r['b']) in X.TSET]
    firsts = {a for a, _ in X.TARGETS}
    ctlA = [r for r in R if r['a'] in firsts and (r['a'], r['b']) not in X.TSET]
    ctlB = [r for r in R if any(r['a'].endswith(a) for a in firsts) and (r['a'], r['b']) not in X.TSET]
    shapes = {(last_g(a), first_g(b)) for a, b in X.TARGETS}
    ctlS = [r for r in R if (last_g(r['a']), first_g(r['b'])) in shapes and (r['a'], r['b']) not in X.TSET]
    v = lambda S: [r[key] for r in S]
    out = dict(target=share(v(tgt)), ctl_same_word=share(v(ctlA)), ctl_same_ending=share(v(ctlB)),
               ctl_shape=share(v(ctlS)), all=share(v(R)),
               p_vs_same_word=mw(v(tgt), v(ctlA)), p_vs_same_ending=mw(v(tgt), v(ctlB)), p_vs_shape=mw(v(tgt), v(ctlS)))
    # shape-matched, per-target weighting: compare each target with controls of identical boundary shape
    per = {}
    for a, b in X.TARGETS:
        t = [r[key] for r in tgt if (r['a'], r['b']) == (a, b)]
        c = [r[key] for r in ctlS if (last_g(r['a']), first_g(r['b'])) == (last_g(a), first_g(b))]
        per[f'{a}.{b}'] = dict(t=share(t), c=share(c))
    out['per_target'] = per
    # stratified permutation: target labels shuffled within boundary-shape strata (shape-matched null)
    strata = collections.defaultdict(list)
    for r in tgt + ctlS: strata[(last_g(r['a']), first_g(r['b']))].append(r)
    obs = np.mean([r[key] < 1 for r in tgt]) if tgt else np.nan
    null = []
    for _ in range(5000):
        s = 0; n = 0
        for k, rs in strata.items():
            nt = sum((r['a'], r['b']) in X.TSET for r in rs)
            if nt == 0: continue
            idx = RNG.choice(len(rs), nt, replace=False)
            s += sum(rs[i][key] < 1 for i in idx); n += nt
        null.append(s / max(n, 1))
    out['strat_perm'] = dict(obs=obs, null_mean=float(np.mean(null)), p_le=float(np.mean(np.array(null) >= obs)))
    return out


def verdict(sc):
    t = sc['target']
    if t.get('n', 0) == 0: return 'UNTESTABLE'
    p = sc['p_vs_same_ending']
    if t['share'] >= 0.60 and p is not None and p < 0.05: return 'SURVIVES'
    if t['share'] < 0.60 and (t['med'] >= sc['ctl_same_ending']['med'] or t['ci'][1] < 0.60): return 'KILLED'
    return 'UNDECIDED'


def latin(R):
    preps = {'in', 'de', 'ad', 'a', 'e', 'ex', 'cu', 'cum', 'ab', 'ꝑ', 'per', 'sub', 'pro', 'ꝓ'}
    P = [r for r in R if r['a'] in preps]
    lens = collections.Counter(len(r['a']) for r in P)
    C = [r for r in R if r['a'] not in preps and len(r['a']) in lens and r['a'].isalpha()]
    out = {}
    for key in ('r', 'r0'):
        out[key] = dict(prep=share([r[key] for r in P]), other_short=share([r[key] for r in C]),
                        p=mw([r[key] for r in P], [r[key] for r in C]))
    return out


if __name__ == '__main__':
    RV = load(['gaps_V.json', 'gaps_VX.json'])
    RL = load(['gaps_L.json'])
    sp = seps_for(RV)
    for r, s in zip(RV, sp): r['sep'] = s
    out = dict(nV=len(RV), nL=len(RL), pages=len({r['folio'] for r in RV}))
    # M1 uncertain vs certain spaces
    out['M1'] = {key: dict(uncertain=share([r[key] for r in RV if r['sep'] == ',']),
                           certain=share([r[key] for r in RV if r['sep'] == '.']),
                           p=mw([r[key] for r in RV if r['sep'] == ','], [r[key] for r in RV if r['sep'] == '.']))
                 for key in ('r', 'r0')}
    out['M2'] = latin(RL)
    out['P1'] = score(RV, 'r'); out['P1']['verdict'] = verdict(out['P1'])
    out['P1_gapw0'] = score(RV, 'r0'); out['P1_gapw0']['verdict'] = verdict(out['P1_gapw0'])
    certain = [r for r in RV if r['sep'] == '.' or (r['a'], r['b']) not in X.TSET]
    out['P1_certain_only'] = score([r for r in RV if r['sep'] == '.'], 'r'); out['P1_certain_only']['verdict'] = verdict(out['P1_certain_only'])
    out['target_seps'] = collections.Counter(str(r['sep']) for r in RV if (r['a'], r['b']) in X.TSET)
    # alignment-agreement subset: junction placed at the same blank run by both aligners
    same = [r for r in RV if r['same0']]
    out['P1_aligners_agree'] = score(same, 'r'); out['P1_aligners_agree']['verdict'] = verdict(out['P1_aligners_agree'])
    out['agree_rate'] = float(np.mean([r['same0'] for r in RV]))
    X.jsave('c1.json', out)
    print(json.dumps(out, indent=1, default=float)[:12000])
