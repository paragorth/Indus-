"""S-DARK-78: THE MISSING COMPARATOR - one-town logographic name registers (CBDB counties, jinshi cohorts).
S-DARK-74: Indus short-inside-long middles exceed a held-out-tuned bigram chain (x1.22-1.41 corpus, x1.37-1.98 within one
site), the excess is local and goes with a shared closer. Every logographic name LIST tested so far was nationwide and sat
below the chain (x0.13-0.51). Here the same statistic and the same calibrated chain (dark_loop74.cstats / SChain /
chain_names, imported unchanged) are run on registers of logographically written personal names from ONE county (index
place in CBDB) or ONE institution (a single jinshi exam year), with the role fields entry route and first office.
Cycle 1: substring share vs slot shuffle and calibrated chain, per register (full names 2-4 chars; given names >= 2),
         size-matched within-town vs pooled-across-towns (dilution), by length pair and position.
Cycle 2: outside facts for containment pairs: same entry route, same first office, kin link, same surname, same dynasty,
         vs a partner of the same length with the same last (or first) character that does not contain the short name.
Cycle 3: what makes the excess: kin / surname / period controls (drop kin-linked pairs, given names only, one dynasty
         within one county, persons with a role only), and the Indus side under the identical code for reference.
Usage: python3 tools/dark_loop78.py <1|2|3> [nnull]
"""
import sys, os, json, random, collections, math
_CY = int(sys.argv[1]); _NN = int(sys.argv[2]) if len(sys.argv) > 2 else 40
sys.argv = ['x', '0', 'seq_raw', '1']; sys.path.insert(0, 'tools')
import dark_loop74 as D
CY, NN = _CY, _NN
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
def save():
    open(f'data/derived/dark/loop78_c{CY}.txt', 'w').write('\n'.join(OUT) + '\n')
q, mean = D.q, D.mean
REC = [json.loads(l) for l in open('data/derived/dark/loop78_corpora/cbdb_registers.jsonl', encoding='utf-8')]
REGS = collections.defaultdict(list)
for p in REC:
    for r in p['reg']: REGS[r].append(p)
def label(r):
    ps = REGS[r]; return f'{r} {ps[0]["addr_name"] if r.startswith("county") else ""}'.strip()
def names_of(ps, field='name', minlen=2):
    return sorted(set(tuple(p[field]) for p in ps if p[field] and len(p[field]) >= minlen))

def chainstat(names, nn, seed=78):
    """observed share, slot-shuffle and calibrated-chain distributions (exactly as dark_loop74 cycle 5/6)."""
    ch = D.SChain(names, random.Random(76)); obs = D.cstats(names); ms = []; sl = []
    for b in range(nn):
        r = random.Random(seed * 100 + b)
        ms.append(D.cstats(sorted(set(D.chain_names(names, 's', r, ch)[0]))))
        sl.append(D.cstats(sorted(set(D.shuf_slot(names, r)))))
    return ch.lam, obs, ms, sl
def ratio(obs, ms, k='share'):
    v = [d[k] for d in ms]; mu = mean(v); lo, hi = q(v, .025), q(v, .975)
    p = (sum(1 for x in v if x >= obs) + 1) / (len(v) + 1)
    return mu, (obs / mu if mu else float('nan')), (obs / hi if hi else float('inf')), (obs / lo if lo else float('inf')), p
def line(lab, names, nn, seed=78, extra=True):
    lam, obs, ms, sl = chainstat(names, nn, seed)
    mu, rt, lo, hi, p = ratio(obs['share'], ms); smu, srt, _, _, sp = ratio(obs['share'], sl)
    lens = collections.Counter(len(n) for n in names)
    s = (f'  {lab:28s} n={len(names):5d} len{dict(sorted(lens.items()))} lam={lam:.2f} share {obs["share"]:.4f} ({obs["npairs"]} pairs) '
         f'| slot {smu:.4f} x{srt:.2f} | chain {mu:.4f} x{rt:.2f} [{lo:.2f},{hi:.2f}] p={p:.3f}')
    if extra:
        for k in ('f23', 'f24', 'f34'):
            m2, r2, _, _, _ = ratio(obs[k], ms, k)
            if obs[k] == obs[k]: s += f' | {k} {obs[k]:.3f}/{m2:.3f}'
        t = obs['pre'] + obs['suf'] + obs['inf']
        if t: s += f' | pre {obs["pre"] / t:.2f} suf {obs["suf"] / t:.2f} inf {obs["inf"] / t:.2f}'
    P(s)
    return dict(n=len(names), lam=lam, obs=obs['share'], chain=mu, ratio=rt, lo=lo, hi=hi, p=p, slot=smu)

COUNTIES = sorted([r for r in REGS if r.startswith('county')], key=lambda r: -len(names_of(REGS[r])))
COHORTS = sorted([r for r in REGS if r.startswith('jinshi')], key=lambda r: -len(names_of(REGS[r])))

# ================================================================== cycle 1
if CY == 1:
    R = {}
    P(f'##### S-DARK-78 cycle 1: substring share vs slot shuffle and the S-DARK-74 calibrated chain, {NN} draws; CI = observed / chain 97.5% and 2.5% quantiles')
    for field, ml in (('name', 2), ('given', 2)):
        P(f'\n=== field = {field} ({"full name, surname + given, 2-4 chars" if field == "name" else "given name only, >= 2 chars"}), distinct strings per register')
        for r in COUNTIES + COHORTS:
            nm = names_of(REGS[r], field, ml)
            if len(nm) < 80: P(f'  {label(r)} n={len(nm)} too small'); continue
            R[f'{field}|{r}'] = line(label(r), nm, NN)
        if field == 'name':
            # dilution at fixed n: within one town vs pooled across towns, same size
            for n0 in (400, 1000):
                P(f'\n  -- size-matched n={n0}: within one county (each county with >= n0 names, one random subsample) vs pooled across all 16 counties (5 random subsamples) vs pooled across 8 jinshi cohorts')
                big = [r for r in COUNTIES if len(names_of(REGS[r])) >= n0]
                W = []
                for r in big:
                    nm = names_of(REGS[r]); sub = sorted(random.Random(781).sample(nm, n0))
                    W.append(line(f'within {label(r)}', sub, NN, extra=False))
                POOL = sorted(set(n for r in COUNTIES for n in names_of(REGS[r])))
                PC = sorted(set(n for r in COHORTS for n in names_of(REGS[r])))
                X = [line(f'pooled counties draw {b}', sorted(random.Random(790 + b).sample(POOL, n0)), NN, extra=False) for b in range(5)]
                Y = [line(f'pooled cohorts draw {b}', sorted(random.Random(795 + b).sample(PC, n0)), NN, extra=False) for b in range(3)] if len(PC) >= n0 else []
                P(f'  == n={n0}: within-county ratio mean {mean([w["ratio"] for w in W]):.2f} (range {min(w["ratio"] for w in W):.2f}-{max(w["ratio"] for w in W):.2f}, {len(W)} counties) | '
                  f'pooled counties {mean([x["ratio"] for x in X]):.2f} | pooled cohorts {mean([y["ratio"] for y in Y]) if Y else float("nan"):.2f}')
                R[f'dilution{n0}'] = dict(within=[w['ratio'] for w in W], pooled=[x['ratio'] for x in X], cohorts=[y['ratio'] for y in Y])
            POOLALL = sorted(set(n for r in COUNTIES for n in names_of(REGS[r])))
            R['pooled_all'] = line('ALL 16 counties pooled', POOLALL, max(10, NN // 4), extra=True)
    json.dump(R, open('data/derived/dark/loop78_c1.json', 'w'), indent=1, ensure_ascii=False)
    save()
