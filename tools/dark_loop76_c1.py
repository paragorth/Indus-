"""S-DARK-76 cycle 1: the recurrence profile of the designation per object kind and per site, against archives
of persons (Linear B personnel, Ur III sealings = owner names over sealed tablets, Ur III administrative names)
and a seal population (Ur III distinct legends), with Zipf / Yule fits and a pooled-draw null.
Recurrence = share of designation tokens (>= 2 elements; one per designation per document) whose designation stands
on ANOTHER document. Matched n (40, 100, 300 tokens), two samplers:
  random  n tokens drawn from the whole kind (40 draws)
  block   a contiguous run of documents (sorted by site, then object / tablet id) until n tokens: one archive deposit
Null (pool): n iid draws from the pooled Indus designation distribution (all kinds, act level) = recurrence expected
if the kind were a random sample of one designation population.
Usage: python3 tools/dark_loop76_c1.py   (writes data/derived/dark/loop76_c1.txt)"""
import sys; sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop76_common import *
NS = [40, 100, 300]; ND = 40

def block_sample(items, n, r, key):
    """items: list of (doc, des, sortkey); contiguous docs from a random start until n tokens."""
    docs = collections.OrderedDict()
    for d, m, k in sorted(items, key=lambda x: x[2]): docs.setdefault(d, []).append(m)
    keys = list(docs)
    if sum(len(v) for v in docs.values()) <= n: return [(d, m) for d in keys for m in docs[d]]
    st = r.randrange(len(keys)); out = []; i = st
    while len(out) < n:
        d = keys[i % len(keys)]; out.extend((d, m) for m in docs[d]); i += 1
    return out[:n]

def ladder(label, items, pool=None):
    """items: (doc, des, sortkey)."""
    toks = [(d, m) for d, m, _ in items]
    pr = profile(toks); cnt = collections.Counter()
    for d, m in set(toks): cnt[m] += 1
    counts = list(cnt.values())
    line = f'  [{label}] tokens {pr["tokens"]}, types {pr["types"]}, docs/type {pr["mean"]:.2f}, types on >= 2 docs {pr["share2"]:.3f}, max {pr["max"]}, recur(all) {pr["recur"]:.3f}; Zipf {zipf_slope(counts):.2f}, Yule rho {yule_rho(counts):.2f}'
    res = dict(profile=pr, zipf=zipf_slope(counts), yule=yule_rho(counts))
    for n in NS:
        if len(toks) < n * 0.9: continue
        ro = []; bo = []; po = []
        for b in range(ND):
            r = random.Random(7600 + b)
            ro.append(recur_share(sub(toks, n, r)))
            bo.append(recur_share(block_sample(items, n, r, None)))
            if pool: po.append(recur_share([(j, m) for j, m in enumerate(r.choices(pool[0], weights=pool[1], k=n))]))
        res[n] = dict(random=q(ro, .5), rlo=q(ro, .025), rhi=q(ro, .975), block=q(bo, .5), blo=q(bo, .025), bhi=q(bo, .975), pool=q(po, .5) if po else None)
        line += f'\n      n={n}: random {q(ro,.5):.3f} [{q(ro,.025):.3f},{q(ro,.975):.3f}]  block {q(bo,.5):.3f} [{q(bo,.025):.3f},{q(bo,.975):.3f}]' + (f'  pool-null {q(po,.5):.3f} [{q(po,.025):.3f},{q(po,.975):.3f}]' if po else '')
    P(line); return res

R = {}
P('== S-DARK-76 cycle 1: recurrence profile of the designation by object kind (one designation per document; >= 2 elements)')
P('#### comparators (document = tablet / distinct legend)')
COMP = {}
for nm, f in [('Linear B personnel names (doc = tablet)', linb_personnel), ('Ur III sealings: owner name over sealed tablets (doc = tablet)', ur3_sealings),
              ('Ur III seals: owner name over DISTINCT legends (doc = one seal)', ur3_seals), ('Ur III administrative names (doc = tablet)', ur3_admin)]:
    X = f()
    items = [(x['doc'], x['des'], (x['site'], int(re.sub(r'\D', '', x['doc']) or 0) if nm.startswith('Ur') else x['doc'])) for x in X]
    COMP[nm] = ladder(nm, items)
R['comp'] = COMP

def run_indus(label, docs, pool):
    out = {}
    for k in ['seal', 'sealing', 'tablet_m', 'tablet_i', 'tablet_c', 'pot', 'other']:
        T = tokens(docs, 2, {k})
        if len(T) < 20: continue
        items = [(i, m, (docs[i]['site'], i)) for i, m, f in T]
        out[k] = ladder(f'{label} {k}', items, pool)
    return out

for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    for filt in ['strict', 'loose']:
        D0 = wells_docs(LV, filt)
        pt = collections.Counter(m for i, m, f in tokens(D0, 2))
        pool = (list(pt), list(pt.values()))
        for lev in ['act', 'die']:
            D = collapse_level(D0, lev)
            P(f'#### Wells {LV} {filt} {lev}: {len(D)} documents')
            R[f'{LV}_{filt}_{lev}'] = run_indus(f'{LV}/{filt}/{lev}', D, pool)
        if LV == 'seq_raw':
            D = collapse_level(D0, 'act')
            P(f'#### Wells {LV} {filt} act, per site (kinds with >= 20 tokens)')
            for site in ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro']:
                Ds = [d for d in D if d['site'] == site]
                R[f'site_{filt}_{site}'] = run_indus(f'{site} {filt}', Ds, pool)
            Dn = [d for d in D if not d['big']]
            P(f'#### Wells {LV} {filt} act, all sites except Mohenjo-daro and Harappa')
            R[f'held_{filt}'] = run_indus(f'non-MD/H {filt}', Dn, pool)
for filt in ['strict', 'loose']:
    I = im77_docs(filt)
    pt = collections.Counter(m for i, m, f in tokens(I, 2)); pool = (list(pt), list(pt.values()))
    P(f'#### IM77 {filt} (sides; sealing includes Harappa moulded tablets recorded as sealings)')
    R[f'im77_{filt}'] = run_indus(f'IM77 {filt}', I, pool)
    for site in ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan']:
        R[f'im77_{filt}_{site}'] = run_indus(f'IM77 {filt} {site}', [d for d in I if d['site'] == site], pool)
json.dump(R, open(DARK + 'loop76_c1.json', 'w'), indent=1, default=str)
save('loop76_c1')
