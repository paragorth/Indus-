"""pe50 cycle 2: missing tablet types (do the primaries' entities get recaptured on big tablets?).

A  Recapture index I = R_sb / sqrt(R_ss * R_bb), R_xy = observed / shuffle-expected entity
   recaptures between small (<= 3 entries) and big (>= 10 entries) tablets.  Shuffle null:
   entity tokens redealt across tablets, set sizes kept (100x).  Clay summaries of primaries
   predict I > 1; big tablets unrelated to primaries predict I ~ 1.
   Controls: planted archives (A: clay summaries of every 8 primaries; B: big lists drawn from
   the office pool, no summaries; MIX: half the offices summarised), thinned to PE size;
   Ur III Drehem/Umma full and PE-sized (1,585 random tablets, 10x).
B  Per-class scan: for entities of each class seen on small tablets, recaptures on big tablets vs
   shuffle expectation; class flagged 'summarised elsewhere' if obs/exp <= 0.33 with Poisson
   p < 0.01.  Plant MIX must flag the unsummarised offices and not the others.
C  Herd office two-sample survival: account blocks with a surviving single-herd primary, and
   single-herd primaries found in a surviving account (exact binomial intervals).
Output data/pe50_ckpt/cycle2.json
"""
import json, os, sys, random, math, collections
from multiprocessing import Pool
import numpy as np
from scipy.stats import poisson, beta as _beta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, '..', 'data', 'pe50_ckpt')
D = json.load(open(os.path.join(CK, 'caps.json')))
CAPS, META = D['caps'], D['meta']


def cls_of(n):
    return 's' if n <= 3 else 'b' if n >= 10 else 'm'


def overlaps(inc, cl):
    cnt = collections.defaultdict(lambda: [0, 0])
    for s, c in zip(inc, cl):
        if c == 'm':
            continue
        j = 0 if c == 's' else 1
        for e in s:
            cnt[e][j] += 1
    sb = sum(a * b for a, b in cnt.values())
    ss = sum(a * (a - 1) / 2 for a, b in cnt.values())
    bb = sum(b * (b - 1) / 2 for a, b in cnt.values())
    return sb, ss, bb


def shuffled(inc, rng):
    pool = [e for s in inc for e in s]
    rng.shuffle(pool)
    out, k = [], 0
    for s in inc:
        out.append(set(pool[k:k + len(s)]))
        k += len(s)
    return out


def index(inc, cl, rng, nsh=60):
    o = overlaps(inc, cl)
    ns = np.array([overlaps(shuffled(inc, rng), cl) for _ in range(nsh)], float)
    e = ns.mean(0)
    R = [o[i] / e[i] if e[i] > 0 else float('nan') for i in range(3)]
    I = R[0] / math.sqrt(R[1] * R[2]) if R[1] > 0 and R[2] > 0 else float('nan')
    # null distribution of I from shuffles (each shuffle against the mean)
    In = [(x[0] / e[0]) / math.sqrt((x[1] / e[1]) * (x[2] / e[2])) for x in ns if x[1] > 0 and x[2] > 0]
    return {'obs': o, 'exp': e.tolist(), 'R_sb': R[0], 'R_ss': R[1], 'R_bb': R[2], 'I': I,
            'I_null_hi': float(np.percentile(In, 97.5)) if In else None, 'n_s': cl.count('s'), 'n_b': cl.count('b')}


def class_scan(inc, cl, ecls, rng, nsh=60):
    """ecls: entity -> class.  returns per class (n_entities_on_small, obs_on_big, exp_on_big)."""
    def stat(I):
        onS, onB = set(), set()
        for s, c in zip(I, cl):
            if c == 's':
                onS |= s
            elif c == 'b':
                onB |= s
        r = collections.Counter()
        n = collections.Counter()
        for e in onS:
            n[ecls.get(e, '?')] += 1
            if e in onB:
                r[ecls.get(e, '?')] += 1
        return n, r
    n, r = stat(inc)
    ex = collections.Counter()
    for _ in range(nsh):
        _, rr = stat(shuffled(inc, rng))
        for k, v in rr.items():
            ex[k] += v / nsh
    out = {}
    for k in n:
        e = ex.get(k, 0)
        p = float(poisson.cdf(r.get(k, 0), e)) if e > 0 else 1.0
        out[k] = {'n_small': n[k], 'obs_big': r.get(k, 0), 'exp_big': e, 'p_low': p,
                  'flag': bool(e >= 3 and r.get(k, 0) / e <= 0.33 and p < 0.01)}
    return out


def plant(rng, mode, O=20, Npool=150, nprim=12000, K=8):
    nr = np.random.default_rng(rng.randrange(1 << 30))
    act = {o: np.exp(nr.normal(0, 1.0, Npool)) for o in range(O)}
    tabs, cl, ecls = [], [], {}
    summarised = {o: (mode == 'A' or (mode == 'MIX' and o % 2 == 0)) for o in range(O)}
    for o in range(O):
        for i in range(Npool):
            ecls[(o, i)] = 'off%02d%s' % (o, 'S' if summarised[o] else 'U')
    per = nprim // O
    for o in range(O):
        p = act[o] / act[o].sum()
        buf = []
        for j in range(per):
            k = 1 + nr.poisson(1.0)
            s = {(o, int(x)) for x in nr.choice(Npool, size=min(k, 3), replace=False, p=p)}
            tabs.append(s)
            cl.append(1)
            buf.append(s)
            if len(buf) == K:
                if summarised[o]:
                    u = set().union(*buf)
                    tabs.append(u)
                    cl.append(len(u))
                buf = []
        if not summarised[o]:
            for j in range(per // K):
                k = 10 + nr.poisson(4)
                tabs.append({(o, int(x)) for x in nr.choice(Npool, size=k, replace=False, p=p)})
                cl.append(k)
    return tabs, cl, ecls


def job_plant(args):
    mode, rep = args
    rng = random.Random(hash((mode, rep)) & 0xffffffff)
    tabs, sizes, ecls = plant(rng, mode)
    idx = rng.sample(range(len(tabs)), 1585)
    inc = [tabs[i] for i in idx]
    cl = [cls_of(len(tabs[i])) for i in idx]
    r = index(inc, cl, rng, 40)
    sc = class_scan(inc, cl, ecls, rng, 40)
    flagged = sorted(k for k, v in sc.items() if v['flag'])
    U = sorted(k for k in sc if k.endswith('U'))
    S = sorted(k for k in sc if k.endswith('S'))
    r['flag_U'] = sum(k in flagged for k in U)
    r['flag_S'] = sum(k in flagged for k in S)
    r['nU'], r['nS'] = len(U), len(S)
    return mode, rep, r


def ur3_classes(arch):
    """tablet class for Ur III = first word after the numeral in the first entry line (the
    commodity word; transaction verbs are on unnumbered lines that the capture file does not keep);
    entity class = commonest class of the tablets it is on."""
    U = json.load(open(os.path.join(CK, 'ur3_caps.json')))[arch]
    tcl = []
    for t in U:
        w = [y for y in (t['entries'][0].split() if t['entries'] else []) if not y[0].isdigit()]
        tcl.append(w[0] if w else 'none')
    c = collections.Counter(tcl)
    keep = {k for k, v in c.most_common(12)}
    return [x if x in keep else 'other' for x in tcl]


def job_ur3(args):
    arch, d, rep = args
    rng = random.Random(hash((arch, d, rep)) & 0xffffffff)
    inc_all = [set(s) for s in CAPS[arch][d]]
    nn = [m['nnum'] for m in META[arch]]
    tcl = UR3CL[arch]
    if rep == 'full':
        idx = list(range(len(inc_all)))
    else:
        idx = rng.sample(range(len(inc_all)), 1585)
    inc = [inc_all[i] for i in idx]
    cl = [cls_of(nn[i]) for i in idx]
    r = index(inc, cl, rng, 20 if rep == 'full' else 40)
    ec = collections.defaultdict(collections.Counter)
    for i in idx:
        for e in inc_all[i]:
            ec[e][tcl[i]] += 1
    ecls = {e: c.most_common(1)[0][0] for e, c in ec.items()}
    r['scan'] = class_scan(inc, cl, ecls, rng, 20 if rep == 'full' else 40)
    return arch, d, rep, r


def job_pe(d):
    rng = random.Random(7)
    inc = [set(s) for s in CAPS['PE'][d]]
    cl = [cls_of(m['nent']) for m in META['PE']]
    r = index(inc, cl, rng, 200)
    # entity class = final class sign of the entry (from FULL strings) or header for HDR
    ecls = {}
    if d in ('FULL2', 'DENT'):
        for s in inc:
            for e in s:
                x = json.loads(e)
                sig = x[0] if d == 'DENT' else x
                ecls[e] = sig[-1]
    elif d == 'MID2':
        full = [set(CAPS['PE']['FULL2'][i]) for i in range(len(inc))]
        for s, f in zip(inc, full):
            for e in s:
                ecls.setdefault(e, 'mid')
    hdr = [m['hdr'] or '-' for m in META['PE']]
    ecls_h = {}
    for s, h in zip(inc, hdr):
        for e in s:
            ecls_h.setdefault(e, h)
    r['scan_final'] = class_scan(inc, cl, ecls, rng, 200) if ecls else {}
    r['scan_hdr'] = class_scan(inc, cl, ecls_h, rng, 200)
    return d, r


UR3CL = {}
if __name__ == '__main__':
    for a in ('DREHEM', 'UMMA'):
        UR3CL[a] = ur3_classes(a)
    out = {}
    with Pool(2) as P:
        out['pe'] = P.map(job_pe, ['MID2', 'FULL2', 'DENT', 'DIRTY2'])
        print('pe done', flush=True)
        out['plant'] = P.map(job_plant, [(m, r) for m in ('A', 'B', 'MIX') for r in range(10)])
        print('plant done', flush=True)
        out['ur3'] = P.map(job_ur3, [(a, d, r) for a in ('DREHEM', 'UMMA') for d in ('OFF', 'DENT')
                                     for r in ['full'] + list(range(8))])
        print('ur3 done', flush=True)
    json.dump(out, open(os.path.join(CK, 'cycle2.json'), 'w'), default=float)
