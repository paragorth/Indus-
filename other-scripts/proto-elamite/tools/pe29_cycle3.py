"""pe29 cycle 3: do the rebuilt sessions behave like daily archives?  None of the five evidences reads a numeral, so
amounts are an independent check.
 (a) same name, same amount: for entry strings (>= 2 base signs) that recur on two tablets, the share written with the
     identical numeral, session pairs vs all other pairs (conditional on the string recurring, so not circular with 'pool').
 (b) rates / scale: |log median entry value| difference (same system) for session pairs vs null pairs.
 (c) cross-tablet arithmetic: a written total (lone numeric line off the obverse) on A equals the entry sum of B, the
     total of B, or the sum of two other tablets' entry sums, inside a session; values >= 6 only.
Nulls: N1 sessions relabelled inside strata (museum prefix x system x line bin), same sizes; N2 'drawer' sessions:
each session replaced by a random tablet and its nearest museum-number neighbours (same size).  Planted control:
in 25 random sessions one tablet's total is overwritten by another member's entry sum.
usage: pe29_cycle3.py [C file] [thr] [NNULL]
"""
import sys, json, itertools, collections, copy
import numpy as np
from pe29_common import *
from pe29_cycle1 import strata, nlbin
from common import entries as _entries

CF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CK, 'C_PE.npy')
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(293)
MODE = sys.argv[4] if len(sys.argv) > 4 else 'sum'   # 'sum': total = another tablet's entry sum (or two); 'tot': total = total; 'all'
SEX = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
CAP = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720}
BV = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 120, 'N48': 1200, 'N51': 1, 'N54': 10, 'N46': 120}


def val(nums):
    sy = system_of(nums)
    tab = {'SDB': SEX, 'C': CAP, 'B': BV}.get(sy)
    if not tab:
        return None, None
    v = 0
    for c, code in nums:
        if code not in tab or not isinstance(c, int):
            return None, None
        v += c * tab[code]
    return sy, v


def tablet_numbers(t):
    lines = t['lines']
    off = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
    tot = None; ent = []
    for i, l in enumerate(lines):
        if not l['numerals']:
            continue
        sy, v = val(l['numerals'])
        clean = not l['lacuna'] and 'x' not in l['signs'] and not l.get('damaged')
        if l['surface'] != 'obverse' and len(off) == 1 and l is off[0]:
            if sy and clean:
                tot = (sy, v)
            continue
        if i == 0:
            continue
        sg = tuple(base(s) for s in l['signs'] if is_sign(s))
        ent.append(dict(sg=sg, sy=sy, v=v, clean=clean and sy is not None))
    esum = {}
    if ent and all(e['clean'] for e in ent):
        for e in ent:
            esum[e['sy']] = esum.get(e['sy'], 0) + e['v']
    return dict(tot=tot, ent=ent, esum=esum)


def sessions_from(C, thr):
    lab = consensus_partition(C, thr)
    return lab


def pairs_of(lab):
    g = collections.defaultdict(list)
    for i, x in enumerate(lab):
        g[x].append(i)
    P = set()
    for mem in g.values():
        if 1 < len(mem) <= 12:
            for a, b in itertools.combinations(sorted(mem), 2):
                P.add((a, b))
    return P, g


def stat_a(P, NUM):
    same = diff = 0
    for a, b in P:
        ea = collections.defaultdict(set)
        for e in NUM[a]['ent']:
            if len(e['sg']) >= 2 and e['clean']:
                ea[e['sg']].add((e['sy'], e['v']))
        for e in NUM[b]['ent']:
            if len(e['sg']) >= 2 and e['clean'] and e['sg'] in ea:
                if (e['sy'], e['v']) in ea[e['sg']]:
                    same += 1
                else:
                    diff += 1
    return same, diff


def stat_b(P, MED):
    d = []
    for a, b in P:
        for sy in set(MED[a]) & set(MED[b]):
            d.append(abs(np.log(MED[a][sy]) - np.log(MED[b][sy])))
    return float(np.mean(d)) if d else float('nan'), len(d)


def stat_c(g, NUM):
    hits = 0; cand = 0; ex = []
    for mem in g.values():
        if not (1 < len(mem) <= 12):
            continue
        for a in mem:
            tA = NUM[a]['tot']
            if not tA or tA[1] < 6:
                continue
            others = [b for b in mem if b != a]
            vals = []
            for b in others:
                if tA[0] in NUM[b]['esum']:
                    vals.append((b, 'sum', NUM[b]['esum'][tA[0]]))
                if NUM[b]['tot'] and NUM[b]['tot'][0] == tA[0]:
                    vals.append((b, 'tot', NUM[b]['tot'][1]))
            cand += 1
            hit = [x for x in vals if x[2] == tA[1] and (MODE == 'all' or x[1] == MODE)]
            es = [x for x in vals if x[1] == 'sum']
            for x, y in (itertools.combinations(es, 2) if MODE != 'tot' else []):
                if x[0] != y[0] and x[2] + y[2] == tA[1]:
                    hit.append((x[0], y[0], 'sum2', tA[1]))
            if hit:
                hits += 1; ex.append((a, hit[:2]))
    return hits, cand, ex


def drawer_sessions(lab, T, rng):
    g = collections.defaultdict(list)
    for i, x in enumerate(lab):
        g[x].append(i)
    bypre = collections.defaultdict(list)
    for i, t in enumerate(T):
        if t['no'] is not None:
            bypre[t['pre']].append(i)
    for k in bypre:
        bypre[k].sort(key=lambda i: T[i]['no'])
    allidx = [i for k in bypre for i in bypre[k]]
    new = np.arange(len(T)) + 10 ** 6; used = set(); c = 0
    for mem in g.values():
        m = len(mem)
        if m < 2:
            continue
        for _ in range(50):
            s = allidx[rng.integers(len(allidx))]; L = bypre[T[s]['pre']]; j = L.index(s)
            blk = L[j:j + m]
            if len(blk) == m and not (set(blk) & used):
                break
        used |= set(blk)
        for i in blk:
            new[i] = c
        c += 1
    return new


if __name__ == '__main__':
    T = pe_tablets()
    NUM = [tablet_numbers(t) for t in T]
    MED = []
    for x in NUM:
        d = collections.defaultdict(list)
        for e in x['ent']:
            if e['clean'] and e['v']:
                d[e['sy']].append(e['v'])
        MED.append({k: float(np.median(v)) for k, v in d.items() if len(v) >= 2})
    C = np.load(CF).astype(np.float32)
    lab = sessions_from(C, THR)
    P, g = pairs_of(lab)
    sz = collections.Counter(len(m) for m in g.values())
    print('sessions: sizes', sorted(sz.items())[:15], 'pairs', len(P), flush=True)
    st, _ = strata(T)
    obs = dict(a=stat_a(P, NUM), b=stat_b(P, MED), c=stat_c(g, NUM)[:2])
    print('observed', obs, flush=True)
    # all-pairs baseline for (a): share of identical amount among recurring strings anywhere
    res = {'obs': obs, 'N1': [], 'N2': []}
    for name in ('N1', 'N2'):
        for r in range(NN):
            if name == 'N1':
                p = np.arange(len(T)); gg = collections.defaultdict(list)
                for i, x in enumerate(st):
                    gg[x].append(i)
                for idx in gg.values():
                    p[idx] = rng.permutation(idx)
                l2 = lab[p]
            else:
                l2 = drawer_sessions(lab, T, rng)
            P2, g2 = pairs_of(l2)
            res[name].append(dict(a=stat_a(P2, NUM), b=stat_b(P2, MED)[0], c=stat_c(g2, NUM)[:2]))
        A = np.array([x['a'][0] / sum(x['a']) if sum(x['a']) else np.nan for x in res[name]]); A = A[~np.isnan(A)]
        Bv = np.array([x['b'] for x in res[name]]); Cc = np.array([x['c'][0] for x in res[name]])
        oa = obs['a'][0] / max(1, sum(obs['a']))
        print(name, 'a: identical-amount share obs %.3f (%d/%d) null %.3f q95 %.3f p %.3f' % (oa, obs['a'][0], sum(obs['a']), A.mean() if len(A) else np.nan, np.quantile(A, .95) if len(A) else np.nan, (1 + (A >= oa).sum()) / (len(A) + 1)))
        print(name, 'b: |dlog median| obs %.3f null %.3f q05 %.3f p %.3f' % (obs['b'][0], np.nanmean(Bv), np.nanquantile(Bv, .05), (1 + (Bv <= obs['b'][0]).sum()) / (NN + 1)))
        print(name, 'c: totals matched obs %d of %d candidates; null %.2f q95 %.1f p %.3f' % (obs['c'][0], obs['c'][1], Cc.mean(), np.quantile(Cc, .95), (1 + (Cc >= obs['c'][0]).sum()) / (NN + 1)), flush=True)
        res[name + '_sum'] = dict(a_null=float(A.mean()) if len(A) else None, a_n=len(A), a_p=float((1 + (A >= oa).sum()) / (len(A) + 1)), b_null=float(np.nanmean(Bv)),
                                  b_p=float((1 + (Bv <= obs['b'][0]).sum()) / (NN + 1)), c_null=float(Cc.mean()), c_p=float((1 + (Cc >= obs['c'][0]).sum()) / (NN + 1)))
    hits, cand, ex = stat_c(g, NUM)
    res['examples'] = [(T[a]['id'], [(T[h[0]]['id'],) + tuple(h[1:]) if isinstance(h[0], (int, np.integer)) else h for h in hs]) for a, hs in ex]
    print('examples', res['examples'][:10])
    # planted arithmetic control
    NP = copy.deepcopy(NUM); planted = 0
    keys = [k for k, m in g.items() if 1 < len(m) <= 12]
    rng.shuffle(keys)
    for k in keys:
        mem = g[k]
        src = [b for b in mem if NP[b]['esum']]
        if not src:
            continue
        b = src[0]; a = [x for x in mem if x != b][0]
        sy, v = next(iter(NP[b]['esum'].items()))
        if v < 6:
            continue
        NP[a]['tot'] = (sy, v); planted += 1
        if planted == 25:
            break
    ph = stat_c(g, NP)[0]
    nullp = [stat_c(pairs_of(drawer_sessions(lab, T, rng))[1], NP)[0] for _ in range(200)]
    print('PLANT c: planted %d; hits %d; drawer-null on planted data %.2f q95 %.1f' % (planted, ph, np.mean(nullp), np.quantile(nullp, .95)))
    res['plant'] = dict(planted=planted, hits=ph, null=float(np.mean(nullp)))
    # session listing for the report
    big = sorted([m for m in g.values() if 3 <= len(m) <= 12], key=len, reverse=True)[:15]
    res['top_sessions'] = [[(T[i]['id'], T[i]['pre'], T[i]['no'], T[i]['pub']) for i in m] for m in big]
    for m in big[:8]:
        print([(T[i]['id'], T[i]['pre'], T[i]['no'], T[i]['sys'], T[i]['nl']) for i in m])
    json.dump(res, open(os.path.join(CK, 'cycle3_' + os.path.basename(CF).replace('.npy', '') + f'_{THR}_{MODE}.json'), 'w'), indent=1, default=str)
