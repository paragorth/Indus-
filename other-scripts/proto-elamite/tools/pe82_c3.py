"""pe82 cycle 3: WRITERS PROTECT WHAT IS WORTH STEALING.

Inverted question: which entries did the writer make tamper-evident?  An entry is PROTECTED when it sits in a run of
entries whose values are re-stated by a later line equal to their exact sum (a checksum).  If writers spent the
checksum on what mattered most, the counted things that are protected more often (beyond tablet length and archive)
are the heavier / dearer ones.

Calibration: Ur III (Umma, Girsu, Drehem; words made irrelevant - only the first word is the key and only the pe80
mass table, frozen by pe80 before it saw counts, says what it weighs).  Random protection definitions (run length,
side, contiguity, start-at-top, key position, stratification) are scored by Spearman(protection excess, log mass) on
two archives and re-tested on the third.  The surviving definitions are frozen and applied once to PE, where the
ruler is the pe80 frozen size ranking (data/pe80_size_ranking_frozen.json, sha256 b692200a...).

usage: python3 pe82_c3.py ur3 | pe | plant
"""
import os, sys, json, math, random, itertools, collections
import numpy as np
from scipy.stats import spearmanr
import pe82_common as pc
import pe80_common as p80

p80.CK = pc.CK  # rebuild Ur III cache in our own checkpoint folder
MASS = json.load(open(os.path.join(pc.DATA, 'pe80_mass_table_frozen.json')))['mass_kg']
RULER = {d['sign']: d['pred_log10_kg'] for d in json.load(open(os.path.join(pc.DATA, 'pe80_size_ranking_frozen.json')))['signs']}


def protected_lines(lines, H):
    """lines: list of dict(v, surf).  Returns set of protected entry indices and set of checksum-line indices."""
    prot, chk = set(), set()
    n = len(lines)
    for j in range(n):
        vj = lines[j]['v']
        if vj is None or vj < 2:
            continue
        if H['side'] == 'rev' and lines[j]['surf'] == 'obverse':
            continue
        starts = [0] if H['top'] else range(0, j)
        end = j - H['skip']
        for a in starts:
            run = list(range(a, end))
            if len(run) < H['minrun']:
                continue
            vs = [lines[i]['v'] for i in run]
            if any(v is None for v in vs):
                continue
            if sum(vs) == vj:
                prot.update(run); chk.add(j)
                break
    return prot, chk


def key_of(l, H):
    t = l['tok']
    if not t:
        return None
    return t[0] if H['key'] == 'first' else t[-1]


def score(tabs, H, keys, groups=None):
    """Per key: protected share minus stratum-expected share (strata = tablet size bin x archive)."""
    obs = collections.Counter(); exp = collections.Counter(); n = collections.Counter()
    strat_p = collections.Counter(); strat_n = collections.Counter()
    per = []
    for t in tabs:
        if groups is not None and t['grp'] not in groups:
            continue
        prot, chk = protected_lines(t['lines'], H)
        nb = min(len(t['lines']), 12) // H['sbin']
        st = (nb, t['grp']) if H['strat'] == 'size_arch' else (nb,)
        for i, l in enumerate(t['lines']):
            if i in chk or l['v'] is None:
                continue
            k = key_of(l, H)
            p = i in prot
            strat_p[st] += p; strat_n[st] += 1
            per.append((k, st, p))
    rate = {s: strat_p[s] / strat_n[s] for s in strat_n}
    for k, st, p in per:
        if k in keys:
            obs[k] += p; exp[k] += rate[st]; n[k] += 1
    ex = {k: (obs[k] - exp[k]) / n[k] for k in keys if n[k] >= H['minn']}
    return ex, sum(strat_p.values())


def rho(ex, ruler):
    ks = [k for k in ex if k in ruler]
    if len(ks) < 6:
        return float('nan'), len(ks)
    r = spearmanr([ex[k] for k in ks], [ruler[k] for k in ks]).correlation
    return float(r), len(ks)


def load_ur3():
    T = p80.build_ur3()
    for t in T:
        t['grp'] = t['prov']
    return T


def load_pe_tabs():
    P = p80.load_pe()
    for t in P:
        b = pc.PE_BATCH.get(t.get('vol'), 'OTHER') if t.get('site') == 'Susa' else 'OTHER'
        t['grp'] = 'A' if b in ('MDP26', 'MDP26S', 'OTHER') else 'B'
    return P


PROT_GRID = list(itertools.product(['any', 'rev'], [False, True], [0, 1], [2, 3, 4, 5]))
SCORE_GRID = list(itertools.product(['first', 'last'], ['size', 'size_arch'], [1, 2, 4], [5, 10, 20]))


def line_table(T, order_shuffle=None):
    """One record per (tablet, line): keys, size bin raw, grp, and a protected flag for each of the 32 protection rules.
    order_shuffle: random.Random -> lines permuted within tablet first (kill control: keeps values, breaks checksums)."""
    recs = []
    for t in T:
        L = list(t['lines'])
        if order_shuffle is not None:
            L = L[:]; order_shuffle.shuffle(L)
        flags = []
        for side, top, skip, minrun in PROT_GRID:
            flags.append(protected_lines(L, dict(side=side, top=top, skip=skip, minrun=minrun)))
        for i, l in enumerate(L):
            if l['v'] is None or not l['tok']:
                continue
            pf = [(-1 if i in c else int(i in p)) for p, c in flags]   # -1 = this line is the checksum itself
            recs.append((l['tok'][0], l['tok'][-1], len(L), t['grp'], pf))
    return recs


def score_tab(recs, pi, key, strat, sbin, minn, keys, groups=None):
    sp = collections.Counter(); sn = collections.Counter(); per = []
    for kf, kl, n, g, pf in recs:
        if groups is not None and g not in groups:
            continue
        p = pf[pi]
        if p < 0:
            continue
        st = (min(n, 12) // sbin, g) if strat == 'size_arch' else (min(n, 12) // sbin,)
        sp[st] += p; sn[st] += 1
        k = kf if key == 'first' else kl
        if k in keys:
            per.append((k, st, p))
    obs = collections.Counter(); exp = collections.Counter(); nn = collections.Counter()
    for k, st, p in per:
        obs[k] += p; exp[k] += sp[st] / sn[st]; nn[k] += 1
    return {k: (obs[k] - exp[k]) / nn[k] for k in nn if nn[k] >= minn}, sum(sp.values())


def all_scores(recs, keys, ruler, groups_list):
    out = []
    for pi, pg in enumerate(PROT_GRID):
        for sg in SCORE_GRID:
            row = dict(prot=pg, score=sg)
            for name, G in groups_list:
                ex, npro = score_tab(recs, pi, *sg, keys, G)
                row[name] = rho(ex, ruler) + (npro,)
            out.append(row)
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if mode in ('ur3', 'ur3kill'):
        T = load_ur3()
        mlog = {k: math.log10(v) for k, v in MASS.items()}
        recs = line_table(T, random.Random(seed) if mode == 'ur3kill' else None)
        res = all_scores(recs, set(MASS), mlog, [('Umma', {'Umma'}), ('Girsu', {'Girsu'}), ('all', None)])
        json.dump(res, open(os.path.join(pc.CK, 'c3_%s_%d.json' % (mode, seed)), 'w'))
        print(mode, seed, 'defs', len(res))
    if mode in ('pe', 'pekill'):
        T = load_pe_tabs()
        recs = line_table(T, random.Random(seed) if mode == 'pekill' else None)
        res = all_scores(recs, set(RULER), RULER, [('A', {'A'}), ('B', {'B'}), ('all', None)])
        json.dump(res, open(os.path.join(pc.CK, 'c3_%s_%d.json' % (mode, seed)), 'w'))
        print(mode, seed, 'defs', len(res))
