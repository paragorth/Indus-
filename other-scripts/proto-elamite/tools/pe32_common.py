"""pe32 WHO GETS THE STANDARD RATION: shared code.

The pe28 class C14 = the 14 final signs of count lines that precede an M288 line holding exactly the standard
allotment 2(N39B) 1(N24) = 60 N39C per unit.  Here:
  table()        per tablet: lines with base signs, final sign, system (CNT / CAP / B / -), count, capacity value
  m288_events()  every line whose final sign is M288 (base), with its predecessor line, the predecessor's final
                 sign and count, the M288 amount in N39C, and whether pe27/pe28 could have used the pair ('seen')
  sign_features() per sign: feature families (position, left neighbour, system of lines it ends, count size,
                 header, next-line system, herd tablets, pe15 office, tablet set), lines defining the class
                 (predecessor -> standard M288) excluded so the test is not circular
  coherence()    mean pairwise similarity of a sign set per family
No sign readings from anyone are used.  Capacity values: a-priori set (N39C 1, N30D 2, N30C 4, N24 12, N39B 24,
N01 120, N14 720 in M288 lines; pe27 F5).
"""
import os, sys, json, re, hashlib
from fractions import Fraction as Fr
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa
from pe28_common import pe_seqs  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe32_ckpt')
os.makedirs(CK, exist_ok=True)

C14 = ['M054', 'M388', 'M371', 'M057', 'M124', 'M128', 'M218', 'M220', 'M228', 'M230', 'M301', 'M066',
       'M320', 'M370']
OFFICE = {'GRAIN': ['M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'],
          'CLASS': ['M387', 'M388', 'M218', 'M124', 'M009', 'M066', 'M057'],
          'BARE': ['M054', 'M367', 'M001', 'M370', 'M032', '|M036+1(N30D)|', 'M206', 'M269', 'M059', 'M102']}
OFF_OF = {s: o for o, L in OFFICE.items() for s in L}
CAPC = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720}
CAPCODES = {'N39C', 'N30D', 'N30C', 'N24', 'N39B'}
SEX = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
STD = 60


def sig(s):
    return base(s.rstrip('#?!').replace('[', '').replace(']', '').replace('<', '').replace('>', ''))


def numok(l):
    tail = l['raw'].split(',')[-1]
    if '[' in tail or '...' in tail or '?' in tail:
        return False
    return bool(l['numerals']) and all(isinstance(n, int) and n > 0 and not str(c).startswith('n')
                                       for n, c in l['numerals'])


def table():
    out = []
    for t in load():
        if t['object_type'] != 'tablet':
            continue
        L = []
        for i, l in enumerate(t['lines']):
            sg = [sig(s) for s in l['signs'] if is_sign(s.lstrip('[<'))]
            codes = {c.split('@')[0] for _, c in l['numerals']}
            sysn, cnt, cap = '-', None, None
            if l['numerals']:
                if codes & CAPCODES:
                    sysn = 'CAP'
                elif codes & {'N51', 'N54', 'N46', 'N51G', 'N54G'}:
                    sysn = 'B'
                elif codes <= set(SEX):
                    sysn = 'CNT'
                else:
                    sysn = 'OTH'
                ok = numok(l)
                if ok and codes <= set(SEX) and not any('@' in c for _, c in l['numerals']):
                    cnt = sum(n * SEX[c] for n, c in l['numerals'])
                if ok and codes <= set(CAPC) and not any('@' in c for _, c in l['numerals']):
                    cap = sum(n * CAPC[c] for n, c in l['numerals'])
            L.append({'i': i, 'sg': sg, 'fin': sg[-1] if sg else '-', 'sys': sysn, 'cnt': cnt, 'cap': cap,
                      'raw': l['raw'], 'surf': l['surface']})
        hdr = L[0]['sg'][0] if L and L[0]['sg'] else '-'
        out.append({'id': t['id'], 'hdr': hdr, 'L': L, 'site': t['provenience'].split(' (')[0]})
    return out


def seen_pairs():
    """(tablet, M288 line) pairs that pe27 / pe28 could see: both lines in pe_seqs, adjacent in it."""
    S = pe_seqs()
    seen = set()
    for tid, L in S:
        for j in range(1, len(L)):
            if base(L[j][4]) == 'M288':
                seen.add((tid, L[j][0]))
    return seen


def m288_events(T=None, seen=None):
    T = T or table()
    seen = seen if seen is not None else seen_pairs()
    E = []
    for t in T:
        L = t['L']
        for k in range(1, len(L)):
            l = L[k]
            if l['fin'] != 'M288' or l['cap'] is None:
                continue
            p = L[k - 1]
            if p['fin'] in ('-', 'x'):
                continue
            E.append({'tid': t['id'], 'line': l['i'], 'pfin': p['fin'], 'x': p['cnt'] if p['sys'] == 'CNT' else None,
                      'psys': p['sys'], 'y': l['cap'], 'seen': (t['id'], l['i']) in seen,
                      'bare': len(l['sg']) == 1, 'praw': p['raw'], 'raw': l['raw']})
    return E


def is_std(e, mult=(1,)):
    if e['x']:
        return any(e['y'] == STD * m * e['x'] for m in mult)
    return any(e['y'] == STD * m for m in mult)


# ------------------------------------------------------------------ features
FAMS = ['pos', 'left', 'sys', 'size', 'hdr', 'next', 'herd', 'office', 'tabs']


def sign_features(T, exclude=frozenset(), top=150):
    """exclude: set of (tablet, line index) pairs whose lines are skipped (class-defining lines)."""
    pos = defaultdict(Counter); left = defaultdict(Counter); sysd = defaultdict(Counter)
    size = defaultdict(Counter); hdr = defaultdict(Counter); nxt = defaultdict(Counter)
    herd = defaultdict(lambda: [0, 0]); tabs = defaultdict(set); fin_n = Counter()
    herd_tabs = {t['id'] for t in T if any(any(s == 'M362' or s.startswith('|M362') for s in l['sg'])
                                             for l in t['L'])}
    for t in T:
        L = t['L']
        for k, l in enumerate(L):
            if (t['id'], l['i']) in exclude:
                continue
            sg = l['sg']
            for j, s in enumerate(sg):
                p = 'sole' if len(sg) == 1 else 'ini' if j == 0 else 'fin' if j == len(sg) - 1 else 'mid'
                pos[s][p] += 1
                tabs[s].add(t['id'])
                herd[s][0] += 1; herd[s][1] += t['id'] in herd_tabs
            if not sg or k == 0:
                continue
            f = sg[-1]
            fin_n[f] += 1
            left[f][sg[-2] if len(sg) > 1 else '^'] += 1
            sysd[f][l['sys']] += 1
            if l['cnt'] is not None:
                c = l['cnt']
                size[f]['1' if c == 1 else '2-4' if c < 5 else '5-9' if c < 10 else '10+'] += 1
            hdr[f][t['hdr']] += 1
            if k + 1 < len(L):
                nl = L[k + 1]
                if nl['fin'] != 'M288':          # the M288 successor is what defines the class: never a feature
                    nxt[f][nl['sys']] += 1
    return {'pos': pos, 'left': left, 'sys': sysd, 'size': size, 'hdr': hdr, 'next': nxt, 'herd': herd,
            'tabs': tabs, 'fin_n': fin_n}


def _jsd_sim(a, b, alpha=0.5):
    keys = set(a) | set(b)
    if not keys:
        return np.nan
    va = np.array([a.get(k, 0) for k in keys], float) + alpha
    vb = np.array([b.get(k, 0) for k in keys], float) + alpha
    if va.sum() <= alpha * len(keys) or vb.sum() <= alpha * len(keys):
        return np.nan
    pa, pb = va / va.sum(), vb / vb.sum()
    m = (pa + pb) / 2
    j = 0.5 * (pa * np.log2(pa / m)).sum() + 0.5 * (pb * np.log2(pb / m)).sum()
    return 1 - j


def pair_sim(F, a, b, fam):
    if fam in ('pos', 'left', 'sys', 'size', 'hdr', 'next'):
        return _jsd_sim(F[fam].get(a, {}), F[fam].get(b, {}))
    if fam == 'herd':
        ha, hb = F['herd'].get(a), F['herd'].get(b)
        if not ha or not hb or not ha[0] or not hb[0]:
            return np.nan
        return 1 - abs(ha[1] / ha[0] - hb[1] / hb[0])
    if fam == 'office':
        return float(OFF_OF.get(a, 'none') == OFF_OF.get(b, 'none') and a in OFF_OF)
    if fam == 'tabs':
        A, B = F['tabs'].get(a, set()), F['tabs'].get(b, set())
        return len(A & B) / max(1, len(A | B))


def sim_matrix(F, signs):
    """per family: matrix of pairwise similarities among `signs` (cached use)."""
    n = len(signs)
    M = {}
    for fam in FAMS:
        X = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                X[i, j] = X[j, i] = pair_sim(F, signs[i], signs[j], fam)
        M[fam] = X
    return M


def coherence(M, idx):
    idx = np.asarray(idx)
    out = {}
    for fam, X in M.items():
        sub = X[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        v = sub[iu]
        out[fam] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return out


def freq_matched(rng, members, pool, freq, bins=None):
    """for each member draw a sign from pool with the same log2 final-frequency bin (no repeats)."""
    lb = {s: int(np.log2(max(freq.get(s, 1), 1))) for s in pool}
    byb = defaultdict(list)
    for s in pool:
        byb[lb[s]].append(s)
    out, used = [], set()
    for m in members:
        b = int(np.log2(max(freq.get(m, 1), 1)))
        cand = []
        for d in (0, 1, -1, 2, -2, 3, -3):
            cand = [s for s in byb.get(b + d, []) if s not in used]
            if cand:
                break
        s = cand[rng.integers(len(cand))]
        used.add(s); out.append(s)
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
