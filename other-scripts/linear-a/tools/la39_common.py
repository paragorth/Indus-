#!/usr/bin/env python3
"""LA-39 shared helpers: do ligature PARTS carry meaning?

Every commodity-logogram token (plain or ligatured) becomes a row
    (base, modifier, doc, feature vector)
with behaviour features that use no sound values and no outside readings:
quantity size, fraction, missing number, position in the document, what precedes it,
how many commodities the document lists, whether the document has a total word,
site, support, and quantity relative to the document's other quantities.

Linear A: corpus.json logograms.  Linear B control: DAMOS (sex markers :m/:f/:x on
animals, ligature adjuncts such as SUS+SI, OLE+PA, TELA+TE).
"""
import json, os, re, unicodedata
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la39_ckpt')
os.makedirs(CK, exist_ok=True)

FEATS = ['logq', 'frac', 'noq', 'relpos', 'afterword', 'ncom', 'total', 'site', 'support', 'relq']

# fraction letters seen in LA ligatures (written as capital Latin letters in the corpus)
FRAC_LETTERS = {'B', 'D', 'E', 'F', 'H', 'J', 'K', 'L', 'L2', 'L4', 'QIf'}
LA_TOTAL = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}


def _la_split(v):
    v = v.replace("'", '').strip('[]')
    if '[?]' in v or '[ ]' in v or '[' in v or ']' in v:
        return None
    parts = v.split('+')
    sex = ''
    mm = re.match(r'^([A-Z]{3,})([mf])$', parts[0])
    if mm:
        parts[0] = mm.group(1); sex = mm.group(2)
    if len(parts) == 1:
        return parts[0], sex
    if sex:
        return parts[0], '+'.join([sex] + parts[1:])
    base = parts[0]
    # MI+JA+X and QA2+[?]+X style: two-part bases are not split further (only MI+JA is clean)
    if parts[0] == 'MI' and len(parts) >= 3 and parts[1] == 'JA':
        return 'MI+JA', '+'.join(parts[2:])
    return base, '+'.join(parts[1:])


def la_rows(corpus=None):
    C = corpus or json.load(open(os.path.join(D, 'corpus.json')))
    rows = []
    for ins in C:
        T = [t for t in ins['tokens'] if t['t'] not in ('div',)]
        words = set('-'.join(t['s']) for t in T if t['t'] == 'word')
        total = int(bool(words & LA_TOTAL))
        logos = [i for i, t in enumerate(T) if t['t'] == 'logo']
        if not logos:
            continue
        bases = set()
        for i in logos:
            sp = _la_split(T[i]['v'])
            if sp: bases.add(sp[0])
        qs = [t['v'] for t in T if t['t'] == 'num' and t['v'] > 0]
        med = np.median(qs) if qs else 1.0
        for k, i in enumerate(logos):
            sp = _la_split(T[i]['v'])
            if not sp: continue
            q, fr = None, 0
            for u in T[i + 1:]:
                if u['t'] == 'nl': continue
                if u['t'] == 'num':
                    q = u['v']; fr = int(bool(u.get('frac'))); break
                break
            prev = None
            for u in T[:i][::-1]:
                if u['t'] == 'nl': continue
                prev = u['t']; break
            f = [np.log2(1 + q) if q is not None else 0.0, fr, int(q is None),
                 k / max(1, len(logos) - 1) if len(logos) > 1 else 0.5,
                 int(prev == 'word'), np.log2(len(bases)), total,
                 int(ins['site'] == 'Haghia Triada'), int(ins['support'] == 'Tablet'),
                 np.log2((q + 0.5) / med) if q else 0.0]
            rows.append((sp[0], sp[1], ins['id'], f))
    return rows


# ---------------- Linear B ----------------
LB_BASES = {'OVIS', 'CAP', 'SUS', 'BOS', 'EQU', 'TELA', 'OLE', 'OLIV', 'CYP', 'GRA', 'HORD', 'VIN',
            'ROTA', 'TUN', 'PYC', 'AROM', 'LANA', 'VIR', 'MUL', 'CERV', 'AES', 'FAR', 'NI', 'FIC'}
LB_TOTAL = {'to-so', 'to-sa', 'to-so-de', 'to-sa-de'}


def _strip(t):
    t = unicodedata.normalize('NFD', t)
    return ''.join(c for c in t if unicodedata.category(c) != 'Mn')


def _lb_logo(t):
    t = _strip(t).strip("[]⟦⟧'<>⌞⌟")
    if '[' in t or ']' in t: return None
    m = re.match(r'^\*?([A-Z±]+)(?:;[0-9x])?(?::([mfx]))?((?:\+[A-Z0-9*]+)*)$', t)
    if not m or m.group(1) not in LB_BASES: return None
    base = m.group(1); mod = []
    if m.group(2): mod.append(m.group(2))
    if m.group(3): mod += [x for x in m.group(3).split('+') if x]
    return base, '+'.join(mod)


def lb_rows():
    rows = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        txt = d.get('content') or ''
        site = (d.get('heading') or '').split()[0] if d.get('heading') else ''
        toks = [_strip(x) for x in txt.split()]
        toks = [x for x in toks if x not in (',', '/') and not re.match(r'^\.', x) and x != 'v.']
        words = set(x.strip('[]') for x in toks)
        total = int(bool(words & LB_TOTAL))
        lg = [(i, _lb_logo(x)) for i, x in enumerate(toks)]
        lg = [(i, s) for i, s in lg if s]
        if not lg: continue
        bases = set(s[0] for _, s in lg)
        nums = [int(x) for x in toks if x.isdigit()]
        med = np.median(nums) if nums else 1.0
        for k, (i, (b, m)) in enumerate(lg):
            q = None
            for u in toks[i + 1:i + 4]:
                u2 = u.strip('[]')
                if u2.isdigit(): q = int(u2); break
                if re.match(r'^[A-Z]$', u2): continue   # metric unit letters
                break
            prev = toks[i - 1] if i > 0 else ''
            pw = int(bool(re.match(r'^[a-z0-9]+(-[a-z0-9]+)+$', prev.strip('[]'))))
            f = [np.log2(1 + q) if q is not None else 0.0, 0, int(q is None),
                 k / max(1, len(lg) - 1) if len(lg) > 1 else 0.5, pw, np.log2(len(bases)), total,
                 int(site == 'KN'), 1, np.log2((q + 0.5) / med) if q else 0.0]
            rows.append((b, m, d['id'], f))
    return rows


LA_COM_BASES = {'OLE', 'GRA', 'CYP', 'VIN', 'VIR', 'HIDE', 'OLIV', 'CAP', 'TELA', 'AROM', 'MI+JA',
                '*304', '*401', '*188', '*306', '*412', '*316', '*330', '*418', '*414', '*416', '*402',
                '*406VAS', '*341', 'FIC'}


def la_commodity_rows():
    out = []
    for b, m, d, f in la_rows():
        b = 'OLIV' if b == '*OLIV' else b
        if b in LA_COM_BASES: out.append((b, m, d, f))
    return out


def to_arrays(rows, min_base_types=1):
    """standardize features; return dict with arrays."""
    X = np.array([r[3] for r in rows], float)
    sd = X.std(0); sd[sd == 0] = 1
    X = (X - X.mean(0)) / sd
    keep = X.std(0) > 0
    X = X[:, keep]
    feats = [f for f, k in zip(FEATS, keep) if k]
    base = np.array([r[0] for r in rows]); mod = np.array([r[1] for r in rows])
    docs = np.array([str(r[2]) for r in rows])
    return dict(X=X, base=base, mod=mod, doc=docs, feats=feats)


def mod_atoms(m):
    return [a for a in m.split('+') if a] if m else []


def transfer_score(A, splits, modlabel=None, Xover=None, min_tok=1, per_mod=None):
    """Held-out cross-base transfer of modifier effects.
    For each split (boolean mask over tokens, by document; train side True), for each ligature type
    t=(b,m) with tokens on the test side and m present on another base b' on the train side:
      r = mean_test(b,m) - mean_test(b, other types)       (observed modifier effect on b)
      d = mean over b' of [mean_train(b',m) - mean_train(b', other types)]   (transferred)
    Returns (mean cosine(r,d), gain fraction 1 - sum|r-d|^2/sum|r|^2, pairs per split).
    modlabel: optional array overriding A['mod'] (for shuffles / plants). Xover: overriding features.
    """
    X = A['X'] if Xover is None else Xover
    mod = A['mod'] if modlabel is None else modlabel
    base = A['base']
    keys = list(zip(base.tolist(), mod.tolist()))
    tl = sorted(set(keys)); ti = {t: k for k, t in enumerate(tl)}
    tcode = np.array([ti[k] for k in keys])
    bl = sorted(set(base.tolist())); bi = {b: k for k, b in enumerate(bl)}
    bcode = np.array([bi[b] for b in base.tolist()])
    t_base = np.array([bi[b] for b, _ in tl]); t_mod = [m for _, m in tl]
    by_mod = defaultdict(list)
    for k, m in enumerate(t_mod):
        if m: by_mod[m].append(k)
    nT, nB, F = len(tl), len(bl), X.shape[1]
    cos_all, g_num, g_den, npair = [], 0.0, 0.0, 0
    for tr in splits:
        res = {}
        for side, msk in (('tr', tr), ('te', ~tr)):
            TS = np.zeros((nT, F)); TN = np.bincount(tcode[msk], minlength=nT).astype(float)
            np.add.at(TS, tcode[msk], X[msk])
            BS = np.zeros((nB, F)); np.add.at(BS, t_base, TS); BN = np.bincount(t_base, weights=TN, minlength=nB)
            on = BN[t_base] - TN
            ok = (TN >= min_tok) & (on >= 1)
            eff = np.zeros((nT, F))
            eff[ok] = TS[ok] / TN[ok, None] - (BS[t_base[ok]] - TS[ok]) / on[ok, None]
            res[side] = (ok, eff, TN)
        oktr, efftr, _ = res['tr']; okte, effte, TNte = res['te']
        for m, ks in by_mod.items():
            ktr = [k for k in ks if oktr[k]]
            if not ktr: continue
            for k in ks:
                if not okte[k]: continue
                oth = [q for q in ktr if t_base[q] != t_base[k]]
                if not oth: continue
                r = effte[k]; d = efftr[oth].mean(0)
                nr, nd = np.linalg.norm(r), np.linalg.norm(d)
                if nr > 0 and nd > 0:
                    c = r @ d / nr / nd
                    cos_all.append(c)
                    if per_mod is not None: per_mod[m].append((bl[t_base[k]], c, TNte[k]))
                g_num += ((r - d) ** 2).sum(); g_den += (r ** 2).sum(); npair += 1
    if not cos_all: return 0.0, 0.0, 0
    return float(np.mean(cos_all)), float(1 - g_num / g_den), npair / len(splits)


def make_splits(A, n, rng):
    ud = np.unique(A['doc'])
    out = []
    for _ in range(n):
        tr = set(ud[rng.random(len(ud)) < 0.5])
        out.append(np.array([d in tr for d in A['doc']]))
    return out


def shuffle_mods(A, rng, tries=200):
    """Permute modifier labels among ligature TYPES (whole types keep their tokens),
    rejecting permutations that put the same label twice on one base."""
    types = sorted(set((b, m) for b, m in zip(A['base'], A['mod']) if m))
    labels = [m for _, m in types]
    for _ in range(tries):
        p = rng.permutation(len(labels))
        new = {t: labels[j] for t, j in zip(types, p)}
        seen = set(); ok = True
        for (b, _), m in new.items():
            if (b, m) in seen: ok = False; break
            seen.add((b, m))
        if ok: break
    return np.array([new.get((b, m), m) if m else '' for b, m in zip(A['base'], A['mod'])])


def shared_mods(A, min_bases=2):
    bm = defaultdict(set)
    for b, m in zip(A['base'], A['mod']):
        if m: bm[m].add(b)
    return {m: sorted(bs) for m, bs in bm.items() if len(bs) >= min_bases}
