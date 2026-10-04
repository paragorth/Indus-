"""pe34 shared code: compound parsing, type behaviour vectors, transfer test,
planted compositional systems."""
import json, os, re, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe34_ckpt')


def tokens():
    return json.load(open(os.path.join(CK, 'tokens.json')))


def strip_var(s):
    return re.sub(r'~[A-Za-z0-9]+', '', s)


def top_split(s, ops):
    """split at depth-0 operators; returns parts and operators."""
    parts, opl, depth, cur = [], [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth == 0 and ch in ops:
            parts.append(cur)
            opl.append(ch)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts, opl


def parse(sign, corpus):
    """-> (base, modifier) or None for simple signs."""
    if corpus == 'LINB':
        m = re.match(r'^([A-Z*][A-Z0-9*]*)(;[0-9x]+)?(:[mf]|\+[A-Z*0-9]+)$', sign)
        if not m:
            m2 = re.match(r'^([A-Z*][A-Z0-9*]*);([0-9x]+)$', sign)
            return (m2.group(1), ';' + m2.group(2)) if m2 else None
        b = m.group(1) + (m.group(2) or '')
        return (b, m.group(3), m.group(1))
    s = sign.strip('|')
    if corpus == 'PE':
        if '+' not in s:
            return None
        parts, _ = top_split(s, '+')
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return None
        a = parts[0]
        rest = [strip_var(p) for p in parts[1:]]
        if len(parts) == 3 and strip_var(parts[2]) == strip_var(a):
            mod = rest[0] + '@frame'
        elif len(parts) == 2 and rest[0] == strip_var(a):
            mod = '@dup'
        else:
            mod = '+'.join(rest)
        return (a, mod, strip_var(a))
    # ARCH
    if not sign.startswith('|'):
        return None
    parts, ops = top_split(s, 'x+.&%')
    if len(parts) < 2:
        return None
    a = parts[0]
    mod = ops[0] + strip_var(''.join(o + p for o, p in zip([''] + ops[1:], parts[1:])))
    return (a, mod, strip_var(a))


def prep(toks, corpus, min_c=2, min_b=3, extra=None):
    """Return dict with feature names, type means (z-scored features), counts,
    and the list of eligible compounds (c, base, mod)."""
    if extra:
        toks = toks + extra
    names = sorted({k for r in toks for k in r['f']})
    X = np.array([[r['f'].get(k, 0.0) for k in names] for r in toks], float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    idx = defaultdict(list)
    for i, r in enumerate(toks):
        idx[r['s']].append(i)
    mean = {s: X[v].mean(0) for s, v in idx.items()}
    n = {s: len(v) for s, v in idx.items()}
    comp = []
    for s in idx:
        p = parse(s, corpus)
        if not p or n[s] < min_c:
            continue
        a, mod = p[0], p[1]
        b = a if n.get(a, 0) >= min_b else (p[2] if len(p) > 2 and n.get(p[2], 0) >= min_b else None)
        if b is None or parse(b, corpus):
            continue
        comp.append((s, b, mod))
    return {'names': names, 'mean': mean, 'n': n, 'comp': comp, 'X': X, 'idx': idx}


def transfer(P, mods, exclude=('X',), targets=None):
    """Leave-one-compound-out prediction of each compound's behaviour.
    mods: list of modifier labels aligned with P['comp'].
    Returns dict of mean errors over targets (compounds whose modifier occurs on
    >= 1 other eligible compound with a different base)."""
    comp, M = P['comp'], P['mean']
    D = np.array([M[c] - M[b] for c, b, _ in comp])
    F = np.array([M[c] for c, _, _ in comp])
    B = np.array([M[b] for _, b, _ in comp])
    bases = [b for _, b, _ in comp]
    by = defaultdict(list)
    for i, m in enumerate(mods):
        if m not in exclude:
            by[m].append(i)
    gsum, nC = D.sum(0), len(comp)
    e = defaultdict(list)
    tl = []
    for i, (c, b, _) in enumerate(comp):
        m = mods[i]
        if m in exclude:
            continue
        if targets is not None and c not in targets:
            continue
        oth = [j for j in by[m] if j != i and bases[j] != b]
        if not oth:
            continue
        dm = D[oth].mean(0)
        gen = (gsum - D[i]) / (nC - 1)
        f = F[i]
        e['base'].append(((B[i] - f) ** 2).mean())
        e['add'].append(((B[i] + dm - f) ** 2).mean())
        e['gen'].append(((B[i] + gen - f) ** 2).mean())
        e['modonly'].append(((F[oth].mean(0) - f) ** 2).mean())
        tl.append(c)
    out = {k: float(np.mean(v)) for k, v in e.items()}
    out['n'] = len(tl)
    if tl:
        out['G'] = out['gen'] - out['add']       # modifier-specific gain over generic shift
        out['S'] = out['base'] - out['add']      # gain over base alone
    out['targets'] = tl
    return out


def perm_test(P, n_perm=2000, seed=0, exclude=('X',), targets=None, stat='G'):
    mods = [m for _, _, m in P['comp']]
    real = transfer(P, mods, exclude, targets)
    rng = random.Random(seed)
    keep = [i for i, m in enumerate(mods) if m not in exclude]
    null = []
    for _ in range(n_perm):
        sh = mods[:]
        vals = [mods[i] for i in keep]
        rng.shuffle(vals)
        for i, v in zip(keep, vals):
            sh[i] = v
        r = transfer(P, sh, exclude, targets)
        if r['n']:
            null.append(r[stat])
    null = np.array(null)
    p = (1 + (null >= real.get(stat, -1e9)).sum()) / (1 + len(null))
    return real, float(null.mean()), float(null.std()), float(p)


def base_test(P, n_perm=2000, seed=0):
    """Does the compound resemble its OWN base more than another compound's base?"""
    comp, M = P['comp'], P['mean']
    F = np.array([M[c] for c, _, _ in comp])
    bases = [b for _, b, _ in comp]
    def err(bl):
        return float(np.mean([((M[b] - F[i]) ** 2).mean() for i, b in enumerate(bl)]))
    real = err(bases)
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        sh = bases[:]
        rng.shuffle(sh)
        null.append(err(sh))
    null = np.array(null)
    return real, float(null.mean()), float((1 + (null <= real).sum()) / (1 + len(null)))


def plant(toks, corpus, seed, strength, n_mod=8, n_base=5, frac=0.3, n_feat=3):
    """Relabel a fraction of tokens of frequent simple signs as planted compounds
    |B+PLk| and push n_feat features of each planted modifier with prob strength.
    Returns (new token list, list of planted compound names, effects)."""
    rng = random.Random(seed)
    cnt = Counter(r['s'] for r in toks)
    simple = [s for s, c in cnt.items() if c >= 20 and not parse(s, corpus)]
    names = sorted({k for r in toks for k in r['f']})
    fams = defaultdict(list)
    for k in names:
        fams[k.split(':')[0] if ':' in k else k].append(k)
    famk = sorted(fams)
    eff = {}
    for k in range(n_mod):
        eff[k] = rng.sample(famk, n_feat)
        eff[k] = [(fam, rng.choice(fams[fam])) for fam in eff[k]]
    out = [dict(r) for r in toks]
    byS = defaultdict(list)
    for i, r in enumerate(out):
        byS[r['s']].append(i)
    planted = []
    for k in range(n_mod):
        for b in rng.sample(simple, n_base):
            ids = byS[b]
            pick = rng.sample(ids, max(3, int(frac * len(ids))))
            cname = '|%s+PL%d|' % (b, k) if corpus != 'LINB' else '%s+PL%d' % (b, k)
            if corpus == 'LINB':
                cname = '%s+PLQ%s' % (b, 'ABCDEFGHIJ'[k])
            planted.append(cname)
            for i in pick:
                f = dict(out[i]['f'])
                for fam, key in eff[k]:
                    if rng.random() >= strength:
                        continue
                    if fam == 'logq' or fam == 'logn':
                        f[fam] = f.get(fam, 0) + 1.5
                    elif ':' in key:
                        for kk in fams[fam]:
                            f[kk] = 0.0
                        f[key] = 1.0
                    else:
                        f[key] = 1.0 - f.get(key, 0.0) if f.get(key, 0.0) else 1.0
                out[i] = {'s': cname, 't': out[i]['t'], 'f': f}
    return out, planted, eff
