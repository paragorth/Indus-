#!/usr/bin/env python3
"""LA-72: the la70 v2 crack reading rebuilt on the restoration-aware corpus (data/corpus_ra.json, la71).

Versions (la71_parse): 'rd' = read + damaged tokens (default; restored and erased tokens become
['X', kind, section word] placeholders, so a section that loses a number is untestable), 'read' = read
tokens only (robustness check), 'all' = the legacy corpus.json (la70 as published).

Every document carries a parallel list 'st' (status of each token: read / damaged / restored / erased /
illegible / '-' for line breaks) so that cycle 3 can see which tokens a prediction rests on.
The la70 machinery (la70_common: gate, decoder, closures, shuffles, LB calibration) is reused unchanged;
cycle 1 adds train-only gates for the parts la70 fixed by hand or derived on all documents (KI test,
D/B values, site defaults)."""
import json, os, re, sys, random, hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la60_common
from la60_common import _pub_source, SITE_CODE, LIB_SUPPORTS, tab_of, admin_docs, split, seed, _num_after, base_of
import la71_parse

D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la72_ckpt')
os.makedirs(CK, exist_ok=True)
SECW = ('KU-RO', 'KI-RO', 'PO-TO-KU-RO')


def load(ver='rd'):
    """la60_common.load_la conversion with token statuses kept; cached per version in la72_ckpt."""
    fn = os.path.join(CK, 'docs_%s.json' % ver)
    src = os.path.join(D, 'corpus_ra.json')
    if os.path.exists(fn) and os.path.getmtime(fn) > os.path.getmtime(src):
        return json.load(open(fn))
    raw = json.load(open(src))
    stat = {d['id']: [t.get('st', '-') for t in d['tokens']] for d in raw}
    C = la71_parse.version(raw, ver)
    pub = _pub_source()
    ids = {d['id'] for d in C}
    out = []
    for d in C:
        if '+' in d['id']:
            parts = re.split(r'\+', re.sub(r'[ab]$', '', d['id']))
            pre = re.match(r'[A-Z]+[A-Za-z]*?(?=\d)', parts[0])
            pre = pre.group(0) if pre else ''
            comp = [parts[0]] + [p if not p[0].isdigit() else pre + p for p in parts[1:]]
            if all(any(i.startswith(c) for i in ids if i != d['id']) for c in comp):
                continue
        toks = []; sts = []
        for k, t in enumerate(d['tokens']):
            s0 = stat[d['id']][k]
            if t['t'] == 'word':
                toks.append(['W', '-'.join(t['s']), None]); sts.append(s0)
            elif t['t'] == 'logo':
                toks.append(['L', t['v'], None]); sts.append(s0)
            elif t['t'] == 'num':
                toks.append(['N', int(t['v']), ''.join(sorted(t['frac'])) if t['frac'] else '']); sts.append(s0)
            elif t['t'] == 'nl':
                if toks and toks[-1][0] != 'NL':
                    toks.append(['NL', None, None]); sts.append('-')
            elif t['t'] == 'unk' and t.get('v') == '#R':
                o = t.get('o') or {}
                ow = '-'.join(o.get('s', [])) if o.get('t') == 'word' else None
                toks.append(['X', t.get('k') or '', ow if ow in SECW else None]); sts.append(s0)
        while toks and toks[-1][0] == 'NL':
            toks.pop(); sts.pop()
        while toks and toks[0][0] == 'NL':
            toks.pop(0); sts.pop(0)
        sup = d['support']
        out.append(dict(id=d['id'], tab=tab_of(d['id']), site=SITE_CODE.get(d['site'], d['site'] or '?'),
                        support=sup, pub=pub.get(d['id'], 'blank'), scribe=d.get('scribe') or '',
                        lib=sup in LIB_SUPPORTS, toks=toks, st=sts))
    json.dump(out, open(fn, 'w'))
    return out


def admin(ver='rd'):
    return admin_docs(load(ver))


# ------------------------------------------------------------------ train-only checks la70 did on all docs / by hand
def ki_cases(docs, w='KI', vals=None):
    out = []
    for d in docs:
        toks = d['toks']
        nums = [i for i, t in enumerate(toks) if t[0] == 'N']
        v = vals[d['id']] if vals else {i: toks[i][1] for i in nums}
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] == w and i + 1 < len(toks) and toks[i + 1][0] == 'N':
                prev = [k for k in nums if k < i]
                if prev and v[prev[-1]] > 0:
                    out.append((d['id'], v[i + 1] < v[prev[-1]]))
    return out


def share(c):
    return sum(x for _, x in c) / len(c) if c else float('nan')


def ki_perm_p(docs, w, real, rng, n=1000):
    ge = 0
    rel = [d for d in docs if any(t[0] == 'W' and t[1] == w for t in d['toks'])]
    for _ in range(n):
        vals = {}
        for d in rel:
            nums = [i for i, t in enumerate(d['toks']) if t[0] == 'N']
            vs = [d['toks'][i][1] for i in nums]; rng.shuffle(vs)
            vals[d['id']] = dict(zip(nums, vs))
        ge += share(ki_cases(rel, w, vals)) >= real
    return (ge + 1) / (n + 1)


def ki_gate(train, rng, ndec=300):
    """la70 c1a survival rule, on TRAINING documents only: within-document P <= 0.05, share without
    HT 118 > 0.5, KI P in the lowest 10 % of single-sign decoys (>= 4 cases)."""
    ki = ki_cases(train); s = share(ki)
    if len(ki) < 3:
        return dict(n=len(ki), share=s, survives=False, reason='< 3 training cases')
    wo = [c for c in ki if c[0] != 'HT118']
    p = ki_perm_p(train, 'KI', s, rng)
    singles = sorted({t[1] for d in train for t in d['toks'] if t[0] == 'W' and '-' not in t[1]} - {'KI'})
    dps = []
    for w in singles:
        c = ki_cases(train, w)
        if len(c) >= 4:
            dps.append(ki_perm_p(train, w, share(c), rng, ndec))
    rank = sum(1 for x in dps if x <= p) / max(1, len(dps))
    surv = p <= 0.05 and (share(wo) > 0.5 if wo else False) and rank <= 0.10
    return dict(n=len(ki), share=s, p=p, n_wo=len(wo), share_wo=share(wo) if wo else None, decoy_rank=rank,
                n_decoys=len(dps), survives=surv)


def db_gate(train):
    """D = 1/5, B = 1/3 kept only if no training amount writes D >= 5 times or B >= 3 times."""
    bad = []
    for d in train:
        for t in d['toks']:
            if t[0] == 'N' and t[2] and (t[2].count('D') >= 5 or t[2].count('B') >= 3):
                bad.append((d['id'], t[1], t[2]))
    return dict(violations=bad, keep=not bad)


def site_default_gate(train, roles, prior):
    """Modal commodity base per site among training documents with a commodity; keep the prior default
    only where the training mode agrees."""
    c = defaultdict(Counter)
    for d in train:
        bs = {base_of(t[1]) for t in d['toks'] if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM')}
        for b in bs:
            c[d['site']][b] += 1
    keep = {}; info = {}
    for s, b in prior.items():
        m = c[s].most_common(3)
        info[s] = m
        if m and m[0][0] == b:
            keep[s] = b
    return keep, info


def sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
