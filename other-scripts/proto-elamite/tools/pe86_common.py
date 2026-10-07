"""pe86: the clay as an independent witness of the header.

The squared header end (pe81/pe83/pe85, B) is a physical mark nobody produced by reading Proto-Elamite. Here it is used
as a label: which text features explain where the clay and our text-based header definition disagree, which random
header definitions predict the clay best on held-out publication batches, and what the clay says about tablets whose
first line is lost. Reuses data/pe83_ckpt outline features through pe85_common.table(); 1 worker.
"""
import sys, os, json, hashlib, re
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe83_common as P
import pe85_common as C

CK = os.path.join(common.DATA, 'pe86_ckpt')
os.makedirs(CK, exist_ok=True)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def base(s):
    return common.base(s.rstrip('?#!')) if s not in ('x', 'X') else 'x'


TOK = re.compile(r'(\|[^|\s]+\||M\d+[~A-Za-z0-9]*|\bx\b)')


def l1_tokens(t, j=0):
    """sign tokens of line j read from the raw transliteration (keeps uncertain '?' readings that the loader drops to x)."""
    l = t['lines'][j]
    raw = re.sub(r'\d+\([A-Z0-9]+\)[#?!]*', ' ', l['raw'])
    return [base(m) for m in TOK.findall(raw)]


def build(intact=True):
    R = C.table(intact)
    Z = P.design(R, [] if intact else [C.col(R, 'l1ok')])  # size, aspect, thickness, line count, damage, reverse
    top = C.col(R, 'cf_top'); bot = C.col(R, 'cf_bot')
    ytop = C.resid(top, Z); ybot = C.resid(bot, Z)
    hd = C.col(R, 'hd')
    vol = np.array([r['vol'] if r['vol'] in ('MDP 06', 'MDP 17', 'MDP 26', 'MDP 26S', 'MDP 31') else 'other' for r in R])
    pub = np.array([r['pub'] if np.isfinite(r['pub']) else 0 for r in R], float)
    batch = np.array([f'{v}_{int(p // 50)}' for v, p in zip(vol, pub)])
    s = P.strata(Z)
    return dict(R=R, Z=Z, top=top, bot=bot, ytop=ytop, ybot=ybot, hd=hd, vol=vol, batch=batch, strata=s)


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(b[ok]) == 0 or np.std(a[ok]) == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def perm_within(v, s, rng):
    v = v.copy()
    for k in np.unique(s):
        m = np.where(s == k)[0]; v[m] = v[rng.permutation(m)]
    return v


def line_feats(R, min_n=5):
    """binary text features of each tablet (no clay): line-1 form, line-1 tokens, line-1 first token, line-2 first token."""
    F = {}
    def add(k, i):
        F.setdefault(k, np.zeros(len(R)))[i] = 1.0
    for i, r in enumerate(R):
        t = r['t']; ls = t['lines']; l = ls[0]
        legible = [s for s in l['signs'] if common.is_sign(s)]
        toks = l1_tokens(t)
        if l['numerals']: add('L1:has_numerals', i)
        if not legible: add('L1:no_legible_sign', i)
        if any(s == 'x' for s in l['signs']): add('L1:has_x', i)
        if '?' in l['raw']: add('L1:uncertain_sign', i)
        if not l['numerals'] and toks and not legible: add('L1:header_shaped_illegible', i)
        if l.get('header_comment'): add('L1:cdli_header_comment', i)
        if len(toks) >= 3: add('L1:3plus_tokens', i)
        if len(toks) == 1: add('L1:1_token', i)
        if l['numerals'] and len(toks) == 1: add('L1:1sign+number', i)
        if l['surface'] != 'obverse': add('L1:not_obverse', i)
        for k in set(toks):
            add('L1tok:' + k, i)
        if toks:
            add('L1first:' + toks[0], i)
        if len(ls) > 1 and ls[1]['signs']:
            t2 = l1_tokens(t, 1)
            if t2: add('L2first:' + t2[0], i)
            if not ls[1]['numerals']: add('L2:no_numerals', i)
    return {k: v for k, v in F.items() if v.sum() >= min_n}
