"""pe36: DO THE NAME SIGNS WRITE SOUNDS?  Shared code.

Reuses the blind consonant x vowel grid annealer of Linear A la21 (linear-a/tools/la21_core.c via la21_common):
signs are opaque identities; a grid of 1 pure-vowel row + R consonant rows x K vowel columns is fitted by
simulated annealing scoring only cross-linguistic universals (independent consonant / vowel tiers, consonant OCP,
pure vowels word-initial, small cells). No sign values from any script are inputs. True values are used only to
SCORE the controls (Old Babylonian / Ur III seal names in cuneiform syllables, Linear B names).

Corpora (each a list of (string tuple, tablet id)):
  PEN   PE name-like middles (pe31 definition: entry lines, final class sign stripped, >= 2 signs), base signs
  PEA   all clean PE sign strings of >= 2 signs (headers, entries, class signs kept), base signs
  PENUM PE numeral notations as strings of N-codes (one string per numeral group, written order) -- not names
  PECLS PE class-sign sequences down each tablet (final class signs of consecutive entries, chunks of <= 5)
  OB    Old Babylonian seal/patronym names (cuneiform syllables)                        -- positive control
  UR3   Ur III seal names (Sumerian, largely logographic name elements + some syllables)
  LINB  Linear B personal names (pe7 set)                                                -- positive control
  PC    proto-cuneiform administrative entry strings (largely logographic)               -- negative control
"""
import os, sys, re, json, random, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LA_TOOLS = os.path.join(HERE, '..', '..', 'linear-a', 'tools')
sys.path.insert(0, LA_TOOLS)
from common import load, is_sign  # noqa: E402
import pe31_lib as P31  # noqa: E402
from la21_common import anneal, coassign, stability, auc, groups, truth_lb  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe36_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
NSIGN = 65


def _clean(s):
    return s != 'x' and '...' not in s and 'X' not in s.replace('|', '').split('+')


def corpus(name):
    if name == 'PEN':
        return [(w, t) for w, t in P31.pe_names(variants=False)]
    T = load()
    out = []
    if name == 'PEA':
        for t in T:
            for l in t['lines']:
                sg = [s for s in l['signs'] if is_sign(s) or s == 'x']
                if len(sg) < 2 or l.get('lacuna') or not all(_clean(s) for s in sg):
                    continue
                out.append((tuple(P31.base(s) for s in sg), t['id']))
        return out
    if name == 'PENUM':
        for t in T:
            for l in t['lines']:
                if len(l['numerals']) >= 2 and not l.get('lacuna'):
                    out.append((tuple(c for _, c in l['numerals']), t['id']))
        return out
    if name == 'PECLS':
        for t in T:
            seq = []
            for i, l in enumerate(t['lines']):
                sg = l['signs']
                if not l['numerals'] or not sg or not _clean(sg[-1]) or not is_sign(sg[-1]):
                    continue
                seq.append(P31.base(sg[-1]))
            for k in range(0, len(seq), 5):
                ch = tuple(seq[k:k + 5])
                if len(ch) >= 2:
                    out.append((ch, t['id']))
        return out
    if name in ('OB', 'UR3', 'LINB'):
        key = {'OB': 'OB_SEAL', 'UR3': 'UR3_SEAL', 'LINB': 'LINB'}[name]
        d = json.load(open(os.path.join(DATA, 'pe7_corpora.json')))[key]
        for x in d:
            w = tuple(s for s in x['seq'] if s)
            if len(w) >= 2:
                out.append((w, x['tablets'][0]))
        return out
    if name == 'PC':
        PC = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
        for t in PC:
            for l in t['lines']:
                sg = l['signs']
                if len(sg) < 2 or l.get('lacuna') or not all(s != 'x' and '...' not in s for s in sg):
                    continue
                out.append((tuple(re.sub(r'~[A-Za-z0-9]+', '', s) for s in sg), t['id']))
        return out
    raise KeyError(name)


def types_of(data):
    """distinct string types, each with its first tablet (la21 works on types)."""
    seen = {}
    for w, t in data:
        if w not in seen:
            seen[w] = t
    return sorted(seen.items())


# ------------------------------------------------------------------ truth (controls only)
CUN = re.compile(r"^(b|d|g|h|k|l|m|n|p|q|r|s|s,|sz|t|t,|w|y|z|i(?=a))?([aeiu])\d*$")


def truth_cun(s):
    """(onset, vowel) of a plain cuneiform CV / V syllable value, else None. 'ia' = ya."""
    s = s.lower()
    if s == 'ia' or re.fullmatch(r'ia\d*', s):
        return ('y', 'a')
    m = CUN.match(s)
    if not m:
        return None
    return (m.group(1) or '', m.group(2))


def truth_linb(s):
    return truth_lb(s.upper())


TRUTH = {'OB': truth_cun, 'UR3': truth_cun, 'LINB': truth_linb}


def counts(types, nsign=NSIGN):
    """top-nsign signs by type frequency; words split at other signs into fragments -> signs, B, I, F."""
    c = collections.Counter(s for w in types for s in w)
    signs = [s for s, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:nsign]]
    ix = {s: i for i, s in enumerate(signs)}
    S = len(signs)
    B = np.zeros((S, S), np.int32); I = np.zeros(S, np.int32); F = np.zeros(S, np.int32)
    for w in types:
        frag = []
        for s in list(w) + [None]:
            if s is not None and s in ix:
                frag.append(ix[s]); continue
            if frag:
                I[frag[0]] += 1; F[frag[-1]] += 1
                for a, b in zip(frag, frag[1:]):
                    B[a, b] += 1
            frag = []
    return signs, B, I, F


def shuffle_within(types, seed):
    r = random.Random(seed); out = []
    for w in types:
        w = list(w); r.shuffle(w); out.append(tuple(w))
    return out


def grid_dims(S):
    """la21 used 65 signs on 1+14 rows x 5 columns, cap 2. Smaller alphabets get proportionally fewer rows."""
    return (15, 5) if S >= 60 else (max(3, int(round(15 * S / 65))), 5)


def truth_scores(signs, PR, PC, tf):
    T = [tf(s) for s in signs]
    idx = [i for i, t in enumerate(T) if t is not None]
    sr, lr, sc, lc, sr2, lr2 = [], [], [], [], [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            sr.append(PR[i, j]); lr.append(T[i][0] == T[j][0])
            sc.append(PC[i, j]); lc.append(T[i][1] == T[j][1])
            if T[i][0] and T[j][0]:
                sr2.append(PR[i, j]); lr2.append(T[i][0] == T[j][0])
    sr2a = np.asarray(sr2); lr2a = np.asarray(lr2, bool)
    m = sr2a >= .7 if len(sr2a) else np.zeros(0, bool)
    return dict(ntruth=len(idx), npos_row=int(lr2a.sum()), row_auc=auc(sr, lr), crow_auc=auc(sr2, lr2),
                col_auc=auc(sc, lc), prec70=(int(lr2a[m].sum()), int(m.sum())))


def rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def load_job(tag):
    import glob
    fs = sorted(glob.glob(os.path.join(CK, f'{tag}__*.npz')))
    if not fs:
        raise FileNotFoundError(tag)
    rows = []; cols = []; sc = []
    for f in fs:
        z = np.load(f)
        rows.append(z['rows']); cols.append(z['cols']); sc.append(z['sc']); signs = [str(s) for s in z['signs']]
        meta = dict(B=z['B'], I=z['I'], F=z['F'])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sc), signs, meta
