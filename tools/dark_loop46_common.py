"""Loop 46 (HANDS): shared loader for the allograph-choice tests.

Variant data: data/derived/merged-corpus-canonical.json. seq_raw keeps Wells's original numbers,
seq_strong / seq_all replace variant forms by the class head (sign_allographs_levels.json). Aligning
seq_raw with the merged sequence recovers which form of each merged sign was cut in each text.

Merge levels used here:
  strong  = classes whose forms are all 'strong' merges (plus the head)
  all     = strong + probable (the canonical seq_all)
  ext     = strong + probable + doubtful (adds 151/156, 511/413, 825/820); 151/156 is one of the three
            transcription-robust distinctions of S-DARK-24, so it is kept in 'ext' although S268 doubts it
Transcription-robust distinctions (S-DARK-24.1 + loop24_pairs.json alignments, both readers draw two signs):
  390 | 405/406  (M162 | M169)      803 | 806 (M387 | M389)      151 | 156 (M12 | M15)
  90 | 93 (M1 | M3, 7 tokens) and 820 | 825 (M391 | M380, 6 tokens) are robust but below the token cut.
Not robust (Mahadevan draws one sign): 154/156/158 = M15, 705/706 = M336, 861/817 = M267, 630/632/636 = M244,
  717/711 = M341, 824/877 = M284, 527/526 = M254, 297/298, 315/317, 336/337, 772/773, 370/371.
"""
import json, csv, collections, itertools, random
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'

ROBUST_HEADS = {390, 803, 156}
MIN_VARIANT_TOKENS = 20


def load_corpus():
    return json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))


def load_classes(level):
    """Return {head: set(forms incl. head)} for a merge level."""
    merges = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']
    allowed = {'strong': {'strong'}, 'all': {'strong', 'probable'},
               'ext': {'strong', 'probable', 'doubtful'}}[level]
    cl = collections.defaultdict(set)
    for m in merges:
        if m['level'] in allowed:
            cl[m['into']].add(m['form']); cl[m['into']].add(m['into'])
    return dict(cl)


def variant_tokens(corpus, classes):
    """For each text: list of (head, form) for every seq_raw token whose form belongs to a class."""
    form2head = {f: h for h, fs in classes.items() for f in fs}
    out = []
    for r in corpus:
        out.append([(form2head[s], s) for s in r['seq_raw'] if s in form2head])
    return out


def usable_classes(corpus, classes, min_tokens=MIN_VARIANT_TOKENS, heads=None):
    """Classes with >= 2 forms having >= min_tokens tokens each. Binary coding: head form vs other forms
    (minor forms below the cut are folded into 'other' only if they are not the head)."""
    tok = collections.Counter()
    for r in corpus:
        for s in r['seq_raw']:
            tok[s] += 1
    use = {}
    for h, fs in classes.items():
        if heads is not None and h not in heads:
            continue
        big = [f for f in fs if tok[f] >= min_tokens]
        if len(big) >= 2 and tok[h] >= min_tokens:
            use[h] = {'forms': sorted(fs), 'counts': {f: tok[f] for f in sorted(fs)}, 'big': big}
    return use


def text_labels(corpus, classes, use):
    """Per text: {head: 0/1} where 1 = non-head form (majority over the text's tokens; ties -> first token).
    Also returns within-text inconsistency records (text idx, head, forms)."""
    vt = variant_tokens(corpus, classes)
    labels = []
    incons = []
    multi = []
    for i, toks in enumerate(vt):
        d = collections.defaultdict(list)
        for h, f in toks:
            if h in use:
                d[h].append(f)
        lab = {}
        for h, fs in d.items():
            if len(fs) >= 2:
                multi.append((i, h, fs))
                if len(set(fs)) > 1:
                    incons.append((i, h, fs))
            nonhead = sum(1 for f in fs if f != h)
            lab[h] = 1 if nonhead * 2 > len(fs) else (0 if nonhead * 2 < len(fs) else (1 if fs[0] != h else 0))
        labels.append(lab)
    return labels, multi, incons


def strata(corpus, keys):
    """Stratum id per text from the given record keys."""
    ids = []
    for r in corpus:
        ids.append(tuple(str(r.get(k, '-')) for k in keys))
    return ids


def permute_within(labels_arr, strat_arr, rng):
    """Permute a 1-d label array within strata."""
    out = labels_arr.copy()
    order = np.argsort(strat_arr, kind='stable')
    s_sorted = strat_arr[order]
    # boundaries
    bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) > 1:
            out[idx] = labels_arr[rng.permutation(idx)]
    return out


def mi_bits(a, b):
    """Mutual information (bits) of two binary arrays."""
    n = len(a)
    if n == 0:
        return 0.0
    tab = np.zeros((2, 2))
    for x in (0, 1):
        for y in (0, 1):
            tab[x, y] = np.sum((a == x) & (b == y))
    p = tab / n
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = p * np.log2(p / (px * py))
    return float(np.nansum(t))


def log_odds(a, b):
    n11 = np.sum((a == 1) & (b == 1)) + .5; n00 = np.sum((a == 0) & (b == 0)) + .5
    n10 = np.sum((a == 1) & (b == 0)) + .5; n01 = np.sum((a == 0) & (b == 1)) + .5
    return float(np.log((n11 * n00) / (n10 * n01)))


def row(cycle, method, result, verdict):
    return f'| S-DARK-46.{cycle} | {method} | {result} | {verdict} |'
