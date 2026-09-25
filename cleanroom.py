"""Clean-room analysis: no Mahadevan, no Parpola, no sign labels, no language prior.

Input: Yajnadevam's transcription only (Wells sign numbers as opaque IDs),
reversed to reading order, deduplicated, damage/line breaks split segments.

1. Unsupervised word segmentation by minimum description length (MDL):
   greedily merge the adjacent pair of units that most reduces
   L(lexicon) + L(corpus | lexicon); stop when no merge helps.
2. Reproducibility: fit on each half of the corpus separately; how many
   multi-sign units are found in both halves, versus a within-text shuffled
   corpus (which has the same signs but no sequence structure)?
3. Typology: for every recurring unit, count distinct left neighbours and
   right neighbours (in reading order). Suffixing languages vary on the right
   of stems; prefixing languages on the left.
"""
import math
import random
from collections import Counter

import indus_core as C
import validator as V


def segs_of(texts):
    return [tuple(s) for t in texts for s in t if s]


def desc_len(corpus):
    """Two-part code: lexicon (each unit spelled in signs) + corpus as unit tokens."""
    units = Counter(u for seg in corpus for u in seg)
    n_tok = sum(units.values())
    signs = Counter(x for u in units for x in u)
    n_sig = sum(signs.values())
    lex = sum(-math.log2(signs[x] / n_sig) for u in units for x in u) + len(units) * math.log2(10)
    cor = -sum(c * math.log2(c / n_tok) for c in units.values())
    return lex + cor


def mdl_segment(segs, max_iter=400, min_count=3):
    corpus = [[(x,) for x in s] for s in segs]
    base = desc_len(corpus)
    merges = []
    for _ in range(max_iter):
        pairs = Counter((a, b) for seg in corpus for a, b in zip(seg, seg[1:]))
        best = None
        for (a, b), c in pairs.most_common(60):
            if c < min_count:
                break
            new = [merge(seg, a, b) for seg in corpus]
            dl = desc_len(new)
            if dl < base and (best is None or dl < best[0]):
                best = (dl, a, b, new)
        if not best:
            break
        base, a, b, corpus = best
        merges.append(a + b)
    return corpus, merges


def merge(seg, a, b):
    out, i = [], 0
    while i < len(seg):
        if i + 1 < len(seg) and seg[i] == a and seg[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(seg[i]); i += 1
    return out


def shuffle_segs(segs, seed):
    rng = random.Random(seed)
    out = []
    for s in segs:
        s = list(s); rng.shuffle(s); out.append(tuple(s))
    return out


def typology(corpus, min_count=5):
    left, right, cnt = {}, {}, Counter()
    for seg in corpus:
        for i, u in enumerate(seg):
            cnt[u] += 1
            left.setdefault(u, set()).add(seg[i - 1] if i else "^")
            right.setdefault(u, set()).add(seg[i + 1] if i + 1 < len(seg) else "$")
    rows = [(u, cnt[u], len(left[u]), len(right[u])) for u in cnt if cnt[u] >= min_count]
    return rows


def main():
    texts = C.unique_texts(C.load_corpus())
    segs = segs_of(texts)
    print(f"segments: {len(segs)}, sign tokens: {sum(map(len, segs))}")
    a, b = C.split_texts(texts, seed=0)
    res = {}
    for name, part in (("half A", segs_of(a)), ("half B", segs_of(b)),
                       ("half A shuffled", shuffle_segs(segs_of(a), 1)),
                       ("half B shuffled", shuffle_segs(segs_of(b), 2))):
        corpus, merges = mdl_segment(part)
        multi = {u for seg in corpus for u in seg if len(u) > 1}
        res[name] = multi
        print(f"{name:16s}: {len(merges)} merges, {len(multi)} multi-sign units")
    both = res["half A"] & res["half B"]
    both_s = res["half A shuffled"] & res["half B shuffled"]
    print(f"units found independently in both halves: real {len(both)}, shuffled {len(both_s)}")

    corpus, merges = mdl_segment(segs)
    units = Counter(u for seg in corpus for u in seg)
    multi = [(u, c) for u, c in units.most_common() if len(u) > 1]
    print(f"\nfull corpus: {len(multi)} multi-sign units; top 30 (reading order, Wells IDs):")
    for u, c in multi[:30]:
        print(f"  {'-'.join(u):28s} x{c}")
    lens = Counter(len(seg) for seg in corpus)
    print("units per segment:", sorted(lens.items())[:10])

    rows = typology(corpus)
    # For units that are neither text-initial-only nor final-only, compare left vs right variety
    lv = sum(r[2] for r in rows); rv = sum(r[3] for r in rows)
    print(f"\ntypology over {len(rows)} units seen >=5 times: distinct left contexts {lv}, right contexts {rv}, right/left = {rv / lv:.2f}")
    # same statistic on shuffled segments segmented the same way
    sc, _ = mdl_segment(shuffle_segs(segs, 3))
    r2 = typology(sc)
    print(f"shuffled control: right/left = {sum(r[3] for r in r2) / sum(r[2] for r in r2):.2f}")
    return corpus


if __name__ == "__main__":
    main()
