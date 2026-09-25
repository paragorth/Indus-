"""Data-only grammar: position classes of MDL units and substitution paradigms,
validated on held-out texts. No labels, no readings.

Units come from cleanroom.mdl_segment fitted on the TRAIN half only; held-out
texts are segmented with the same merges.
  * position profile: share of a unit's tokens that are segment-initial / -final
  * paradigms: units that alternate in the same frame (same left and right unit)
  * held-out test: a 3-class slot model (initial-type, medial, final-type units)
    predicts whether a held-out unit is initial/final far better than chance?
"""
from collections import Counter, defaultdict

import cleanroom as CR
import indus_core as C


def apply_merges(seg, merges):
    units = [(x,) for x in seg]
    for m in merges:
        a, b = None, None
        # a merge is a concatenated tuple; find the split that produced it
        for k in range(1, len(m)):
            a, b = m[:k], m[k:]
            if any(units[i] == a and units[i + 1] == b for i in range(len(units) - 1)):
                units = CR.merge(units, a, b)
    return units


def profile(corpus):
    pos = defaultdict(Counter)
    for seg in corpus:
        for i, u in enumerate(seg):
            if len(seg) == 1:
                pos[u]["only"] += 1
            elif i == 0:
                pos[u]["initial"] += 1
            elif i == len(seg) - 1:
                pos[u]["final"] += 1
            else:
                pos[u]["medial"] += 1
    return pos


def klass(c):
    n = sum(c.values())
    if c["initial"] / n >= .6:
        return "I"
    if (c["final"] + c["only"]) / n >= .6:
        return "F"
    return "M"


def main(corpus_name="yajnadevam"):
    texts = C.unique_texts(C.load_corpus() if corpus_name == "yajnadevam" else C.load_im77_corpus())
    tr, he = C.split_texts(texts, seed=0)
    trc, merges = CR.mdl_segment(CR.segs_of(tr))
    hec = [apply_merges(s, merges) for s in CR.segs_of(he)]
    pos = profile(trc)
    cls = {u: klass(c) for u, c in pos.items() if sum(c.values()) >= 5}
    # held-out: for units with a class, is the position consistent?
    ok = tot = 0
    base = Counter()
    for seg in hec:
        if len(seg) < 2:
            continue
        for i, u in enumerate(seg):
            p = "I" if i == 0 else ("F" if i == len(seg) - 1 else "M")
            base[p] += 1
            if u in cls:
                tot += 1
                ok += cls[u] == p
    n = sum(base.values())
    chance = sum((v / n) ** 2 for v in base.values())
    print(f"[{corpus_name}] held-out units with a class: {tot}; position predicted correctly {ok / tot:.3f} vs chance {chance:.3f}")
    # paradigms: frames (left, right) with >=3 distinct fillers, in train
    frames = defaultdict(Counter)
    for seg in trc:
        s = ["^"] + list(seg) + ["$"]
        for i in range(1, len(s) - 1):
            frames[(s[i - 1], s[i + 1])][s[i]] += 1
    big = sorted(((k, v) for k, v in frames.items() if len(v) >= 3), key=lambda kv: -len(kv[1]))
    fmt = lambda u: u if isinstance(u, str) else "-".join(map(str, u))
    print(f"frames with >=3 alternating fillers: {len(big)}; the largest:")
    for (l, r), v in big[:8]:
        print(f"  [{fmt(l)}] _ [{fmt(r)}]: {len(v)} fillers, e.g. {', '.join(fmt(u) for u, _ in v.most_common(6))}")
    classes = Counter(cls.values())
    print("unit classes (train):", dict(classes))


if __name__ == "__main__":
    main("yajnadevam")
    main("im77")
