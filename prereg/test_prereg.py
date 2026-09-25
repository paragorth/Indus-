"""Tests for the fourteen frozen predictions in preregistration-2-frozen.json (Q1-Q14).

    python prereg/test_prereg.py                 # every file in new_data/
    python prereg/test_prereg.py FILE [FILE ...] # specific corpora

With new_data/ empty it runs on the three corpora already used by the project
(merged reading-order JSON, Mahadevan IM77, Yajnadevam CSV) to show the harness
works.  Those runs are IN-SAMPLE: the predictions were derived from them, so
they check the code, not the predictions.

Accepted corpus formats (detected automatically):
  *.json   list of records like data/derived/merged-corpus-reading-order.json
           (cisi, site, type, symbol, time, period, seq = Wells/Yajnadevam glyph
           numbers in reading order)
  *.csv    with a `signs_clean` column: IM77 export (Mahadevan numbers, reading
           order); converted to glyph numbers through data/derived/bridge_extended.json
  *.csv    with a `text` column: Yajnadevam inscriptions.csv format (stored
           left-to-right; reversed here)

Each prediction resolves to PASS (statement met), FALSIFIED (falsifier met), or
INDETERMINATE (neither, or too few tokens for the falsifier to apply).

Interpretation choices the frozen text leaves open (fixed here, before any new
data is seen):
  * Deduplication: identical (site, object class, text, emblem) records count
    once, for every test except Q8/Q9.  Those count multi-faced tablets as
    objects, because a mass-produced denomination series is the phenomenon under
    test.
  * Short-stroke numerals: glyphs 1,2,3,4,5,16,17,18 = values 1-8.  Tall
    strokes: glyphs 31-34 = I-IIII.
  * Q4 counts short-stroke values 3-8 only.  Values 1 and 2 are the
    connectives (glyphs 1, 2), and tall II + fish is a fixed compound (Q5-type).
  * Q5 counts every stroke sign (short or tall) as a numeral token.
  * "Text-final" is the last sign of a record (one face or line).
"""
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SHORT = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8}
TALL = {31: 1, 32: 2, 33: 3, 34: 4}
STROKES = set(SHORT) | set(TALL)
TREE = {390, 405, 407}
CRESCENT = 900
JAR, MJAR = 740, 741
FISH, FISH4 = 220, 226
LEAF = 700
OPENERS = {817, 861, 820}

# Mahadevan -> glyph for IM77 input: the bridge where it is one-to-one, plus the
# families the predictions name explicitly.
M_OVERRIDE = {342: 740, 343: 741, 344: 741, 345: 741, 346: 741,
              161: 390, 162: 390, 167: 390, 168: 390, 169: 390, 267: 861, 60: 226}


# ---------------------------------------------------------------- loading


def cls_of(t):
    t = (t or "").upper()
    if t.startswith("SEAL"):
        return "seal"
    if t.startswith("TAG") or t == "SEALING":
        return "sealing"
    if t.startswith("TAB") or "TABLET" in t:
        return "tablet"
    return "other"


def load_merged(path):
    recs = []
    for i, r in enumerate(json.load(open(path))):
        obj = r["cisi"] if r.get("cisi") not in (None, "", "-") else f"_anon{i}"
        recs.append(dict(obj=obj, site=r.get("site", ""), cls=cls_of(r.get("type")),
                         symbol=norm_sym(r.get("symbol")),
                         period=(r.get("time") or "-", r.get("period") or "-"),
                         seq=[int(x) for x in r["seq"] if int(x) != 0]))
    return recs


def load_yajnadevam(path):
    recs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            toks = [int(x) for x in re.findall(r"\d{3}", r["text"].split("/")[0])][::-1]
            obj = r["id"].split(".")[0]
            recs.append(dict(obj=obj, site=r["site"], cls=cls_of(r["type"]),
                             symbol=norm_sym(r["symbol"]), period=(r["time"], r["period"]),
                             seq=[t for t in toks if t not in (0, 999)]))
    return recs


def load_im77(path):
    bridge = json.load(open(os.path.join(ROOT, "data", "derived", "bridge_extended.json")))
    inv = defaultdict(set)
    for g, ms in bridge.items():
        for m in ms:
            inv[m].add(int(g))
    m2g = {m: next(iter(gs)) for m, gs in inv.items() if len(gs) == 1}
    m2g.update(M_OVERRIDE)
    objtype = {"1": "seal", "2": "sealing", "3": "tablet", "5": "tablet"}
    recs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["line"] == "9":
                continue
            seq = []
            for x in r["signs_clean"].split():
                m = int(x)
                if m == 0:
                    continue
                seq.append(m2g.get(m, 10_000 + m))   # unbridged signs stay distinct
            recs.append(dict(obj=r["text_no"], site=r["site"],
                             cls=objtype.get(r["object_code"], "other"),
                             symbol=norm_sym(r["fs80"] if r["fs80"] not in ("0", "999") else ""),
                             period=(r["level"] or "-", "-"), seq=seq,
                             face=r["side"]))
    return recs


def norm_sym(s):
    s = (s or "").strip()
    return "" if s in ("", "-", "None", "none", "0") else s


def load(path):
    if path.endswith(".json"):
        return load_merged(path)
    with open(path, newline="") as f:
        head = f.readline()
    return load_im77(path) if "signs_clean" in head else load_yajnadevam(path)


def dedup(recs):
    seen, out = set(), []
    for r in recs:
        k = (r["site"], r["cls"], tuple(r["seq"]), r["symbol"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------------------------------------------------------------- helpers


def final_share(recs, signs):
    n = fin = 0
    for r in recs:
        for i, s in enumerate(r["seq"]):
            if s in signs:
                n += 1
                fin += i == len(r["seq"]) - 1
    return (fin / n if n else float("nan")), n


def preceding(recs, targets, numeral_map):
    c = Counter()
    for r in recs:
        q = r["seq"]
        for i in range(1, len(q)):
            if q[i] in targets and q[i - 1] in numeral_map:
                c[numeral_map[q[i - 1]]] += 1
    return c


def verdict(passed, falsified):
    return "FALSIFIED" if falsified else ("PASS" if passed else "INDETERMINATE")


def f(x):
    return "nan" if x != x else f"{x:.3f}"


# ---------------------------------------------------------------- Q1-Q14


def q1(R):
    j, nj = final_share(R, {JAR}); m, nm = final_share(R, {MJAR})
    ok = j >= .60 and m <= .15
    return verdict(ok, nm >= 30 and not ok), f"jar final {f(j)} (n={nj}); marked jar final {f(m)} (n={nm})"


def q2(R):
    a, na = final_share(R, {FISH4}); b, nb = final_share(R, {FISH})
    ok = a >= .40 and b <= .15
    return verdict(ok, na >= 20 and not ok), f"four-stroke fish final {f(a)} (n={na}); plain fish final {f(b)} (n={nb})"


def q3(R):
    c = preceding(R, TREE, SHORT)
    hi, lo = sum(v for k, v in c.items() if 3 <= k <= 8), c[1] + c[2]
    ratio = hi / lo if lo else float("inf") if hi else float("nan")
    n = hi + lo
    return verdict(ratio >= 8, n >= 40 and ratio < 4), f"values 3-8 : 1-2 = {hi}:{lo} (ratio {ratio:.2f}, n={n}); profile {dict(sorted(c.items()))}"


def q4(R):
    c = preceding(R, {FISH}, {g: v for g, v in SHORT.items() if v >= 3})
    n = sum(c.values())
    p5 = c[5] / n if n else float("nan")
    p36 = (c[3] + c[6]) / n if n else float("nan")
    ok = n > 0 and p5 <= .05 and p36 >= .60
    return verdict(ok, n >= 25 and (p5 >= .15 or p36 < .40)), f"n={n}, value 5 {f(p5)}, values 3+6 {f(p36)}; profile {dict(sorted(c.items()))}"


def q5(R):
    allnum = {**{g: ("s", v) for g, v in SHORT.items()}, **{g: ("t", v) for g, v in TALL.items()}}
    out, ok, fal = [], True, False
    for target, want in ((575, ("s", 7)), (585, ("s", 7)), (520, ("t", 3))):
        c = preceding(R, {target}, allnum)
        n = sum(c.values()); p = c[want] / n if n else float("nan")
        ok &= n > 0 and p >= .70
        fal |= n >= 10 and p < .50
        out.append(f"{want[1]}{'T' if want[0] == 't' else ''}+{target}: {f(p)} of {n}")
    return verdict(ok, fal), "; ".join(out)


def rate_in(recs, signs):
    tot = sum(len(r["seq"]) for r in recs)
    return sum(1 for r in recs for s in r["seq"] if s in signs) / tot if tot else float("nan")


def q6(R):
    tab = [r for r in R if r["cls"] == "tablet"]; seal = [r for r in R if r["cls"] == "seal"]
    ratio = lambda s: rate_in(tab, s) / rate_in(seal, s) if rate_in(seal, s) else float("inf")
    comb, jar = ratio({400}), ratio({JAR})
    inv = {k: (rate_in(seal, s) / rate_in(tab, s) if rate_in(tab, s) else float("inf"))
           for k, s in (("man", {90}), ("arrow", {520}), ("burden", {151, 156}))}
    ok = comb >= 3 and .7 <= jar <= 1.4 and all(v >= 1.5 for v in inv.values())
    fal = comb < 1.5 or not (.5 <= jar <= 2.0)
    return verdict(ok, fal), (f"comb tablet/seal {comb:.2f}; jar tablet/seal {jar:.2f}; seal/tablet "
                              + ", ".join(f"{k} {v:.2f}" for k, v in inv.items())
                              + f" (tablets {len(tab)}, seals {len(seal)})")


def is_count(q):
    return any(q[i - 1] in SHORT and 3 <= SHORT[q[i - 1]] <= 8 and (q[i] in TREE or q[i] == CRESCENT)
               for i in range(1, len(q)))


def q7(R):
    R2 = [r for r in R if len(r["seq"]) >= 2]
    cnt = [r for r in R2 if is_count(r["seq"])]; oth = [r for r in R2 if not is_count(r["seq"])]
    jf = lambda rs: sum(r["seq"][-1] == JAR for r in rs) / len(rs) if rs else float("nan")
    a, b = jf(cnt), jf(oth)
    ratio = a / b if b else float("nan")
    return verdict(ratio <= .60, len(cnt) >= 40 and ratio >= .90), f"jar-final: count texts {f(a)} (n={len(cnt)}), others {f(b)}; ratio {f(ratio)}"


def tablet_objects(R):
    objs = defaultdict(list)
    for r in R:
        if r["cls"] == "tablet" and not r["obj"].startswith("_anon"):
            objs[(r["site"], r["obj"])].append(r)
    return {k: v for k, v in objs.items() if len(v) >= 2}


def leaf_face(q):
    return len(q) == 2 and q[0] in TALL and q[1] == LEAF


def q8(R, Rall):
    two = tablet_objects(Rall)
    with_leaf = [v for v in two.values() if any(leaf_face(r["seq"]) for r in v)]
    share = len(with_leaf) / len(two) if two else float("nan")
    tab = [r for r in R if r["cls"] == "tablet" and r["seq"]]; seal = [r for r in R if r["cls"] == "seal" and r["seq"]]
    lf = lambda rs: sum(r["seq"][-1] == LEAF for r in rs) / len(rs) if rs else float("nan")
    lr = lf(tab) / lf(seal) if lf(seal) else float("inf")
    vals = Counter(TALL[r["seq"][0]] for v in with_leaf for r in v if leaf_face(r["seq"]))
    ok = share >= .30 and lr >= 5 and all(vals[k] >= 5 * vals[1] and vals[k] > 0 for k in (2, 3, 4))
    fal = bool(two) and (share < .15 or (vals and vals.most_common(1)[0][0] == 1))
    return verdict(ok, fal), (f"multi-faced tablets {len(two)}, with tall-stroke+leaf face {f(share)}; "
                              f"leaf-final tablet/seal {lr:.2f}; face values {dict(sorted(vals.items()))}")


def q9(Rall):
    names = defaultdict(Counter)
    for v in tablet_objects(Rall).values():
        leaf = [r for r in v if leaf_face(r["seq"])]
        other = [r for r in v if not leaf_face(r["seq"]) and r["seq"]]
        if len(leaf) == 1 and other:
            den = TALL[leaf[0]["seq"][0]]
            for r in other:
                names[tuple(r["seq"])][den] += 1
    elig = {k: c for k, c in names.items() if sum(c.values()) >= 5}
    locked = sum(1 for c in elig.values() if c.most_common(1)[0][1] / sum(c.values()) >= .80)
    share = locked / len(elig) if elig else float("nan")
    return verdict(share >= .50, bool(elig) and share < .25), f"names with >=5 leaf-face attestations: {len(elig)}, locked to one denomination: {locked} ({f(share)})"


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact p for enrichment in cell a of [[a, b], [c, d]]."""
    n1, n2, k, N = a + b, c + d, a + c, a + b + c + d
    denom = math.comb(N, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / denom


def q10(R):
    seals = [r for r in R if r["cls"] == "seal" and r["symbol"] and r["seq"]]
    n = len(seals)
    emb = Counter(r["symbol"] for r in seals)
    sg = Counter(s for r in seals for s in set(r["seq"]))
    tests = []
    for e, ne in emb.items():
        if ne < 12:
            continue
        for s, ns in sg.items():
            if ns < 5:
                continue
            a = sum(1 for r in seals if r["symbol"] == e and s in r["seq"])
            tests.append((fisher_greater(a, ne - a, ns - a, n - ne - ns + a), s, e, a, ne, ns))
    tests.sort()
    m = len(tests); surv = []
    for i, t in enumerate(tests, 1):   # Benjamini-Hochberg step-up
        if t[0] <= .05 * i / m:
            surv = tests[:i]
    det = f"inscribed seals with emblem {n}, tests {m}, surviving BH-FDR 5%: {len(surv)}"
    if surv:
        det += "; " + ", ".join(f"sign {s} x {e} ({a}/{ne} vs {ns} overall, p={p:.1e})" for p, s, e, a, ne, ns in surv[:5])
    if n < 500:
        return "INDETERMINATE", det + " (<500 seals)"
    # the falsifier excludes enrichment due to duplicate impressions; records are deduplicated
    return verdict(not surv, bool(surv)), det


def q11(R):
    seal_sites = defaultdict(set)
    for r in R:
        if r["cls"] == "seal" and len(r["seq"]) >= 3:
            seal_sites[tuple(r["seq"])].add(r["site"])
    matches = [(r["site"], seal_sites[tuple(r["seq"])]) for r in R
               if r["cls"] == "sealing" and tuple(r["seq"]) in seal_sites]
    cross = sum(1 for site, ss in matches if site not in ss)
    share = cross / len(matches) if matches else float("nan")
    return verdict(share >= .80, len(matches) >= 20 and share < .50), f"sealings matching a seal text: {len(matches)}, seal at a different site: {cross} ({f(share)})"


def q12(R):
    texts = sorted({tuple(r["seq"]) for r in R if r["cls"] == "seal" and len(r["seq"]) >= 3})
    bylen = defaultdict(list)
    for t in texts:
        bylen[len(t)].append(t)
    twin, first, pre = set(), 0, 0
    for L, ts in bylen.items():
        # bucket by text with one position masked
        for i in range(L):
            groups = defaultdict(list)
            for t in ts:
                groups[t[:i] + t[i + 1:]].append(t)
            for g in groups.values():
                if len(g) > 1:
                    twin.update(g)
                    pairs = len(g) * (len(g) - 1) // 2
                    if i == 0:
                        first += pairs
                    if i == L - 2:
                        pre += pairs
    share = len(twin) / len(texts) if texts else float("nan")
    ok = share >= .20 and pre > first
    fal = share < .10 or (pre and first >= 1.5 * pre) or (not pre and first > 0)
    return verdict(ok, fal), f"distinct seal texts >=3 signs {len(texts)}, with a one-substitution twin {f(share)}; twin pairs differing before the ending {pre}, at first position {first}"


PERIOD_ORDER = [  # (site, field index in period tuple, ordered bins of regexes)
    ("Harappa", 0, [r"Period 3A|Period 3B(?!/)", r"Period 3C", r"Period 4|Period 5"]),
    ("Mohenjo-daro", 1, [r"^Early$", r"^Intermediate$", r"^Late$"]),
]


def q13(R):
    out, ok, fal, any_ = [], True, False, False
    for site, fi, bins in PERIOD_ORDER:
        seals = [r for r in R if r["cls"] == "seal" and r["site"] == site and r["seq"]]
        grouped = [[r for r in seals if re.search(b, r["period"][fi])] for b in bins]
        grouped = [g for g in grouped if g]
        if len(grouped) < 2:
            out.append(f"{site}: <2 dated periods"); continue
        any_ = True
        e, l = grouped[0], grouped[-1]
        jf = lambda rs: sum(r["seq"][-1] == JAR for r in rs) / len(rs)
        op = lambda rs: sum(r["seq"][0] in OPENERS for r in rs) / len(rs)
        dj, do = abs(jf(e) - jf(l)) * 100, abs(op(e) - op(l)) * 100
        ok &= dj <= 10 and do <= 10
        fal |= (dj >= 20 or do >= 20) and len(e) >= 100 and len(l) >= 100
        out.append(f"{site}: earliest n={len(e)} vs latest n={len(l)}: jar-final diff {dj:.1f} pts, opener diff {do:.1f} pts")
    return (verdict(ok, fal) if any_ else "INDETERMINATE"), "; ".join(out)


def q14(R, path):
    """Criterion for keys: every key in new_data/keys/*.json (default: Yajnadevam's
    published key vs Monier-Williams) on this corpus: real rate vs within-text
    scrambled, and vs 200 value-shuffled keys."""
    import indus_core as C
    import validator as V
    keys = sorted(glob.glob(os.path.join(ROOT, "new_data", "keys", "*.json")))
    ks = [(os.path.basename(k), {str(a): b for a, b in json.load(open(k)).items()}) for k in keys]
    if not ks:
        if not os.path.exists(os.path.join(C.RAW, "xlits.csv")):
            return "INDETERMINATE", "no keys and no published key (run scripts/fetch_data.sh)"
        ks = [("yajnadevam-xlits", {str(int(s)): v for s, v in V.published_key().items()})]
    lex = C.load_sanskrit()
    fs = C.forms(lex, "full")
    texts = [[[str(s) for s in r["seq"]]] for r in R if len(r["seq"]) >= 2]
    rng = random.Random(0)
    out, fails = [], 0
    for name, key in ks:
        real = V.rate(key, texts, fs, C.MINLEN["full"])
        scr = [V.rate(key, V.scramble(texts, i), fs, C.MINLEN["full"]) for i in range(20)]
        vals = list(key.values()); nulls = []
        for _ in range(200):
            rng.shuffle(vals)
            nulls.append(V.rate(dict(zip(key, vals)), texts, fs, C.MINLEN["full"]))
        mu = sum(nulls) / len(nulls); sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** .5
        smu = sum(scr) / len(scr); ssd = (sum((x - smu) ** 2 for x in scr) / len(scr)) ** .5
        p95 = sorted(nulls)[int(.95 * len(nulls))]
        z = (real - mu) / sd if sd else 0; zs = (real - smu) / ssd if ssd else 0
        ok = real > p95 and z >= 2 and zs >= 2
        fails += not ok
        out.append(f"{name}: real {real:.4f}, scrambled {smu:.4f} (z {zs:.1f}), value-shuffled {mu:.4f} ± {sd:.4f} (z {z:.1f}, 95th pct {p95:.4f}) -> {'accepted' if ok else 'rejected'}")
    return ("PASS (all keys rejected)" if fails == len(ks) else "KEY ACCEPTED"), "; ".join(out) + \
        " [criterion for keys; Sanskrit lexicon, full forms >=5 letters]"


def run(path):
    Rall = load(path)
    R = dedup([r for r in Rall if len(r["seq"]) >= 2])
    rep = [("Q1", q1(R)), ("Q2", q2(R)), ("Q3", q3(R)), ("Q4", q4(R)), ("Q5", q5(R)), ("Q6", q6(R)),
           ("Q7", q7(R)), ("Q8", q8(R, Rall)), ("Q9", q9(Rall)), ("Q10", q10(R)), ("Q11", q11(R)),
           ("Q12", q12(R)), ("Q13", q13(R)), ("Q14", q14(R, path))]
    print(f"\n## {os.path.relpath(path, ROOT)}: {len(Rall)} records, {len(R)} after dedup (>=2 signs)\n")
    print("| id | result | detail |\n|---|---|---|")
    for q, (v, d) in rep:
        print(f"| {q} | {v} | {d} |")
    return rep


def main(paths):
    if not paths:
        paths = sorted(p for p in glob.glob(os.path.join(ROOT, "new_data", "*")) if p.endswith((".json", ".csv")))
    if not paths:
        print("new_data/ is empty: running on the project's own corpora (IN-SAMPLE, harness check only)")
        paths = [os.path.join(ROOT, "data", "derived", "merged-corpus-reading-order.json"),
                 os.path.join(ROOT, "data", "im77", "im77_corpus_lines.csv")]
        y = os.path.join(ROOT, "data", "raw", "inscriptions.csv")
        if os.path.exists(y):
            paths.append(y)
    for p in paths:
        run(p)


if __name__ == "__main__":
    main(sys.argv[1:])
