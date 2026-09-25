"""Shared data loading for the Indus key-search experiments.

Everything is reduced to one small Latin alphabet so that Tamil and Sanskrit
are compared on equal terms ("skeletonised by stripping diacritics"):

  consonants  k g c j t d n p b m y r l v s h
  vowels      a i u e o          (length dropped; ai -> e, au -> o)

Retroflex/dental, aspirated/plain and the three nasals and sibilants merge.
Tamil has no voiced letters, so it never uses g j d b except via Grantha j.
"""
import csv
import os
import random
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")

VOWELS = set("aiueo")
DAMAGE = "000"   # damaged / illegible sign
SPACE = "999"    # explicit word space in Yajnadevam's coding

# ---------------------------------------------------------------- corpus


def load_corpus(path=None):
    """Yajnadevam's corpus as a list of (object_id, [segments]).

    Each segment is a list of sign codes in *reading order*.  The file stores
    every text in normalised left-to-right glyph order (the jar sign 740, a
    text-final sign, comes first in 851/1241 R/L texts), so tokens are reversed.
    Damage (000), '/' (line break), and spaces (999) split a text into
    segments; windows never cross a segment boundary.
    """
    path = path or os.path.join(RAW, "inscriptions.csv")
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            segs = []
            for line in r["text"].split("/"):
                toks = re.findall(r"\d{3}", line)[::-1]
                cur = []
                for t in toks:
                    if t in (DAMAGE, SPACE):
                        if cur:
                            segs.append(cur)
                        cur = []
                    else:
                        cur.append(t)
                if cur:
                    segs.append(cur)
            if segs:
                out.append((r["id"], segs))
    # line order in multi-line texts is ambiguous, so lines are separate segments
    return out


def unique_texts(corpus):
    """Deduplicate identical texts (same segment sequence); returns list of segment-lists."""
    seen, out = set(), []
    for _, segs in corpus:
        k = "|".join("-".join(s) for s in segs)
        if k not in seen:
            seen.add(k)
            out.append(segs)
    return out


def split_texts(texts, seed=0):
    """50/50 train / held-out split by (unique) text."""
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    return [texts[i] for i in idx[:h]], [texts[i] for i in idx[h:]]


def sign_freq(texts):
    c = Counter()
    for segs in texts:
        for s in segs:
            c.update(s)
    return c


# ---------------------------------------------------------------- Tamil

_TA_CONS = {
    "க": "k", "ங": "n", "ச": "c", "ஞ": "n", "ட": "t", "ண": "n", "த": "t",
    "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "l", "ள": "l", "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "s", "ஸ": "s",
    "ஹ": "h", "ஶ": "s",
}
_TA_VOW = {"அ": "a", "ஆ": "a", "இ": "i", "ஈ": "i", "உ": "u", "ஊ": "u",
           "எ": "e", "ஏ": "e", "ஐ": "e", "ஒ": "o", "ஓ": "o", "ஔ": "o"}
_TA_SIGN = {"ா": "a", "ி": "i", "ீ": "i", "ு": "u", "ூ": "u", "ெ": "e",
            "ே": "e", "ை": "e", "ொ": "o", "ோ": "o", "ௌ": "o"}
_PULLI = "்"


def tamil_to_latin(w):
    out, i = [], 0
    while i < len(w):
        ch = w[i]
        if ch in _TA_CONS:
            c = _TA_CONS[ch]
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt == _PULLI:
                out.append(c); i += 2
            elif nxt in _TA_SIGN:
                out.append(c + _TA_SIGN[nxt]); i += 2
            elif nxt == "ௗ":  # au length mark
                out.append(c + "o"); i += 2
            else:
                out.append(c + "a"); i += 1
        elif ch in _TA_VOW:
            out.append(_TA_VOW[ch]); i += 1
        else:  # aytham, punctuation, anything else: drop
            i += 1
    return "".join(out)


def load_tamil(path=None):
    """TamilVU dictionary headwords, reduced.  Verbal nouns in -(t)tal also
    contribute their stem (curuttutal -> curuttu), which is how DEDR cites verbs."""
    path = path or os.path.join(RAW, "tamilvu_dictionary_words.txt")
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = tamil_to_latin(line.strip())
            if not w:
                continue
            words.add(w)
            for suf in ("ttal", "tal"):
                if w.endswith(suf) and len(w) > len(suf) + 1:
                    words.add(w[: -len(suf)])
                    break
    return {w for w in words if 2 <= len(w) <= 15}


# ---------------------------------------------------------------- Sanskrit

_SLP = {
    "a": "a", "A": "a", "i": "i", "I": "i", "u": "u", "U": "u", "f": "r", "F": "r",
    "x": "l", "X": "l", "e": "e", "E": "e", "o": "o", "O": "o",
    "k": "k", "K": "k", "g": "g", "G": "g", "N": "n",
    "c": "c", "C": "c", "j": "j", "J": "j", "Y": "n",
    "w": "t", "W": "t", "q": "d", "Q": "d", "R": "n",
    "t": "t", "T": "t", "d": "d", "D": "d", "n": "n",
    "p": "p", "P": "p", "b": "b", "B": "b", "m": "m",
    "y": "y", "r": "r", "l": "l", "v": "v", "S": "s", "z": "s", "s": "s",
    "h": "h", "M": "m", "H": "",
}


def slp1_to_latin(w):
    return "".join(_SLP.get(ch, "") for ch in w)


def load_sanskrit(path=None):
    """Monier-Williams headwords (<k1>, SLP1), reduced."""
    path = path or os.path.join(RAW, "mw.txt")
    words = set()
    rx = re.compile(r"<k1>([^<]+)<")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("<L>"):
                m = rx.search(line)
                if m:
                    words.add(slp1_to_latin(m.group(1)))
    return {w for w in words if 2 <= len(w) <= 15}


def load_lexicon(lang):
    return load_tamil() if lang == "tamil" else load_sanskrit()


# ---------------------------------------------------------------- forms


def consonants_of(lex):
    c = Counter(ch for w in lex for ch in w if ch not in VOWELS)
    return sorted(c)


def skeleton(w):
    """Consonant skeleton with geminates collapsed: kattu -> kt."""
    out = []
    for ch in w:
        if ch in VOWELS:
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def forms(lex, mode):
    if mode == "full":
        return set(lex)
    return {s for s in (skeleton(w) for w in lex) if s}


def fake_lexicon(lex, seed):
    """Shuffle letters within each word, consonants among consonant slots and
    vowels among vowel slots, so the fake keeps each word's length, letters and
    CV template (hence its syllable structure) but not its identity."""
    rng = random.Random(seed)
    out = set()
    for w in sorted(lex):
        cs = [ch for ch in w if ch not in VOWELS]
        vs = [ch for ch in w if ch in VOWELS]
        rng.shuffle(cs); rng.shuffle(vs)
        ci = vi = 0
        s = []
        for ch in w:
            if ch in VOWELS:
                s.append(vs[vi]); vi += 1
            else:
                s.append(cs[ci]); ci += 1
        out.add("".join(s))
    return out


def fake_forms(lex, mode, seed, tries=30):
    """Fake form set of exactly the same size as forms(lex, mode).

    full: letters shuffled within each word, consonants among consonant slots
          and vowels among vowel slots (keeps length, letters, CV template).
    skel: consonants shuffled within each distinct skeleton, no adjacent
          repeats (a real skeleton never has one).
    A shuffle that collides with a form already produced is retried, so the
    fake set is never smaller (and so never easier to miss) than the real one."""
    rng = random.Random(seed)
    real = sorted(forms(lex, mode))
    out = set()
    for w in real:
        for _ in range(tries):
            if mode == "full":
                cs = [ch for ch in w if ch not in VOWELS]
                vs = [ch for ch in w if ch in VOWELS]
                rng.shuffle(cs); rng.shuffle(vs)
                ci = vi = 0
                s = []
                for ch in w:
                    if ch in VOWELS:
                        s.append(vs[vi]); vi += 1
                    else:
                        s.append(cs[ci]); ci += 1
                f = "".join(s)
            else:
                cs = list(w)
                rng.shuffle(cs)
                f = "".join(cs)
                if any(a == b for a, b in zip(f, f[1:])):
                    continue
            if f not in out:
                break
        else:
            f = w if w not in out else f
        out.add(f)
    return out


def unit_inventory(lex, mode):
    """Values a sign may take.  full: V, CV, CVC syllables.  skel: one consonant
    or nothing (a pure-vowel sign, invisible in the skeleton)."""
    C = consonants_of(lex)
    V = sorted(VOWELS)
    if mode == "skel":
        return C + [""]
    return V + [c + v for c in C for v in V] + [c + v + d for c in C for v in V for d in C]


# Shortest rendered string that may count as a hit.  Chosen on the planted
# positive control (synth.py), before any real-corpus search at these settings:
# full >= 5 letters detects planted Tamil at +7 SD; shorter minima saturate
# (almost every window hits for real and fake lexicons alike).  No skeleton
# minimum gave skel mode any power (planted gap ~0 SD); it is run and reported
# as uninformative.
MINLEN = {"full": 5, "skel": 3}
