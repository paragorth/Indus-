#!/usr/bin/env python3
"""la83 corrected restoration-aware Linear A parser: data/LinearAInscriptions.js -> data/corpus_ra_v2.json.

corpus_ra.json (tools/la71_parse.py) is NOT overwritten.  v2 = v1 plus the rule changes below, found by the
la83 audit (tools/la83_audit.py -> data/la83_audit.json).  Token schema, status values and versions
('all' | 'rd' | 'read' | 'rnd') are those of la71_parse; load with la83_parse.load('rd').

SigLA flags: the SigLA pages used by la71 are not on disk (git-ignored checkpoint).  For every document whose
source pairs are untouched by a rule below, v2 copies the v1 tokens verbatim (so all la71 flags, SigLA and
stipple included, are kept).  For a document touched by a rule, the document is re-parsed with the la71 flag
rules and the SigLA-derived flags (unsure, unid, variant, nodraw, worn) are carried over from the v1 tokens
by aligning v1 and v2 tokens (Needleman-Wunsch on token keys).

Rule changes (each logged per document in data/corpus_ra_v2.json['_log'] and data/la83_parse_log.json):
 R1  truncation: la71 cut every record to min(len(words), len(transliteratedWords)).  When the
     transliteration list is short (KNZg57b: 10 words, 0 transliterations), the missing transliterations
     are rebuilt from the Unicode signs with the corpus-wide majority value of each sign code (built from
     pairs whose sign count equals their component count).  New flag 'fromuni' (counts as damage, so these
     tokens are in 'rd', not in 'read').
 R2  number glyphs inside a sign pair (HTZb162 'DI-<3>-TI-DU', PSZa2 '<6>-JA-TI'): the pair is split at the
     boundary between Linear A signs and AEGEAN NUMBER characters; the number becomes a 'num' token (v1 made a
     logogram whose value was the Unicode digit).  Fraction glyphs that an edition reads as syllables
     (LACHZa1 SE-JE, HTWa1025/1026 E, MAZb15 *906) are left as the transliteration has them (editorial).
 R3  transliteration that is itself Unicode (KH79 '<PA><DA>' with break marks, MYZf2 a Linear B character,
     GOWc3 '<20>,<2>'): rebuilt from the Unicode (majority sign values; Linear B syllable B043 A3 -> A3;
     comma-separated numerals -> two numbers flagged 'unsure').  v1 made one logogram per pair.
 R4  erasure brackets around a sign value ('[[NI]]', '[[VIN]]', '[[KI+MU]]'): the brackets are stripped
     before classification and the token keeps 'erased' (v1 made logograms named '[[NI]]', '[[VIN]]').
 R5  ASCII digits written for the subscript in a syllable of a hyphenated word ('TO-PA3-DI') -> PA3 with
     a subscript (v1 split the word into TO | logogram PA3 | DI).
 R6  an unencoded (private-use) character inside a hyphenated word (HT17 'RA-<U+FD1EB>-TI') -> one word
     RA-*0-TI with the unknown sign written *0 and flagged 'unid' (v1: RA | logogram | TI).
 R7  tally strokes '|||||||||||||' (PH11) -> 'unk' (v1: a logogram named '|||||||||||||').
 R8  site field contradicting the siglum (KHZc106 site 'Knossos'; every other KH record is Khania) -> the
     siglum's site; 'site_source' keeps the record's value.
 R9  join fragments stored twice (KH79 inside KH79+89; HTWa1733 inside HTWa1845+1733): the fragment record
     is kept with 'superseded_by' and empty 'tokens' (its tokens in 'tokens_superseded'), so its signs are
     not counted twice.
 R10 editorial readings that differ from the record (la82 side audit, sign identities only):
     KHZc106 SA-SA-RA-ME -> MI-RA-O (AB 73-60-61, broken at right: edgeR);
     THEZb15 (= THE Zb 14 in the record's own 'names') RE-SA 2 -> SE-*332 2 (AB 09-A332).
     Tokens get flag 'editor' (counts as damage).  Set LA83_NO_EDITOR=1 to build without R10.
Not changed (logged by the audit only): the 'transcription' / 'parsedInscription' fields (an older or
differently notated copy; KNZg57b's is a copy of KNZg57a's), SigLA variant readings (already 'damaged' in v1),
the possible duplicate PEWy5 / PEZg5 (same text, different support; unresolved), *4xx-VS single glyphs.
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_corpus as bc
import la71_parse as P

DATA = os.path.join(HERE, '..', 'data')
GAP = P.GAP
SIGLA_FLAGS = {'unsure', 'unid', 'variant', 'nodraw', 'worn'}
P.DAMAGE_FLAGS |= {'fromuni', 'editor'}
SUPERSEDED = {'KH79': 'KH79+89', 'HTWa1733': 'HTWa1845+1733'}
EDITOR = {
    'KHZc106': {'pairs': [('\U00010649', 'X')], 'words': [GAP and None], 'note': ''},
}
SUB = {'2': '₂', '3': '₃'}


def ucode(ch):
    m = re.match(r'LINEAR A SIGN (AB\d+|A\d+[A-Z]?)', unicodedata.name(ch, ''))
    return m.group(1) if m else None


def ukind(ch):
    if ch == GAP: return 'gap'
    if bc.num_value(ch) is not None: return 'num'
    if bc.frac_letter(ch) is not None: return 'frac'
    if ucode(ch): return 'sign'
    if unicodedata.name(ch, '').startswith('LINEAR B SYLLABLE'): return 'lb'
    return 'other'


def majority(raw):
    votes = defaultdict(Counter)
    for v in raw.values():
        for w, t in zip(v['words'], v['transliteratedWords']):
            sg = [c for c in w if ukind(c) == 'sign']
            comps = [x for x in (t or '').strip().split('-') if x]
            if sg and len(sg) == len(comps) and '+' not in t and not any(ukind(c) in ('num', 'frac') for c in w):
                for c, x in zip(sg, comps):
                    votes[ucode(c)][x] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items()}


def from_unicode(w, maj):
    """transliteration of a Unicode pair: signs -> majority values joined by '-', numbers -> digits."""
    parts = []
    for c in w:
        k = ukind(c)
        if k == 'sign': parts.append(maj.get(ucode(c), '*' + re.sub(r'\D', '', ucode(c)).lstrip('0')))
        elif k == 'lb':
            m = re.search(r'B\d+ ([A-Z0-9]+)$', unicodedata.name(c))
            v = m.group(1) if m else '?'
            parts.append(re.sub(r'([A-Z])([23])$', lambda z: z.group(1) + SUB[z.group(2)], v))
    return '-'.join(parts)


def split_mixed(w, t, maj):
    """R2: split a pair whose Unicode mixes Linear A signs and AEGEAN NUMBER characters."""
    runs = []
    for c in w:
        k = ukind(c)
        k = 'num' if k == 'num' else ('gap' if k == 'gap' else 'sign')
        if k == 'gap' and runs: runs[-1][1] += c; continue
        if runs and (runs[-1][0] == k or runs[-1][0] == 'gap'):
            if runs[-1][0] == 'gap': runs[-1][0] = k
            runs[-1][1] += c
        else:
            runs.append([k, c])
    comps = [x for x in t.split('-') if x and not any(ukind(c) == 'num' for c in x)]
    out = []
    for k, s in runs:
        if k == 'num':
            out.append((s, str(sum(bc.num_value(c) for c in s if ukind(c) == 'num'))))
        else:
            n = sum(1 for c in s if ukind(c) == 'sign')
            take, comps = comps[:n], comps[n:]
            tr = '-'.join(take) if len(take) == n else from_unicode(s, maj)
            out.append((s, tr))
    return out


def fix_record(k, v, maj, raw, use_editor=True):
    """return (pairs, extra_flags_per_pair, site, log) for one record."""
    W, T = list(v.get('words', [])), list(v.get('transliteratedWords', []))
    log = []; extra = []
    if len(T) < len(W):
        log.append('R1 %d transliterations rebuilt from Unicode' % (len(W) - len(T)))
        T = T + [from_unicode(w, maj) if any(ukind(c) in ('sign', 'lb') for c in w) else
                 ('\n' if w == '\n' else (str(sum(bc.num_value(c) for c in w if ukind(c) == 'num')) if any(ukind(c) == 'num' for c in w) else w))
                 for w in W[len(T):]]
        extra_from = len(v.get('transliteratedWords', []))
    else:
        extra_from = None
    pairs = []
    for i, (w, t) in enumerate(zip(W, T)):
        f = set()
        if extra_from is not None and i >= extra_from and any(ukind(c) in ('sign',) for c in w): f.add('fromuni')
        tt = t
        # R3 transliteration is Unicode
        if tt and any(ukind(c) in ('sign', 'lb', 'num') for c in tt) and not re.search(r'[A-Za-z]', tt):
            if ',' in w:
                log.append('R3 comma-separated numerals split: %r' % w)
                for j, piece in enumerate(w.split(',')):
                    pairs.append((piece, str(sum(bc.num_value(c) for c in piece if ukind(c) == 'num')), f | {'unsure'}))
                continue
            if all(ukind(c) in ('num', 'gap') for c in w):
                pass  # e.g. translit '120' style handled by classify; Unicode numbers already parse
            else:
                new = from_unicode(w, maj)
                log.append('R3 translit rebuilt from Unicode: %r -> %s' % (tt, new)); tt = new
        # R2 number glyphs inside a sign pair
        if any(ukind(c) == 'sign' for c in w) and any(ukind(c) == 'num' for c in w):
            sp = split_mixed(w, tt, maj)
            log.append('R2 split %r -> %s' % (tt, [x[1] for x in sp]))
            for s, x in sp: pairs.append((s, x, set(f)))
            continue
        # R4 erasure brackets around a value
        if re.fullmatch(r'\[\[[^\[\]]+\]\]', tt or '') and not re.fullmatch(r'\[\[[\d\s¹⁄₀-₉]+\]\]', tt):
            log.append('R4 erasure brackets stripped: %s' % tt); f.add('erased'); tt = tt[2:-2]
        # R5 ASCII subscript digits in a hyphenated word
        if '-' in (tt or '') and '+' not in tt:
            comps = tt.split('-'); new = []
            for c in comps:
                m = re.fullmatch(r'([A-Z]{1,2})([23])', c)
                new.append(m.group(1) + SUB[m.group(2)] if m else c)
            # R6 unencoded character inside a word
            for j, c in enumerate(new):
                if len(c) == 1 and ord(c) > 0xFFFF and ukind(c) == 'other':
                    new[j] = '*0'; f.add('unid')
            if new != comps:
                log.append('R5/R6 word components %s -> %s' % (tt, '-'.join(new))); tt = '-'.join(new)
        # R7 tally strokes
        if tt and re.fullmatch(r'\|+', tt):
            log.append('R7 tally strokes -> unk: %s' % tt); tt = '—'
        pairs.append((w, tt, f))
    site = v.get('site')
    m = re.match(r'([A-Z]+?)(?=Z[a-g]|W[a-g]|fr|tab|\d|$)', k)
    if m:
        pre = m.group(1)
        others = Counter(ov.get('site') for ok, ov in raw.items() if ok != k and
                         (re.match(r'([A-Z]+?)(?=Z[a-g]|W[a-g]|fr|tab|\d|$)', ok) or [None, None])[1] == pre)
        if others and len(others) == 1 and site not in others:
            log.append('R8 site %s -> %s (siglum %s)' % (site, next(iter(others)), pre)); site = next(iter(others))
    if use_editor and k in EDITORIAL:
        ed = EDITORIAL[k]
        log.append('R10 editorial reading: %s' % ed['note'])
        pairs = [(w, t, set(fl) | {'editor'}) for w, t, fl in ed['pairs']]
    return pairs, site, log


# R10 editorial readings, written as source-style pairs (Unicode, transliteration)
EDITORIAL = {
    'KHZc106': {'pairs': [('\U00010646\U00010634\U00010635' + GAP, 'MI-RA-O', set())],
                'note': 'KH Zc 106 = AB 73-60-61-[ (MI-RA-O, broken right); record SA-SA-RA-ME'},
    'THEZb15': {'pairs': [('\U00010608\U0001076e', 'SE-*332', set()), ('\U00010108', '2', set())],
                'note': 'THE Zb 14 = AB 09-A332 2 (SE-*332 2); record RE-SA 2'},
}


def parse_pairs(pairs):
    """la71_parse.parse_doc on explicit (word, translit, extra flags) pairs."""
    W = [p[0] for p in pairs]; T = [p[1] for p in pairs]
    toks = []
    for i, (w, t, ex) in enumerate(pairs):
        f = P.pair_flags(W, T, i) | set(ex)
        sub = bc.classify(w, t)
        for tk in sub:
            tk = dict(tk)
            if tk['t'] in ('nl', 'div'):
                toks.append(tk); continue
            tk['fl'] = set(f)
            toks.append(tk)
    merged = []
    for tk in toks:
        if tk['t'] == 'frac' and merged and merged[-1]['t'] == 'num':
            merged[-1]['frac'] = merged[-1]['frac'] + tk['v']; merged[-1]['fl'] |= tk['fl']
        elif tk['t'] == 'frac':
            merged.append({'t': 'num', 'v': 0, 'frac': tk['v'], 'fl': tk['fl']})
        else:
            merged.append(tk)
    return merged


def tkey(t):
    if t['t'] == 'word': return 'w:' + '-'.join(t['s'])
    if t['t'] in ('logo', 'unk'): return t['t'] + ':' + str(t.get('v'))
    if t['t'] == 'num': return 'n:%s:%s' % (t['v'], ''.join(t['frac']))
    return t['t']


def finish(toks, old):
    """carry SigLA-derived flags from v1 tokens (aligned) and set status."""
    if old:
        al = P.nw([tkey(t) for t in toks], [tkey(t) for t in old])
        for i, j in al:
            if i is not None and j is not None and 'fl' in toks[i]:
                toks[i]['fl'] |= set(old[j].get('fl', [])) & SIGLA_FLAGS
    out = []
    for tk in toks:
        if tk['t'] in ('nl', 'div'):
            out.append(tk); continue
        fl = tk.pop('fl')
        tk['st'] = P.status_of(fl, tk['t'])
        tk['fl'] = sorted(fl) + (['illegible'] if tk['t'] == 'unk' else [])
        out.append(tk)
    return out


def build(use_editor=True):
    raw = bc.load_raw()
    v1 = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus_ra.json')))}
    maj = majority(raw)
    out = []; LOG = {}
    for k, v in raw.items():
        pairs, site, log = fix_record(k, v, maj, raw, use_editor)
        d1 = v1[k]
        e = {x: y for x, y in d1.items() if x != 'tokens'}
        src_pairs = list(zip(v.get('words', []), v.get('transliteratedWords', [])))
        touched = [(w, t) for w, t, _ in pairs] != src_pairs or any(f for _, _, f in pairs)
        if touched:
            toks = finish(parse_pairs(pairs), d1['tokens'])
        else:
            toks = d1['tokens']
        if site != v.get('site'):
            e['site_source'] = v.get('site'); e['site'] = site
        if k in SUPERSEDED and SUPERSEDED[k] in raw:
            log.append('R9 superseded by join %s' % SUPERSEDED[k])
            e['superseded_by'] = SUPERSEDED[k]; e['tokens_superseded'] = toks; toks = []
        e['tokens'] = toks
        if log:
            LOG[k] = log; e['la83'] = log
        out.append(e)
    return out, LOG


def version(corpus, ver='rd'):
    return P.version(corpus, ver)


def load(ver='rd', path=None):
    """Default loader: la83_parse.load('rd') (same versions as la71_parse.load)."""
    return P.version(json.load(open(path or os.path.join(DATA, 'corpus_ra_v2.json'))), ver)


def main():
    use_ed = os.environ.get('LA83_NO_EDITOR') != '1'
    corpus, LOG = build(use_ed)
    fn = os.path.join(DATA, 'corpus_ra_v2.json' if use_ed else 'corpus_ra_v2_noeditor.json')
    json.dump(corpus, open(fn, 'w'), ensure_ascii=False, separators=(',', ':'))
    json.dump(LOG, open(os.path.join(DATA, 'la83_parse_log.json'), 'w'), ensure_ascii=False, indent=1)
    v1 = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus_ra.json')))}
    ch = 0; dt = Counter()
    for d in corpus:
        a = [tkey(t) + ':' + t.get('st', '') for t in d['tokens']]
        b = [tkey(t) + ':' + t.get('st', '') for t in v1[d['id']]['tokens']]
        if a != b: ch += 1
        dt['v2'] += sum(1 for t in d['tokens'] if t['t'] not in ('nl', 'div'))
        dt['v1'] += sum(1 for t in v1[d['id']]['tokens'] if t['t'] not in ('nl', 'div'))
    print('docs', len(corpus), 'docs changed', ch, 'tokens', dict(dt), 'logged', len(LOG))
    for k, l in LOG.items(): print(k, l)


if __name__ == '__main__':
    main()
