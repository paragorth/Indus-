#!/usr/bin/env python3
"""la83 step 1: audit data/corpus_ra.json (la71 parse) against its source records (lineara.xyz
LinearAInscriptions.js; SigLA disagreements as already recorded by la71 in the token flags).

Data only.  For every document and every source (word, transliteration) pair it checks:
  T1 pairs lost because words / transliteratedWords differ in length (parser truncates to the shorter)
  T2 pairs whose Unicode signs carry a sign but whose transliteration is blank / None / dash -> 'unk'
  T3 Unicode vs transliteration disagreement inside a pair: number value, sign count, or a sign value that
     contradicts the corpus-wide majority value of that Unicode code (the parser follows the transliteration)
  T4 'transcription' / 'parsedInscription' fields that do not equal the concatenated 'words' (record copied
     from another record or stale)
  T5 a record whose 'names' list holds a second siglum (identity doubt)
  T6 tokens SigLA reads as a different sign (la71 flag 'variant') or not at all ('nodraw')
  T7 published editorial readings that differ from the record (la82_new_material.json side audit)
  T8 parser rule effects: number chars inside a sign pair, non-Linear-A characters, glued logograms,
     ligature pieces dropped, characters ignored
Cause classes: PARSER (our rule), SOURCE (record internally inconsistent), EDITORIAL (record consistent,
differs from another edition).
Output: data/la83_audit.json; prints a summary.
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_corpus as bc

DATA = os.path.join(HERE, '..', 'data')
GAP = '\U0001076b'
BLANK = ('', 'None', '?', '—', '≈')


def code(ch):
    n = unicodedata.name(ch, '')
    m = re.match(r'LINEAR A SIGN (AB\d+|A\d+[A-Z]?)', n)
    return m.group(1) if m else None


def kind(ch):
    if ch == GAP: return 'gap'
    if bc.num_value(ch) is not None: return 'num'
    if bc.frac_letter(ch) is not None: return 'frac'
    if code(ch): return 'sign'
    if ch in '\n': return 'nl'
    if ch == '\U00010101': return 'div'
    return 'other'


def majority_map(raw):
    """code -> Counter of transliteration values, from pairs whose sign count equals the component count."""
    votes = defaultdict(Counter)
    for v in raw.values():
        for w, t in zip(v['words'], v['transliteratedWords']):
            sg = [c for c in w if kind(c) == 'sign']
            comps = [x for x in (t or '').strip().split('-') if x]
            if sg and len(sg) == len(comps) and '+' not in t:
                for c, x in zip(sg, comps): votes[code(c)][x] += 1
    return votes


def audit():
    raw = bc.load_raw()
    ra = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus_ra.json')))}
    votes = majority_map(raw)
    maj = {c: v.most_common(1)[0][0] for c, v in votes.items()}
    rows = []; per = Counter(); docs_hit = defaultdict(set)

    def add(doc, test, cause, detail):
        rows.append({'doc': doc, 'test': test, 'cause': cause, 'detail': detail})
        per[(test, cause)] += 1; docs_hit[(test, cause)].add(doc)

    for k, v in raw.items():
        W, T = v['words'], v['transliteratedWords']
        if len(W) != len(T):
            lost = W[min(len(W), len(T)):]
            nsig = sum(1 for w in lost for c in w if kind(c) == 'sign')
            add(k, 'T1', 'PARSER', {'words': len(W), 'translit': len(T), 'signs_lost': nsig,
                                    'lost_codes': [code(c) for w in lost for c in w if kind(c) == 'sign']})
        for i, (w, t) in enumerate(zip(W, T)):
            ks = [kind(c) for c in w]
            sg = [c for c in w if kind(c) == 'sign']
            tt = (t or '').strip()
            if sg and tt in BLANK:
                add(k, 'T2', 'PARSER', {'pair': i, 'codes': [code(c) for c in sg], 'translit': tt})
            if 'other' in ks:
                oth = ''.join(c for c in w if kind(c) == 'other')
                if oth.strip(' ') and not all(c in '—≈?|' for c in oth) and oth not in ('None',):
                    add(k, 'T8', 'PARSER', {'pair': i, 'chars': ['U+%04X %s' % (ord(c), unicodedata.name(c, '?')) for c in oth]})
            if sg and any(x in ('num', 'frac') for x in ks):
                add(k, 'T8', 'PARSER', {'pair': i, 'mixed_sign_number': w, 'translit': tt})
            # T3 numbers
            if ks and all(x in ('num', 'frac', 'gap') for x in ks) and any(x == 'num' for x in ks):
                iv = sum(bc.num_value(c) for c in w if kind(c) == 'num')
                m = re.match(r'^\s*\[?(\d+)', tt)
                if m and int(m.group(1)) != iv:
                    add(k, 'T3', 'SOURCE', {'pair': i, 'unicode_value': iv, 'translit': tt})
            # T3 signs
            if sg and tt not in BLANK and not any(x in ('num', 'frac') for x in ks):
                comps = [x for x in tt.split('-') if x]
                if '+' not in tt:
                    if len(comps) != len(sg) and len(sg) > 1:   # one glyph = *4xx-VS etc. is notation
                        add(k, 'T3', 'SOURCE', {'pair': i, 'codes': [code(c) for c in sg], 'translit': tt,
                                                'why': 'sign count'})
                    else:
                        bad = [(code(c), x, maj.get(code(c))) for c, x in zip(sg, comps)
                               if maj.get(code(c)) not in (None, x) and votes[code(c)][x] <= 2]
                        if bad:
                            add(k, 'T3', 'SOURCE', {'pair': i, 'translit': tt, 'why': 'value',
                                                    'diff': [{'code': a, 'translit': b, 'majority': c} for a, b, c in bad]})
        sig = lambda z: [c for c in z if kind(c) == 'sign']
        cat = sig(''.join(W))
        for fld in ('transcription', 'parsedInscription'):
            s = sig(v.get(fld) or '')
            if s and s != cat:   # sign content differs (break marks, numbers, fraction notation ignored)
                twin = [o for o, ov in raw.items() if o != k and sig(''.join(ov['words'])) == s]
                add(k, 'T4', 'SOURCE', {'field': fld, 'signs_field': len(s), 'signs_words': len(cat),
                                        'equals_words_of': twin})
        if len(v.get('names') or []) > 1:
            add(k, 'T5', 'SOURCE', {'names': v['names']})
        d = ra.get(k)
        if d is None:
            add(k, 'T0', 'PARSER', 'document missing from corpus_ra'); continue
        for j, tk in enumerate(d['tokens']):
            fl = tk.get('fl', [])
            if 'variant' in fl or 'nodraw' in fl:
                add(k, 'T6', 'EDITORIAL', {'tok': j, 'flag': [f for f in fl if f in ('variant', 'nodraw')],
                                           'lineara': tk.get('s') or tk.get('v')})
        nsig = sum(1 for w in W for c in w if kind(c) == 'sign')
        ntok = sum(len(t['s']) if t['t'] == 'word' else 1 for t in d['tokens'] if t['t'] in ('word', 'logo'))
        if nsig and ntok == 0:
            add(k, 'T9', 'PARSER', {'source_signs': nsig, 'corpus_tokens': 0})
    nm = json.load(open(os.path.join(DATA, 'la82_new_material.json')))
    ed = {'KHZc106': {'record': 'SA-SA-RA-ME (AB31-31-60-13)', 'editors': 'KH Zc 106: AB 73-60-61-[ (MI-RA-O-[)',
                      'source': 'la82 side audit (Kanta et al. / Del Freo, Ariadne Suppl. 5)'},
          'THEZb15': {'record': 'RE-SA 2 (AB27-31), names THEZb15 + THEZb14', 'editors': 'THE Zb 14: AB 09-A332 2 (SE-*332 2)',
                      'source': 'la82 side audit'}}
    for k, x in ed.items():
        add(k, 'T7', 'EDITORIAL', x)
    return rows, per, docs_hit, raw, ra, nm


def main():
    rows, per, dh, raw, ra, _ = audit()
    out = {'n_docs_source': len(raw), 'n_docs_corpus_ra': len(ra),
           'summary': {'%s/%s' % k: {'rows': n, 'docs': len(dh[k])} for k, n in sorted(per.items())},
           'rows': rows}
    json.dump(out, open(os.path.join(DATA, 'la83_audit.json'), 'w'), ensure_ascii=False, indent=0)
    print(json.dumps(out['summary'], indent=0))
    for r in rows:
        if r['test'] not in ('T6',): print(r['doc'], r['test'], r['cause'], json.dumps(r['detail'], ensure_ascii=False)[:220])


if __name__ == '__main__':
    main()
