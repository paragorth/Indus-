#!/usr/bin/env python3
"""la22 data layer: Linear A documents WITH their breaks, from the raw lineara.xyz file.

The earlier corpus.json drops the break marker U+1076B (shown '#').  In the lineara.xyz
'transcription' field '#' sits at the edges of physical lines that are broken off and on
lines of its own where whole lines are lost, so it is the edition's lacuna mark.  Here every
'#' becomes a {'t':'gap'} token, kept in reading order.

Sign identity is the Unicode name code (AB081, A301 ... -> 'AB81', 'A301'), which is also
the code SigLA uses, so the two editions can be aligned without transliteration tables.

Token kinds: w (syllabic word: c=codes, tr=translit), L (logogram/ligature: c, tr),
N (number: v, frac letters), gap, div, nl, unk (edition's '?', '—', '≈').
"""
import json, os, re, subprocess, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la22_ckpt')
GAP = '\U0001076b'


def load_raw():
    p = os.path.join(CK, 'raw.json')
    if not os.path.exists(p):
        js = os.path.join(DATA, 'LinearAInscriptions.js')
        code = ("const fs=require('fs');eval(fs.readFileSync(%r,'utf8').replace('var inscriptions','global.inscriptions'));"
                "const o={};for(const [k,v] of inscriptions)o[k]=v;process.stdout.write(JSON.stringify(o));") % js
        out = subprocess.run(['node', '-e', code], capture_output=True, check=True).stdout
        os.makedirs(CK, exist_ok=True)
        open(p, 'wb').write(out)
    return json.load(open(p))


def num_value(ch):
    c = ord(ch)
    if 0x10107 <= c <= 0x1010F: return c - 0x10106
    if 0x10110 <= c <= 0x10118: return (c - 0x1010F) * 10
    if 0x10119 <= c <= 0x10121: return (c - 0x10118) * 100
    if 0x10122 <= c <= 0x1012A: return (c - 0x10121) * 1000
    return None


def code_of(ch):
    n = unicodedata.name(ch, '')
    m = re.match(r'LINEAR A SIGN (AB|A)(\d+)', n)
    if not m:
        return None
    return m.group(1) + str(int(m.group(2)))


def is_frac(ch):
    return 0x10740 <= ord(ch) <= 0x10755


def frac_letter(ch):
    return unicodedata.name(ch).split()[-1]


def is_syl(code):
    if code is None: return False
    m = re.match(r'(AB|A)(\d+)$', code)
    k, n = m.group(1), int(m.group(2))
    return (k == 'AB' and n < 100) or (k == 'A' and 300 <= n < 400)


def parse_token(w, tr):
    """-> list of tokens, with gap tokens where '#' sits."""
    if w == '\n': return [{'t': 'nl'}]
    if w == '𐄁' or tr == '𐄁': return [{'t': 'div'}]
    lead = w.startswith(GAP); trail = w.endswith(GAP) and len(w) > 1
    chars = [ch for ch in w if ch != GAP]
    out = []
    if lead or (w == GAP): out.append({'t': 'gap'})
    if chars:
        trs = tr.strip()
        if trs in ('?', '—', '≈', '', 'None'):
            out.append({'t': 'unk', 'v': trs})
        elif all(num_value(ch) is not None or is_frac(ch) for ch in chars):
            v = sum(num_value(ch) or 0 for ch in chars)
            fl = [frac_letter(ch) for ch in chars if is_frac(ch)]
            has_int = any(num_value(ch) is not None for ch in chars)
            out.append({'t': 'N', 'v': v if has_int else 0, 'frac': fl})
        else:
            trc = trs.split('-')
            same = len(trc) == len(chars)
            syl, sylt = [], []
            for i, ch in enumerate(chars):
                c = code_of(ch)
                t = trc[i] if same else (c or '?')
                if num_value(ch) is not None or is_frac(ch):
                    if syl: out.append({'t': 'w', 'c': syl, 'tr': sylt}); syl, sylt = [], []
                    out.append({'t': 'N', 'v': num_value(ch) or 0, 'frac': [frac_letter(ch)] if is_frac(ch) else []})
                elif is_syl(c):
                    syl.append(c); sylt.append(t)
                else:
                    if syl: out.append({'t': 'w', 'c': syl, 'tr': sylt}); syl, sylt = [], []
                    out.append({'t': 'L', 'c': c or ('U%X' % ord(ch)), 'tr': t if same else trs})
            if syl: out.append({'t': 'w', 'c': syl, 'tr': sylt})
    if trail: out.append({'t': 'gap'})
    return out


def build():
    raw = load_raw()
    docs = []
    for k, v in raw.items():
        W, T = v.get('words', []), v.get('transliteratedWords', [])
        n = min(len(W), len(T))
        toks = []
        for w, t in zip(W[:n], T[:n]):
            toks.extend(parse_token(w, t))
        # merge adjacent numbers (integer then fraction) and collapse repeated gaps
        merged = []
        for tk in toks:
            if tk['t'] == 'N' and merged and merged[-1]['t'] == 'N' and tk['v'] == 0 and tk['frac']:
                merged[-1]['frac'] = merged[-1]['frac'] + tk['frac']
            elif tk['t'] == 'gap' and merged and merged[-1]['t'] == 'gap':
                continue
            else:
                merged.append(tk)
        docs.append({'id': k, 'site': v.get('site') or '', 'support': v.get('support') or '',
                     'scribe': v.get('scribe') or '', 'toks': merged})
    return docs


def load():
    p = os.path.join(CK, 'docs.json')
    if not os.path.exists(p):
        json.dump(build(), open(p, 'w'), ensure_ascii=False)
    return json.load(open(p))


if __name__ == '__main__':
    from collections import Counter
    D = build()
    json.dump(D, open(os.path.join(CK, 'docs.json'), 'w'), ensure_ascii=False)
    C = Counter(t['t'] for d in D for t in d['toks'])
    print('docs', len(D), C)
    sy = Counter(c for d in D for t in d['toks'] if t['t'] == 'w' for c in t['c'])
    print('syllabic sign types', len(sy), 'tokens', sum(sy.values()))
    # words touching a gap
    nb = 0
    for d in D:
        T = d['toks']
        for i, t in enumerate(T):
            if t['t'] == 'w' and ((i > 0 and T[i-1]['t'] == 'gap') or (i + 1 < len(T) and T[i+1]['t'] == 'gap')):
                nb += 1
    print('words adjacent to a gap', nb)
    print('docs with a gap', sum(any(t['t'] == 'gap' for t in d['toks']) for d in D))
