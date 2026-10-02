#!/usr/bin/env python3
"""Build a clean machine-readable Linear A corpus from mwenge/lineara.xyz.

Input : data/LinearAInscriptions.js (raw download from
        https://raw.githubusercontent.com/mwenge/lineara.xyz/master/LinearAInscriptions.js)
Output: data/corpus.json  (list of inscriptions)

Only the sign-level fields are used ("words" = Unicode signs, "transliteratedWords"
= conventional Linear-B-derived sign values). The site's "translatedWords" and its
fraction values (e.g. "1/2") are NOT used: fractions are kept as raw sign letters
(J, E, F, K, L, D, ...) decoded from the Unicode names.

Each token: {"t": kind, ...}
  word      {"t":"word","s":["KU","RO"]}
  logo      {"t":"logo","v":"VIN"}           commodity / ideogram / ligature
  num       {"t":"num","v":31,"frac":["JE"]} integer + fraction-sign letters
  frac      {"t":"frac","v":["J"]}           fraction with no integer before it
  div       word divider, nl newline
"""
import json, os, re, subprocess, unicodedata, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

def load_raw():
    js = os.path.join(DATA, 'LinearAInscriptions.js')
    code = ("const fs=require('fs');eval(fs.readFileSync(%r,'utf8').replace('var inscriptions','global.inscriptions'));"
            "const o={};for(const [k,v] of inscriptions)o[k]=v;process.stdout.write(JSON.stringify(o));") % js
    out = subprocess.run(['node', '-e', code], capture_output=True, check=True).stdout
    return json.loads(out)

def num_value(ch):
    c = ord(ch)
    if 0x10107 <= c <= 0x1010F: return c - 0x10106
    if 0x10110 <= c <= 0x10118: return (c - 0x1010F) * 10
    if 0x10119 <= c <= 0x10121: return (c - 0x10118) * 100
    if 0x10122 <= c <= 0x1012A: return (c - 0x10121) * 1000
    return None

def frac_letter(ch):
    c = ord(ch)
    if 0x10740 <= c <= 0x10755:
        n = unicodedata.name(ch)            # e.g. LINEAR A SIGN A707 J
        return n.split()[-1]
    return None

SYL = re.compile(r'^([A-Z]{1,3}[₂₃]?|\*\d+[A-Za-z]?)$')
LOGO_NAMES = {'GRA', 'VIN', 'OLE', 'OLIV', 'AROM', 'CYP', 'VIR', 'GAL', 'TELA', 'HIDE', 'FIC',
              'BOS', 'OVIS', 'CAP', 'SUS', 'MUL', 'FEM', 'CAPm', 'BOSm', 'OVISm', 'SUSm'}
# Unicode Linear A block: syllabograms 10600-10636 (with *-numbers for unknown signs);
# signs >= 10637 (*118 onward) are ideograms/ligatures in this edition.
def is_syllable(comp):
    if comp in LOGO_NAMES: return False
    return bool(SYL.match(comp))

def classify(word_u, translit):
    if word_u == '\n': return [{'t': 'nl'}]
    if word_u == '𐄁' or translit == '𐄁': return [{'t': 'div'}]
    chars = [ch for ch in word_u if ord(ch) != 0x1076B]      # 1076B = line-continuation mark in this edition
    if not chars: return []
    nums = [num_value(ch) for ch in chars]
    fracs = [frac_letter(ch) for ch in chars]
    if all(n is not None or f is not None for n, f in zip(nums, fracs)):
        iv = sum(n for n in nums if n is not None)
        fl = [f for f in fracs if f]
        has_int = any(n is not None for n in nums)
        return [{'t': 'num', 'v': iv, 'frac': fl}] if has_int else [{'t': 'frac', 'v': fl}]
    tr = translit.strip()
    if tr in ('', 'None', '?', '—', '≈'): return [{'t': 'unk', 'v': tr}]
    if '+' in tr and '-' not in tr: return [{'t': 'logo', 'v': tr}]
    comps = tr.split('-')
    # peel trailing / leading logograms that the edition glued to a word (e.g. TA-I-AROM)
    out = []
    syl = []
    for c in comps:
        if is_syllable(c) and not ('+' in c):
            syl.append(c)
        else:
            if syl: out.append({'t': 'word', 's': syl}); syl = []
            out.append({'t': 'logo', 'v': c})
    if syl: out.append({'t': 'word', 's': syl})
    return out

def main():
    raw = load_raw()
    corpus = []
    for k, v in raw.items():
        W, T = v.get('words', []), v.get('transliteratedWords', [])
        if len(W) != len(T):
            n = min(len(W), len(T)); W, T = W[:n], T[:n]
        toks = []
        for w, t in zip(W, T):
            toks.extend(classify(w, t))
        # attach bare fractions to preceding number token
        merged = []
        for tk in toks:
            if tk['t'] == 'frac' and merged and merged[-1]['t'] == 'num':
                merged[-1]['frac'] = merged[-1]['frac'] + tk['v']
            elif tk['t'] == 'frac':        # fraction-only quantity: integer part 0
                merged.append({'t': 'num', 'v': 0, 'frac': tk['v']})
            else:
                merged.append(tk)
        words = [tk['s'] for tk in merged if tk['t'] == 'word']
        corpus.append({
            'id': k, 'site': v.get('site'), 'support': v.get('support'),
            'scribe': v.get('scribe'), 'context': v.get('context'),
            'tokens': merged,
            'words': ['-'.join(w) for w in words],
            'logograms': [tk['v'] for tk in merged if tk['t'] == 'logo'],
            'numbers': [[tk['v'], tk['frac']] for tk in merged if tk['t'] == 'num'],
        })
    json.dump(corpus, open(os.path.join(DATA, 'corpus.json'), 'w'), ensure_ascii=False, indent=0)
    from collections import Counter
    print('inscriptions', len(corpus))
    print('with >=1 word', sum(1 for c in corpus if c['words']))
    wt = Counter(w for c in corpus for w in c['words'])
    print('word tokens', sum(wt.values()), 'word types', len(wt))
    print('multi-sign word tokens', sum(n for w, n in wt.items() if '-' in w),
          'types', sum(1 for w in wt if '-' in w))
    print('number tokens', sum(len(c['numbers']) for c in corpus),
          'with fractions', sum(1 for c in corpus for n in c['numbers'] if n[1]))
    print('logogram tokens', sum(len(c['logograms']) for c in corpus))
    print('top logograms', Counter(l for c in corpus for l in c['logograms']).most_common(20))
    print('fraction letters', Counter(f for c in corpus for n in c['numbers'] for f in n[1]).most_common())
    print('support', Counter(c['support'] for c in corpus).most_common(6))
    print('top words', wt.most_common(25))

if __name__ == '__main__':
    main()
