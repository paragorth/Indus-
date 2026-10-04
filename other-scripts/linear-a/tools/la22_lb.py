#!/usr/bin/env python3
"""la22: Linear B (DAMOS) documents in the la22 token format (positive control).

Only spellings, logogram names, numbers, line breaks and the edition's break marks are used.
'[' or ']' at a token edge -> gap token. A sign with a dot below (doubtful) is kept and its
word is flagged 'dot' so it is never used as a known answer.
"""
import json, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la22_ckpt')
SYL = re.compile(r'^(\*\d+|[a-z]+\d?)$')


def strip_dots(s):
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if ch != '̣')


def parse_doc(content):
    toks = []
    for line in content.replace('\r', '').split('\n'):
        line = re.sub(r'^\s*\.[0-9a-zA-Z]+\s*', ' ', line)       # line number
        line = re.sub(r'⟦[^⟧]*⟧', ' ', line)                     # erasures
        line = re.sub(r"'[^']*'", ' ', line)                      # superscript / small text
        line = line.replace('⌞', '').replace('⌟', '')
        parts = re.split(r'[\s,/:;•]+', line)
        had = False
        for p in parts:
            if not p or p in ('vac.', 'vest.', 'mut.', 'v.↓', 'v.', 'lat.', 'inf.', 'sup.', 'deest', 'qs', 'vacat', 'vacant'):
                continue
            lead = p.startswith('['); trail = p.endswith(']') is False and p.endswith('[')
            gl = p.startswith(']') or p.startswith('['); gr = p.endswith('[') or p.endswith(']')
            core = p.strip('[]')
            if gl: toks.append({'t': 'gap'})
            if '[' in core or ']' in core:                         # break inside a word: split
                core = core.replace('[', ' ').replace(']', ' ')
                pieces = [x for x in core.split() if x]
            else:
                pieces = [core] if core else []
            for k, pc in enumerate(pieces):
                if k > 0: toks.append({'t': 'gap'})
                dot = '̣' in unicodedata.normalize('NFD', pc)
                q = strip_dots(pc).strip('-')
                if not q: continue
                if re.fullmatch(r'\d+', q):
                    toks.append({'t': 'N', 'v': int(q), 'frac': [], 'dot': dot})
                elif '-' in q or SYL.match(q):
                    ss = q.split('-')
                    if all(SYL.match(s) for s in ss):
                        toks.append({'t': 'w', 'c': ss, 'tr': ss, 'dot': dot})
                    else:
                        toks.append({'t': 'unk', 'v': q})
                elif re.fullmatch(r'[A-Z][A-Za-z0-9+*±]*', q):
                    toks.append({'t': 'L', 'c': q, 'tr': q, 'dot': dot})
                else:
                    toks.append({'t': 'unk', 'v': q})
            if gr: toks.append({'t': 'gap'})
            had = True
        if had: toks.append({'t': 'nl'})
    out = []
    for tk in toks:
        if tk['t'] == 'gap' and out and out[-1]['t'] == 'gap': continue
        out.append(tk)
    # merge numbers split by spaces after the same logogram/measure are left separate on purpose
    return out


def load():
    p = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(p):
        return json.load(open(p))
    D = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(l)
        if not x.get('content'): continue
        site = x['heading'].split()[0]
        D.append({'id': x['heading'].strip(), 'site': site, 'support': 'Tablet', 'scribe': '',
                  'toks': parse_doc(x['content'])})
    json.dump(D, open(p, 'w'), ensure_ascii=False)
    return D


if __name__ == '__main__':
    from collections import Counter
    D = load()
    print(len(D), Counter(t['t'] for d in D for t in d['toks']))
    sy = Counter(c for d in D for t in d['toks'] if t['t'] == 'w' for c in t['c'])
    print('sign types', len(sy), 'tokens', sum(sy.values()))
    print(D[1500]['id'], D[1500]['toks'][:20])
