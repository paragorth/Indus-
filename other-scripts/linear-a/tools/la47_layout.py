#!/usr/bin/env python3
"""la47: build a 2-D layout record for every Linear A document.
Physical lines come from lineara.xyz 'transcription' (GORILA line layout; checked against SigLA box
y-coordinates), logical tokens from corpus.json.  Each corpus token is aligned to transcription characters
by character class (S sign, N numeral, F fraction, D divider, L lacuna/other) with difflib, giving the
physical line(s) each token occupies.  SigLA boxes (data/la34_ckpt/occ_index.json) add x/y/size for
signs where SigLA has the document.  Output: data/la47_ckpt/layout.json"""
import json, os, re, subprocess, difflib, collections, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); CK = os.path.join(D, 'la47_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def js_dump():
    out = os.path.join(CK, 'lineara_js.json')
    if os.path.exists(out): return json.load(open(out))
    code = r'''const fs=require("fs");let s=fs.readFileSync(process.argv[1],"utf8");
s=s.replace(/^[\s\S]*?var inscriptions = /,"var inscriptions = ");eval(s+";globalThis.I=inscriptions;");
let o={};for(const [k,v] of I){o[k]={p:v.parsedInscription,t:v.transcription,w:v.transliteratedWords,sup:v.support}}
fs.writeFileSync(process.argv[2],JSON.stringify(o));'''
    subprocess.run(['node', '-e', code, os.path.join(D, 'LinearAInscriptions.js'), out], check=True)
    return json.load(open(out))


def cls(ch):
    o = ord(ch)
    if o == 0x10101: return 'D'
    if 0x10107 <= o <= 0x1013F: return 'N'
    if 0x10740 <= o <= 0x10755: return 'F'
    if (0x10600 <= o <= 0x1073F) or (0x10760 <= o <= 0x10767) or (0x10000 <= o <= 0x100FF): return 'S'
    if ch.isspace(): return None
    return 'L'


def tok_cls(t):
    if t['t'] == 'word': return 'S' * len(t['s'])
    if t['t'] == 'logo': return 'S'
    if t['t'] == 'div': return 'D'
    if t['t'] == 'num':
        k = sum(1 for c in str(int(t['v'])) if c != '0') if t.get('v') else 0
        return 'N' * k + 'F' * len(t.get('frac', []))
    if t['t'] == 'unk': return 'L'
    return ''


def build():
    J = js_dump()
    C = json.load(open(os.path.join(D, 'corpus.json')))
    docs = {}; stats = collections.Counter()
    for d in C:
        j = J.get(d['id'])
        if not j: stats['nojs'] += 1; continue
        chars = []; line = 0
        for ch in j['t']:
            if ch == '\n': line += 1; continue
            c = cls(ch)
            if c: chars.append((c, line))
        nlines = line + 1
        toks = [t for t in d['tokens'] if t['t'] != 'nl']
        # logical line index of each token
        ll = 0; lline = []
        for t in d['tokens']:
            if t['t'] == 'nl': ll += 1
            else: lline.append(ll)
        tc = [tok_cls(t) for t in toks]
        exp = ''.join(tc); act = ''.join(c for c, _ in chars)
        owner = []
        for i, s in enumerate(tc): owner += [i] * len(s)
        sm = difflib.SequenceMatcher(None, exp, act, autojunk=False)
        tl = collections.defaultdict(list)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n): tl[owner[a + k]].append(chars[b + k][1])
        matched = sum(n for _, _, n in sm.get_matching_blocks())
        q = matched / max(1, len(exp))
        stats['exact' if exp == act else ('good' if q >= 0.9 else 'poor')] += 1
        rec = []
        for i, t in enumerate(toks):
            L = tl.get(i, [])
            r = {'t': t['t'], 'll': lline[i], 'pl': (min(L) if L else None), 'pl2': (max(L) if L else None)}
            if t['t'] == 'word': r['w'] = '-'.join(t['s'])
            elif t['t'] == 'logo': r['w'] = t['v']
            elif t['t'] == 'num': r['v'] = t['v']; r['frac'] = t.get('frac', [])
            rec.append(r)
        # fill missing physical lines by neighbours
        for i, r in enumerate(rec):
            if r['pl'] is None:
                prev = next((rec[k]['pl2'] for k in range(i - 1, -1, -1) if rec[k]['pl2'] is not None), 0)
                r['pl'] = r['pl2'] = prev; r['imputed'] = 1
        docs[d['id']] = {'site': d['site'], 'support': d['support'], 'scribe': d.get('scribe', ''), 'nlines': nlines,
                         'align_q': round(q, 3), 'toks': rec}
    # SigLA geometry: per document, the boxes in reading order, attached to word/logo tokens by sign count
    try:
        from la34_common import load
        occ, meta, cid = load()
        by = collections.defaultdict(list)
        for o in occ:
            if o['did'] and o['role'] not in ('erasure',): by[o['did']].append(o)
        for did, os_ in by.items():
            if did not in docs: continue
            os_.sort(key=lambda o: o['n'])
            rec = docs[did]; vb = os_[0]['vb']
            seq = []  # sign-bearing tokens in order (word signs and logograms)
            for i, r in enumerate(rec['toks']):
                if r['t'] == 'word': seq += [(i, 'syllabogram')] * len(r['w'].split('-'))
                elif r['t'] == 'logo': seq += [(i, 'logogram')]
            boxes = [o for o in os_ if o['role'] in ('syllabogram', 'logogram', 'transaction')]
            if len(boxes) != len(seq): stats['sigla_mismatch'] += 1; continue
            stats['sigla_ok'] += 1
            rec['vb'] = vb
            g = collections.defaultdict(list)
            for (i, _), o in zip(seq, boxes): g[i].append(o['rect'])
            for i, rs in g.items(): rec['toks'][i]['box'] = rs
            rec['frac_boxes'] = [o['rect'] for o in os_ if o['role'] == 'fraction']
    except Exception as e:
        print('sigla skipped', e)
    json.dump(docs, open(os.path.join(CK, 'layout.json'), 'w'))
    print(dict(stats), 'docs', len(docs))


if __name__ == '__main__':
    build()
