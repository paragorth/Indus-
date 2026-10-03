#!/usr/bin/env python3
"""V4 step 1: extract the zodiac nymph labels (locus type Lz, f70v2-f73v) from
ZL3b-n.txt with page, ring (a new ring starts at each '@Lz' locus), order within
ring and page, and the clock position given in the <!hh:mm> comment.
Output: data/derived/v4_zodiac_labels.json
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'data', 'ZL3b-n.txt')
OUT = os.path.join(ROOT, 'data', 'derived', 'v4_zodiac_labels.json')
# page -> month block (half-month pages are merged in folio order)
MONTH = {'f70v2': 'Pisces', 'f70v1': 'Aries', 'f71r': 'Aries', 'f71v': 'Taurus', 'f72r1': 'Taurus',
         'f72r2': 'Gemini', 'f72r3': 'Cancer', 'f72v3': 'Leo', 'f72v2': 'Virgo', 'f72v1': 'Libra',
         'f73r': 'Scorpio', 'f73v': 'Sagittarius'}

def clean(t):
    t = re.sub(r'<![^>]*>', '', t); t = re.sub(r'<[^>]*>', '', t)
    t = re.sub(r'\[([^:\]]*)(:[^\]]*)?\]', r'\1', t)
    t = t.replace('{', '').replace('}', '').replace("'", '')
    t = re.sub(r'@\d+;', '?', t)
    return [w for w in t.replace(',', '.').split('.') if w]

def extract():
    out, ring, idx = [], {}, {}
    for raw in open(SRC, encoding='utf-8', errors='replace'):
        m = re.match(r'^<(f\w+)\.(\d+),([@&+=*~])Lz>\s*(.*)$', raw.rstrip('\n'))
        if not m: continue
        page, n, pre, text = m.groups()
        if page not in MONTH: continue
        if pre == '@': ring[page] = ring.get(page, 0) + 1
        clk = re.search(r'<!(\d\d):(\d\d)>', text)
        ang = None
        if clk:
            h, mi = int(clk.group(1)), int(clk.group(2)); ang = ((h % 12) * 60 + mi) / 720 * 360
        words = clean(text)
        if not words: continue
        idx[page] = idx.get(page, 0) + 1
        out.append({'page': page, 'month': MONTH[page], 'n': int(n), 'ring': ring.get(page, 1),
                    'page_idx': idx[page], 'angle': ang, 'words': words, 'label': ''.join(words),
                    'uncertain': '?' in ''.join(words)})
    return out

if __name__ == '__main__':
    labs = extract(); json.dump(labs, open(OUT, 'w'), indent=0)
    from collections import Counter
    print('labels', len(labs), 'pages', Counter(l['page'] for l in labs))
    print('rings', Counter((l['page'], l['ring']) for l in labs))
