#!/usr/bin/env python3
"""Parse an IVTFF 2.0 EVA transliteration (e.g. voynich.nu ZL3b-n.txt) into JSON lines.

Output: data/derived/<name>_lines.json, a list of records:
  folio, quire, panel, illus (H herbal, A astro, Z zodiac, B bio, C cosmo,
  P pharma, S stars/recipes, T text-only), lang (Currier A/B or None), hand,
  n (line number on page), ltype (P paragraph, L label, C circle, R radial),
  ltcode (full locus code), para_start, para_end, words (list of EVA words),
  uncertain (list of bools: word contains ? or rare-glyph code).

Conventions (stated in FINDINGS.md):
  '.' = word space, ',' = uncertain space -> treated as space (default),
  '<->' drawing interruption -> word break, [a:b] alternative -> first reading,
  {..} ligature braces stripped, '@nnn;' rare glyph -> '?' (word flagged).
"""
import json, re, sys, os

def clean(text):
    para_start = '<%>' in text
    para_end = '<$>' in text
    text = text.replace('<->', '.')
    text = re.sub(r'<![^>]*>', '', text)          # inline comments
    text = re.sub(r'<[^>]*>', '', text)            # other markers
    text = re.sub(r'\[([^:\]]*)(:[^\]]*)?\]', r'\1', text)  # alternatives
    text = text.replace('{', '').replace('}', '').replace("'", '')
    text = re.sub(r'@\d+;', '?', text)
    text = text.replace(',', '.')
    words = [w for w in text.split('.') if w]
    return words, para_start, para_end

def parse(path):
    recs = []
    page = {}
    for raw in open(path, encoding='utf-8', errors='replace'):
        raw = raw.rstrip('\n')
        if not raw or raw.startswith('#'):
            continue
        m = re.match(r'^<(f\w+)>\s*<!(.*)>', raw)
        if m:
            meta = dict(re.findall(r'\$(\w)=(\S)', m.group(2)))
            page = {'folio': m.group(1), 'quire': meta.get('Q'), 'panel': meta.get('P'),
                    'illus': meta.get('I'), 'lang': meta.get('L'), 'hand': meta.get('H')}
            continue
        m = re.match(r'^<(f\w+)\.(\d+),([@+*=&~])(\w)(\w*)>\s*(.*)$', raw)
        if not m:
            continue
        folio, n, pre, lt, sub, text = m.groups()
        words, ps, pe = clean(text)
        if not words:
            continue
        r = dict(page)
        r.update({'folio': folio, 'n': int(n), 'ltype': lt, 'ltcode': pre + lt + sub,
                  'para_start': ps, 'para_end': pe, 'words': words,
                  'uncertain': [('?' in w) for w in words]})
        recs.append(r)
    return recs

if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'data/ZL3b-n.txt'
    recs = parse(src)
    os.makedirs(os.path.join(os.path.dirname(src), 'derived'), exist_ok=True)
    name = os.path.basename(src).split('-')[0]
    out = os.path.join(os.path.dirname(src), 'derived', name + '_lines.json')
    json.dump(recs, open(out, 'w'))
    from collections import Counter
    nw = sum(len(r['words']) for r in recs)
    print(src, '->', out)
    print('lines', len(recs), 'words', nw, 'pages', len({r["folio"] for r in recs}))
    for key in ('ltype', 'illus', 'lang', 'hand'):
        c = Counter()
        for r in recs:
            c[r[key]] += len(r['words'])
        print(key, dict(c.most_common()))
    print('para starts', sum(r['para_start'] for r in recs), 'para ends', sum(r['para_end'] for r in recs))
    print('uncertain words', sum(sum(r['uncertain']) for r in recs))
