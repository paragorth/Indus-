#!/usr/bin/env python3
"""Extract every label locus from the ZL3b IVTFF file with page, section, object class,
clock position, ring (from the transliterator's comments) and pharma jar-group.

Output: data/derived/v2_labels.json  (list of dicts), plus data/derived/v2_text_words.json
(paragraph/circle/radial words per page, for the label-in-text test).

Object class from the locus subtype (IVTFF 2.0):
  Lc container (pharma jar), Lf plant fragment (pharma), Lp plant (herbal),
  Ls star, Lz zodiac figure, Ln nymph (bio), Lt tube/pipe (bio), La/Lx other,
  L0 untyped (class then taken from section: C -> cosmo, A -> astro, T/H/B -> other).
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from parse_ivtff import clean

SUB = {'c': 'jar', 'f': 'frag', 'p': 'plant', 's': 'star', 'z': 'zodiac', 'n': 'nymph',
       't': 'tube', 'a': 'other', 'x': 'other'}


def main(src=os.path.join(ROOT, 'data', 'ZL3b-n.txt')):
    labels, text = [], {}
    page, ring, group, order = {}, None, -1, 0
    for raw in open(src, encoding='utf-8', errors='replace'):
        raw = raw.rstrip('\n')
        if raw.startswith('#'):
            c = raw.lower()
            for key in ('outer', 'middle', 'second', 'inner', 'top', 'central', 'centre'):
                if key in c and ('ring' in c or 'band' in c or 'label' in c or 'row' in c
                                 or 'star' in c or 'circle' in c or 'nymph' in c):
                    ring = {'second': 'middle', 'centre': 'central'}.get(key, key)
                    break
            continue
        m = re.match(r'^<(f\w+)>\s*<!(.*)>', raw)
        if m:
            meta = dict(re.findall(r'\$(\w)=(\S)', m.group(2)))
            page = {'folio': m.group(1), 'illus': meta.get('I'), 'lang': meta.get('L'),
                    'hand': meta.get('H'), 'quire': meta.get('Q')}
            ring, group, order = None, -1, 0
            continue
        m = re.match(r'^<(f\w+)\.(\d+),([@+*=&~])(\w)(\w*)>\s*(.*)$', raw)
        if not m:
            continue
        folio, n, pre, lt, sub, body = m.groups()
        clock = None
        cm = re.match(r'^<!(\d\d):(\d\d)>', body)
        if cm:
            clock = (int(cm.group(1)) % 12) + int(cm.group(2)) / 60.0
        words, _, _ = clean(body)
        if not words:
            continue
        if lt != 'L':
            text.setdefault(folio, []).extend(words)
            continue
        cls = SUB.get(sub)
        if cls is None:
            cls = {'C': 'cosmo', 'A': 'astro', 'Z': 'zodiac'}.get(page.get('illus'), 'other')
        if cls == 'jar':
            group += 1
        lab = dict(page)
        lab.update({'folio': folio, 'n': int(n), 'ltcode': pre + lt + sub, 'cls': cls,
                    'words': words, 'label': '.'.join(words), 'clock': clock,
                    'ring': ring if page.get('illus') in ('Z', 'A', 'C') else None,
                    'group': group if cls in ('jar', 'frag') else None,
                    'order': order,
                    'uncertain': any('?' in w for w in words)})
        order += 1
        labels.append(lab)
    out = os.path.join(ROOT, 'data', 'derived')
    json.dump(labels, open(os.path.join(out, 'v2_labels.json'), 'w'), indent=0)
    json.dump(text, open(os.path.join(out, 'v2_text_words.json'), 'w'))
    from collections import Counter
    print('labels', len(labels), Counter(l['cls'] for l in labels))
    print('label words', sum(len(l['words']) for l in labels))
    print('rings', Counter((l['cls'], l['ring']) for l in labels if l['ring']))
    print('with clock', sum(l['clock'] is not None for l in labels))
    print('text pages', len(text), 'text words', sum(map(len, text.values())))


if __name__ == '__main__':
    main()
