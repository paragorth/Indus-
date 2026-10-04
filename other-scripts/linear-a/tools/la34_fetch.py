#!/usr/bin/env python3
"""la34: list every SigLA per-occurrence drawing (from the document pages cached by la22) with its
bounding rectangle on the document drawing, role, sign code and reading; then download the
occurrence PNGs, throttled (one request at a time, 0.35 s apart), into the scratchpad (never the repo).
Output: data/la34_ckpt/occ_index.json  (doc, n, role, code, read, sure, rect x,y,w,h, image url)."""
import os, re, json, glob, html, time, sys, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
PAGES = os.path.join(D, 'la22_ckpt', 'sigla')
CK = os.path.join(D, 'la34_ckpt')
IMG = os.environ.get('LA34_IMG', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/la34img')
RECT = re.compile(r'getElementById\(&quot;occ-(\d+)&quot;\)[^>]*class="sign ([a-z ]+?)"[^>]*x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"')
POP = re.compile(r'class="popup popup-(?:left|right) ([a-z ]+?)" id="occ-(\d+)">(.*?)<img src="\.\./\.\./([^"]+)" alt="Image of the occurrence"', re.S)
CODE = re.compile(r'reading-pattern:\(([A-Z0-9]+), (?:false|true)\)')
RD = re.compile(r'<span class="(sure|unsure)-reading">([^<]*)</span>')

def index():
    out = []
    for p in sorted(glob.glob(os.path.join(PAGES, '*.html'))):
        s = open(p, encoding='utf-8', errors='replace').read()
        m = re.search(r'<div class="title">([^<]*)</div>', s)
        doc = html.unescape(m.group(1)) if m else os.path.basename(p)[:-5]
        rects = {int(n): (cls, float(x), float(y), float(w), float(h)) for n, cls, x, y, w, h in RECT.findall(s)}
        m = re.search(r'viewBox="0 0 (\d+) (\d+)"', s)
        vb = (int(m.group(1)), int(m.group(2))) if m else None
        for cls, n, body, src in POP.findall(s):
            n = int(n); c = CODE.search(body); r = RD.search(body)
            rc = rects.get(n)
            out.append({'doc': doc, 'n': n, 'role': cls.split()[0], 'par': (cls.split() + [''])[1],
                        'code': c.group(1) if c else '', 'read': html.unescape(r.group(2)) if r else '',
                        'sure': bool(r and r.group(1) == 'sure'), 'rect': rc[1:] if rc else None,
                        'vb': vb, 'src': html.unescape(src)})
    return out

def fname(src):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', src.split('document/', 1)[-1])

def main():
    occ = index()
    os.makedirs(CK, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    json.dump(occ, open(os.path.join(CK, 'occ_index.json'), 'w'), ensure_ascii=False)
    print('occurrences', len(occ), 'docs', len({o['doc'] for o in occ}), flush=True)
    if '--index-only' in sys.argv: return
    todo = [o for o in occ if not os.path.exists(os.path.join(IMG, fname(o['src'])))]
    # syllabograms first (they carry the hand signal), then the rest
    todo.sort(key=lambda o: o['role'] != 'syllabogram')
    print('to fetch', len(todo), flush=True)
    for i, o in enumerate(todo):
        url = 'https://sigla.phis.me/' + urllib.parse.quote(o['src'])
        for attempt in range(3):
            try:
                b = urllib.request.urlopen(url, timeout=30).read()
                open(os.path.join(IMG, fname(o['src'])), 'wb').write(b); break
            except Exception as e:
                print('err', url, e, flush=True); time.sleep(3)
        time.sleep(0.35)
        if i % 250 == 0: print(i, flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
