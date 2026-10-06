#!/usr/bin/env python3
"""la71 surface-wear measure per sign from SigLA's drawings (CC BY-NC-SA; drawings are fetched to a
temporary file, measured and deleted; only numbers are kept).

In the SigLA copies (as in GORILA) worn or damaged surface is drawn as stipple: small round dots.
lineara.xyz carries no per-sign damage, so this is the only per-sign 'damaged but read' signal.
For each SigLA sign box (svg rect, pixel coordinates = image) we count stipple dots inside the box
(dot centres are stored too; dot = connected ink component with bbox <= DOTMAX px on both sides and fill >= 0.4) and the share
of the box area that lies within R px of a dot ('cover').  Output la71_ckpt/wear.json:
  {doc: {'size':[w,h], 'boxes':[[occ_n, role, x, y, w, h, ndots, cover]], 'doc_cover': float}}
Workers: 2.
"""
import os, re, sys, json, glob, subprocess, tempfile
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la71_ckpt')
SIG = os.path.join(DATA, 'la22_ckpt', 'sigla')
DOTMAX, R = 16, 12
RECT = re.compile(r'<a href="[^"]*/index-(\d+)\.html"><rect[^>]*class="sign ([a-z]+)[^"]*"[^>]*x="([\d.]+)" '
                  r'y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"')


def measure(png, boxes):
    a = np.array(Image.open(png).convert('RGBA'))
    ink = (a[..., 3] > 128) & (a[..., :3].mean(-1) < 128)
    lab, n = ndimage.label(ink)
    objs = ndimage.find_objects(lab)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    dot = np.zeros(ink.shape, bool); cy = []; cx = []
    for i, sl in enumerate(objs, 1):
        if sl is None: continue
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        if h <= DOTMAX and w <= DOTMAX and sizes[i] >= 0.4 * h * w:
            dot[sl] |= (lab[sl] == i)
            cy.append((sl[0].start + sl[0].stop) / 2); cx.append((sl[1].start + sl[1].stop) / 2)
    near = ndimage.binary_dilation(dot, iterations=R) if dot.any() else dot
    cy = np.array(cy); cx = np.array(cx)
    out = []
    for occ, role, x, y, w, h in boxes:
        x0, y0 = int(float(x)), int(float(y)); x1, y1 = x0 + int(float(w)), y0 + int(float(h))
        nd = int(((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)).sum()) if len(cx) else 0
        sub = near[max(y0, 0):y1, max(x0, 0):x1]
        cov = float(sub.mean()) if sub.size else 0.0
        out.append([int(occ), role, x0, y0, x1 - x0, y1 - y0, nd, round(cov, 4)])
    doc_cov = float(near[ink.any(1)].mean()) if ink.any() else 0.0
    dots = [[int(x), int(y)] for x, y in zip(cx, cy)]
    return list(ink.shape[::-1]), out, round(doc_cov, 4), dots


def one(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'<div class="title">([^<]*)</div>', s)
    if not m: return None
    import html
    name = html.unescape(m.group(1))
    boxes = RECT.findall(s)
    if not boxes: return None
    u = 'https://sigla.phis.me/document/%s/%s.png' % (quote(name), quote(name))
    fd, tmp = tempfile.mkstemp(suffix='.png', dir=os.path.join(CK, 'tmp')); os.close(fd)
    try:
        r = subprocess.run(['curl', '-sS', '--retry', '3', '-o', tmp, '-w', '%{http_code}', u], capture_output=True, text=True)
        if r.stdout != '200': return name, {'err': r.stdout}
        size, out, dc, dots = measure(tmp, boxes)
        return name, {'size': size, 'boxes': out, 'doc_cover': dc, 'dots': dots}
    finally:
        os.remove(tmp)


def main():
    os.makedirs(os.path.join(CK, 'tmp'), exist_ok=True)
    outp = os.path.join(CK, 'wear.json')
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    paths = sorted(glob.glob(os.path.join(SIG, '*.html')))
    def name_of(p):
        m = re.search(r'<div class="title">([^<]*)</div>', open(p, encoding='utf-8', errors='replace').read())
        import html
        return html.unescape(m.group(1)) if m else None
    todo = [p for p in paths if not (name_of(p) in res and 'dots' in res[name_of(p)])]
    done = 0
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(one, todo):
            if r is None: continue
            name, v = r
            if name in res and 'err' not in res[name]: continue
            res[name] = v; done += 1
            if done % 50 == 0:
                json.dump(res, open(outp, 'w')); print(done, flush=True)
    json.dump(res, open(outp, 'w'))
    ok = [v for v in res.values() if 'boxes' in v]
    print('docs', len(res), 'measured', len(ok), 'boxes', sum(len(v['boxes']) for v in ok),
          'errors', sum(1 for v in res.values() if 'err' in v))


if __name__ == '__main__':
    main()
