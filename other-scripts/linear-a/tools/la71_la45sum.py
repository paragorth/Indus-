#!/usr/bin/env python3
"""la71 cycle 2: summarise the la45 self-play re-runs (6 populations, 40 rounds, tag v71, same seeds in
every version) for the two la45 B claims: the counted-commodity class (VIN OLE OLIV VIR GRA NI KI DI RE TI
MA) and the sealing heading class (*301 KA KU SI RO ZE).  A member 'holds' if its modal meaning across
populations is the class name with agreement >= 0.8 (la45's stability rule).  Reference: the original
la45 cycle-1 runs (tag c1)."""
import os, sys, json, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la45_common as C
LA = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(LA, '..', '..'))
COM = ['L:VIN', 'L:OLE', 'L:OLIV', 'L:VIR+[?]', 'L:GRA', 'NI', 'KI', 'DI', 'RE', 'TI', 'MA']
HDR = ['*301', 'KA', 'KU', 'SI', 'RO', 'ZE']


def summ(files):
    runs = [json.load(open(f)) for f in files]
    st = C.stability(runs, minc=3)
    m = st['modal']
    def holds(ws, name):
        return {w: (m[w][0], round(m[w][1], 2)) if w in m else None for w in ws}, \
               sum(1 for w in ws if w in m and m[w][0] == name and m[w][1] >= 0.8)
    com, nc = holds(COM, 'COMMODITY'); hdr, nh = holds(HDR, 'HEADER')
    return dict(npop=len(runs), ari=round(st['ari_mean'], 3), gain=st['final_gain'], com_hold=nc, com=com, hdr_hold=nh, hdr=hdr)


res = {}
ref = sorted(f for f in glob.glob(os.path.join(C.CK, 'c1_LA_*.json')) if re.fullmatch(r'c1_LA_\d+\.json', os.path.basename(f)))
res['original_c1'] = summ(ref)
for v in ('all', 'rd', 'read'):
    d = os.path.join(LA, 'data', 'la71_ckpt', 'redir', v, os.path.relpath(C.CK, REPO))
    fs = sorted(f for f in glob.glob(os.path.join(d, 'v71_LA_*.json')) if re.fullmatch(r'v71_LA_\d+\.json', os.path.basename(f)))
    if fs: res[v] = summ(fs)
for k, v in res.items(): print(k, json.dumps(v, ensure_ascii=False))
json.dump(res, open(os.path.join(LA, 'data', 'la71_ckpt', 'c2_la45.json'), 'w'), indent=1, ensure_ascii=False)
