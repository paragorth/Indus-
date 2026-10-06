#!/usr/bin/env python3
"""la71 cycle 3: re-run the la22 SigLA comparison (la22_sigla_test.py, frozen predictions untouched) on the
fixed SigLA parse.  Writes a la22-format sigla.json built from la71_ckpt/sigla71.json into the 'all'
redirect tree, then runs la22_sigla_test.py through la71_run.py (outputs land in the redirect tree)."""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
LA = os.path.abspath(os.path.join(HERE, '..')); REPO = os.path.abspath(os.path.join(LA, '..', '..'))
CK = os.path.join(LA, 'data', 'la71_ckpt')
S = json.load(open(os.path.join(CK, 'sigla71.json')))
out = {}
for name, v in S.items():
    # comparable with la22: SigLA '[?]' (unidentified) occurrences left out as la22's parse did; variant
    # letters stripped (AB21f -> AB21); the only change is that subscripted readings are now kept
    import re
    occ = [dict(o, code=re.sub(r'[a-z]$', '', o['code'])) for o in v['occ'] if o['parsed']]
    words = []; cur = []; last = None
    for o in occ:
        if o['role'] == 'syllabogram':
            if cur and o['par'] != last: words.append(cur); cur = []
            cur.append(o); last = o['par']
        else:
            if cur: words.append(cur); cur = []
            last = None
    if cur: words.append(cur)
    out[name] = {'name': name, 'occ': occ, 'words': [[(o['code'], o['read'], o['sure']) for o in w] for w in words]}
dst = os.path.join(CK, 'redir', 'all', os.path.relpath(os.path.join(LA, 'data', 'la22_ckpt', 'sigla.json'), REPO))
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, 'w'), ensure_ascii=False)
print('wrote', dst, len(out), sum(len(v['occ']) for v in out.values()))
subprocess.run([sys.executable, os.path.join(HERE, 'la71_run.py'), 'all', os.path.join(HERE, 'la22_sigla_test.py')], check=True)
