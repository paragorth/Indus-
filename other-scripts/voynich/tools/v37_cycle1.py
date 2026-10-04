"""v37 cycle 1: key -> value coupling. Does the line's first word (or first glyph) predict words two or three
places later better than a mid-line word or the last word does, beyond paragraph-level coherence (swap null)?
Usage: python3 v37_cycle1.py NAME [NAME...]   (results to v37_ckpt/c1_NAME.json)"""
import sys, time
from v37_lib import *

def get(name, seed=0):
    if name in ('V', 'VI'): return voy('ZL3b' if name == 'V' else 'IT2a')
    if name == 'Vshuf': return shuffle_within(voy(), random.Random(seed))
    if name == 'Vgen': return forge_v26(voy(), seed=seed + 1)
    if name.startswith('Vplant'):
        return plant_schema(voy(), random.Random(seed), rho=float(name[6:]))
    if name.startswith('VA') or name.startswith('VB'):
        lang = name[1]
        return [p for p in voy() if p['sec'].endswith(lang)]
    return control_corpora()[name]

for name in sys.argv[1:]:
    t = time.time(); out = {}
    for seed in (0, 1):
        C = get(name, seed)
        body = name.startswith('V')
        out[f'joint_s{seed}'] = coupling(C, seed=seed, body_only=body)
        out[f'glyph_s{seed}'] = coupling(C, seed=seed, body_only=body, anchor='glyph')
    save(f'c1_{name}.json', out)
    j = out['joint_s0']; g = out['glyph_s0']
    print(name, f"{time.time()-t:.0f}s n={j['n_lines']}", 'joint KEY %.1f MID %.1f END %.1f KEYNESS %.1f+-%.1f ENDNESS %.1f+-%.1f' %
          (j['KEY'], j['MID'], j['END'], j['KEYNESS'], j['se']['KEYNESS'], j['ENDNESS'], j['se']['ENDNESS']),
          '| glyph KEYNESS %.1f+-%.1f' % (g['KEYNESS'], g['se']['KEYNESS']), flush=True)
