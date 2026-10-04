"""v37 cycle 1: key -> value coupling. Does the line's first word (or first glyph) predict interior words at
distance >= 2 better than a mid-line word or the last word does, beyond paragraph-level coherence (swap null)?
Also lexical echo: is the first word re-stated (same 3-unit prefix) later in its own line more than a mid word?
Usage: python3 v37_cycle1.py NAME [NAME...]   (results to v37_ckpt/c1_NAME.json)"""
import sys, time
from v37_lib import *

def get(name, seed=0):
    if name in ('V', 'VI'): return voy('ZL3b' if name == 'V' else 'IT2a')
    if name == 'Vshuf': return shuffle_within(voy(), random.Random(seed))
    if name == 'Vgen': return forge_v26(voy(), seed=seed + 1)
    if name.startswith('Vplant'):
        return plant_schema(voy(), random.Random(seed), rho=float(name[6:]))
    if name in ('VA', 'VB'):
        return [p for p in voy() if p['sec'].endswith(name[1])]
    return control_corpora()[name]

if __name__ == '__main__':
  for name in sys.argv[1:]:
    t = time.time(); out = {}
    for seed in (0, 1, 2):
        C = get(name, seed)
        body = name.startswith('V')
        out[f'joint_s{seed}'] = coupling(C, seed=seed, body_only=body, nboot=100)
        out[f'glyph_s{seed}'] = coupling(C, seed=seed, body_only=body, anchor='glyph', nboot=100)
        out[f'echo_s{seed}'] = echo(C, seed=seed, body_only=body, nboot=100)
    save(f'c1_{name}.json', out)
    def m(kind, key): return np.mean([out[f'{kind}_s{s}'][key] for s in range(3)])
    def se(kind, key): return np.mean([out[f'{kind}_s{s}']['se'][key] for s in range(3)])
    print(name, f"{time.time()-t:.0f}s", ' | '.join(f"{kind} KEY {m(kind,'KEY'):.1f} MID {m(kind,'MID'):.1f} END {m(kind,'END'):.1f} KEYNESS {m(kind,'KEYNESS'):.1f}+-{se(kind,'KEYNESS'):.1f}"
          for kind in ('joint', 'glyph', 'echo')), flush=True)
