#!/usr/bin/env python3
"""LA-59 cycle-1 report: rows for loops/la59_cycle1.txt from data/la59_ckpt/c1_*.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *

rng = np.random.default_rng(5901)

def zp(x, null):
    null = np.asarray(null, float)
    return (x - null.mean()) / (null.std() + 1e-12), (1 + (null >= x).sum()) / (len(null) + 1)

LBFEAT = dict(p=('lab', 'stop', 0), t=('cor', 'stop', 0), k=('dor', 'stop', 0), q=('dor', 'stop', 0),
              d=('cor', 'stop', 1), m=('lab', 'nas', 1), n=('cor', 'nas', 1), s=('cor', 'fric', 0))

def truth_cv(sign, script):
    if script == 'LB':
        cv = lb_cv(sign)
        return cv if cv else (None, None)
    # LA outside check only: conventional transliteration labels (never a fit input)
    m = re.match(r'^([A-Z]*?)([AEIOU])\d?$', sign)
    if not m:
        return (None, None)
    return (m.group(1).lower(), m.group(2).lower())

def truth_eval(signs, s, script, nperm=2000):
    tc = [truth_cv(x, script)[0] for x in signs]; tv = [truth_cv(x, script)[1] for x in signs]
    tv = [v if v in VOWELS else None for v in tv]
    def stats(st):
        fc = [C_LABELS[x // 5] for x in st]; fv = [VOWELS[x % 5] for x in st]
        r = dict(co_c=co_assign_agreement(st, tc, lambda x: x // 5), co_v=co_assign_agreement(st, tv, lambda x: x % 5))
        iv = [i for i in range(len(st)) if tv[i]]
        r['acc_v'] = float(np.mean([fv[i] == tv[i] for i in iv]))
        ifc = [i for i in range(len(st)) if tc[i] in LBFEAT and fc[i] in PLACE]
        for k, F in (('place', PLACE), ('manner', MANNER), ('voice', VOICE)):
            j = ('place', 'manner', 'voice').index(k)
            r['acc_' + k] = float(np.mean([F[fc[i]] == LBFEAT[tc[i]][j] for i in ifc])) if ifc else np.nan
        r['n_feat'] = len(ifc); r['n_v'] = len(iv)
        return r
    obs = stats(np.asarray(s))
    nul = [stats(rng.permutation(s)) for _ in range(nperm)]
    out = {}
    for k in obs:
        if k.startswith('n_'):
            out[k] = obs[k]; continue
        v = np.array([n[k] for n in nul], float)
        out[k] = (obs[k], float(np.nanmean(v)), float((1 + (v >= obs[k]).sum()) / (len(v) + 1)))
    return out

def sharp(signs, real):
    """per-sign agreement of consonant and vowel over the top restarts (modal share)."""
    S = np.array([r['s'] for r in real[:4]])
    cs = []; vs = []
    for i in range(S.shape[1]):
        c = collections.Counter(C_LABELS[x // 5] for x in S[:, i]); v = collections.Counter(VOWELS[x % 5] for x in S[:, i])
        cs.append(c.most_common(1)[0][1] / len(S)); vs.append(v.most_common(1)[0][1] / len(S))
    # co-assignment stability between best and second-best restarts
    a, b = S[0], S[1]
    stab_c = co_assign_agreement(a, [x // 5 for x in b], lambda x: x // 5)
    stab_v = co_assign_agreement(a, [x % 5 for x in b], lambda x: x % 5)
    return float(np.mean(cs)), float(np.mean(vs)), stab_c, stab_v

def main():
    rows = []
    for tag in ('LA', 'LBs', 'LB'):
        p = os.path.join(CK, f'c1_{tag}.json')
        if not os.path.exists(p):
            continue
        o = json.load(open(p)); signs = o['signs']; real = o['real']
        import itertools as it
        gs8 = [r['g'] for r in real]; g = float(np.mean([max(c) for c in it.combinations(gs8, 4)]))   # expected best-of-4, matched to the nulls' 4 restarts
        zk, pk = zp(g, o['null_key']); zr, pr = zp(g, o['null_rewire'])
        zc, pc = zp(g, o['null_keyC']); zv, pv = zp(g, o['null_keyV'])
        mc, mv, stc, stv = sharp(signs, real)
        rows.append(f"| LA-59.1{tag} | {tag}: {len(signs)} signs, graph weight {np.sum(o['A'])/2:.0f} (1,000 random extraction settings). Fit 18 C x 5 V states, 8 restarts. Nulls: {len(o['null_key'])} shuffled keys (C and V), {len(o['null_keyC'])} C-only, {len(o['null_keyV'])} V-only, {len(o['null_rewire'])} rewired graphs; 200,000 random assignments. | gain (expected best of 4 restarts, as the nulls) {g:.4f} nats/edge, best of 8 {real[0]['g']:.4f} (lam {real[0]['lam']}); shuffled key {np.mean(o['null_key']):.4f} (z {zk:.2f}, P {pk:.3f}); C-shuffled {np.mean(o['null_keyC']):.4f} (z {zc:.2f}, P {pc:.3f}); V-shuffled {np.mean(o['null_keyV']):.4f} (z {zv:.2f}, P {pv:.3f}); rewired {np.mean(o['null_rewire']):.4f} (z {zr:.2f}, P {pr:.3f}); random max {o['random']['max']:.4f}. Sharpness: modal C share over top-4 restarts {mc:.2f}, V {mv:.2f}; best-vs-second co-assignment C {stc:.2f}, V {stv:.2f}. | see cycle verdict |")
        for pl in o['planted']:
            pass
        P = o['planted']
        for sh in (1.0, 0.5, 0.25):
            q = [x for x in P if x['share'] == sh]
            rows.append(f"| LA-59.1{tag}-plant{sh} | planted kernel graph on {tag}'s degrees/weight, sound share {sh}, 3 draws (chance: C 1/18, V 1/5) | fit gain {np.mean([x['g'] for x in q]):.4f} vs true-state gain {np.mean([x['g_true'] for x in q]):.4f}; label accuracy C {np.mean([x['acc_c'] for x in q]):.2f}, V {np.mean([x['acc_v'] for x in q]):.2f}; co-assignment C {np.mean([x['co_c'] for x in q]):.2f}, V {np.mean([x['co_v'] for x in q]):.2f} | - |")
        script = 'LB' if tag.startswith('LB') else 'LA'
        te = truth_eval(signs, real[0]['s'], script)
        lab = 'LB truth (values hidden in fit)' if script == 'LB' else 'OUTSIDE CHECK ONLY: conventional LA transliteration labels'
        rows.append(f"| LA-59.1{tag}-truth | {lab}; 2,000 permutations of fitted states over signs | " +
                    '; '.join(f"{k} {v[0]:.3f} vs {v[1]:.3f} (P {v[2]:.3f})" for k, v in te.items() if not k.startswith('n_')) +
                    f" (n feature {te['n_feat']}, n vowel {te['n_v']}) | - |")
        if tag == 'LA':
            s = real[0]['s']
            grp = collections.defaultdict(list)
            for x, st in zip(signs, s):
                grp[cv(st)].append(x)
            rows.append('| LA-59.1LA-map | best LA assignment (state: signs) | ' + '; '.join(f"{c}{v}: {' '.join(xs)}" for (c, v), xs in sorted(grp.items())) + ' | - |')
    open(os.path.join(LOOPS, 'la59_cycle1.txt'), 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
