"""v57 cycle 4b: is the 'load starts before q-, after -y' effect pen time or junction shape?
(a) Darkening/lightening odds at the joint junction (left ends -y AND right starts q-;
    Latin: left ends e/u AND right starts p/d/f) in real lines vs word-shuffled lines
    (shuffled lines also contain such junctions, with the same glyph shapes).
(b) Word-level sawtooth skew with all y|q junction increments removed, and computed only
    on them; Latin analogue."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v57_lib import Corpus, CK, vglyphs, lglyphs
from v57_c2 import extract
from v57_c1 import page_lines, skew
from v57_c3 import gap_events
from v57_c4 import shuffle_row, type_resid


def joint(which, gl):
    if which == 'V':
        return lambda l, r: gl(l)[-1] == 'y' and gl(r)[0] == 'q'
    return lambda l, r: gl(l)[-1] in 'eu' and gl(r)[0] in 'pdf'


def odds(ev, f):
    c = {(a, s): 0 for a in (True, False) for s in (1, -1)}
    for e in ev:
        c[(f(e['left'], e['right']), e['sg'])] += 1
    return c, (c[(True, 1)] / max(1, c[(True, -1)])) / (c[(False, 1)] / max(1, c[(False, -1)]))


if __name__ == '__main__':
    rng = np.random.default_rng(5714)
    out = {}
    for which in ['L', 'V']:
        gl = vglyphs if which == 'V' else lglyphs
        f = joint(which, gl)
        rows = extract(which)
        res = {}
        for wg, q, tolg in [(3, 0.9, 1.0), (3, 0.97, 1.0), (1.5, 0.97, 1.0)]:
            c, o = odds(gap_events(rows, wg, q, tolg), f)
            sh = [odds(gap_events([shuffle_row(r, rng) for r in rows], wg, q, tolg), f)[1] for _ in range(12)]
            res[f'{wg}_{q}_{tolg}'] = dict(counts={str(k): v for k, v in c.items()}, odds_real=round(o, 3),
                                         odds_shuf=(round(float(np.mean(sh)), 3), round(float(np.std(sh)), 3)),
                                         z=round(float((o - np.mean(sh)) / (np.std(sh) + 1e-9)), 2))
            print(which, wg, q, tolg, json.dumps(res[f'{wg}_{q}_{tolg}']), flush=True)
        C = Corpus(which, 'top'); pl = page_lines(C); rt = type_resid(C)
        inc_j, inc_o = [], []
        for ls in pl.values():
            for L in ls:
                for a, b in zip(L[:-1], L[1:]):
                    (inc_j if f(C.word[a], C.word[b]) else inc_o).append(rt[b] - rt[a])
        inc_j, inc_o = np.array(inc_j), np.array(inc_o)
        # null for skew of a subset: within-line shuffles of rt, same subset definition by position
        def null_skews(n=200):
            s_o = []
            for _ in range(n):
                rr = rt.copy()
                for ls in pl.values():
                    for L in ls:
                        rr[L] = rt[rng.permutation(L)]
                v = [rr[b] - rr[a] for ls in pl.values() for L in ls for a, b in zip(L[:-1], L[1:]) if not f(C.word[a], C.word[b])]
                s_o.append(skew(np.array(v)))
            return np.array(s_o)
        ns = null_skews()
        res['skew_without_joint'] = dict(n=len(inc_o), skew=round(float(skew(inc_o)), 3), z=round(float((skew(inc_o) - ns.mean()) / ns.std()), 2))
        res['joint_increments'] = dict(n=len(inc_j), mean=round(float(inc_j.mean()), 3) if len(inc_j) else None,
                                       skew=round(float(skew(inc_j)), 3) if len(inc_j) > 5 else None)
        print(which, json.dumps({k: res[k] for k in ['skew_without_joint', 'joint_increments']}), flush=True)
        out[which] = res
    json.dump(out, open(os.path.join(CK, 'c4b.json'), 'w'), indent=1)
