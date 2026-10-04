"""v42 cycle 2c: kill test for the ASEMIC result -- is the CS-Voynich bond just a stroke-level transliteration?

Meaningful long texts and mechanical generators are rewritten in a Gothic-minim stroke scheme (m -> iii, n/u/v -> ii,
a -> ci, d -> cl, o -> cd, r -> ir, ...: the kind of unit inventory EVA and Ponzi's CS alphabet use), with and without
the 18% OCR channel. If the ASEMIC class (CS volumes 1 + 2) absorbs these, the cycle-2 result is a transliteration
artefact (kill). Same LONG/SHORT battery, line metrics dropped.
"""
import os, sys, json, random, zlib
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v42_battery as B, v42_cycle2 as C2

FN = 'v42_cycle2.txt'
MIN = {'a': 'ci', 'b': 'ld', 'c': 'c', 'd': 'cl', 'e': 'e', 'f': 'lt', 'g': 'cj', 'h': 'li', 'i': 'i', 'j': 'ij', 'k': 'lx',
       'l': 'l', 'm': 'iii', 'n': 'ii', 'o': 'cd', 'p': 'jd', 'q': 'cjt', 'r': 'ir', 's': 's', 't': 'ct', 'u': 'ii', 'v': 'ii',
       'w': 'iiii', 'x': 'x', 'y': 'iij', 'z': 'z'}
LINE = {'linit', 'lfin'}


def minim(w):
    return ''.join(MIN.get(c, c) for c in w)


def main():
    C = L.all_corpora()
    objs = {}
    for k in ('L_Isidore', 'L_Latin_Lite', 'L_Italian_Lite', 'L_msG_Bav2', 'L_English_Lite', 'G_slot1', 'G_selfcit1', 'G_grille1', 'G_tri_la'):
        d = L.map_docs(C[k][1], minim)
        objs[f'M_{k}'] = d
        objs[f'M_{k}:n18'] = L.noise_docs(d, 0.18, seed=zlib.crc32(k.encode()) & 0xffff)
    rows = []
    for name, d in objs.items():
        for sc, n in B.SCALES.items():
            for i, b in enumerate(B.blocks(d, n, 2)):
                if sum(len(l) for l in b) < n: continue
                rows.append({'corpus': name, 'cls': 'TEST', 'var': 'x', 'scale': sc, 'i': i,
                             'F': B.battery(b, random.Random(zlib.crc32(f'{name}{sc}{i}'.encode())), sc)})
        print(name, flush=True)
    L.save('cycle2c_battery.json', rows)
    R = L.load('battery.json')
    res = {}
    for scale in ('LONG', 'SHORT'):
        for world in ('n18', 'clean'):
            def sel(r):
                if r['scale'] != scale: return False
                if r['corpus'].startswith('S_'): return r['var'] == 'clean'
                return r['var'] == world
            Rs = [r for r in R if sel(r)]
            Ts = [r for r in rows if r['scale'] == scale and (r['corpus'].endswith(':n18') == (world == 'n18'))]
            keys = sorted({k for r in Rs for k in r['F']} - LINE - ({'gap'} if scale == 'SHORT' else set()))
            X = C2.mat(Rs, keys); XT = C2.mat(Ts, keys)
            corp = np.array([r['corpus'] for r in Rs]); cls = np.array([r['cls'] for r in Rs])
            tcorp = np.array([r['corpus'] for r in Ts])
            isT = np.array([str(c).startswith(('S_', 'V_', 'T_')) for c in corp])
            train_cls = ['LANG', 'INVENT', 'CIPH', 'GEN'] + (['GIBB'] if scale == 'SHORT' else [])
            tr = np.isin(cls, train_cls) & ~isT
            m1 = corp == 'S_CS1'; m2 = corp == 'S_CS2'
            P, cl = C2.fit(np.vstack([X[tr], X[m1], X[m2]]), np.r_[cls[tr], ['ASEMIC'] * (m1.sum() + m2.sum())])
            post = {}
            for t in sorted(set(tcorp)):
                post[t] = dict(zip(cl, P(XT[tcorp == t]).mean(0).round(2)))
            post['V_ZL'] = dict(zip(cl, P(X[corp == 'V_ZL']).mean(0).round(2)))
            res[f'{scale}|{world}'] = post
            print(scale, world, {t: (max(p, key=p.get), p.get('ASEMIC')) for t, p in post.items()}, flush=True)
    L.save('cycle2c.json', res)
    for k, post in res.items():
        a = {t: p.get('ASEMIC', 0) for t, p in post.items() if t != 'V_ZL'}
        L.row(FN, f'V-42.2.{19 + list(res).index(k)}', f'{k}: KILL TEST -- 5 meaningful texts (Isidore, Latin, Italian, Bavarian MS German, English) and 4 generators (slot, self-citation, grille, Latin trigram) rewritten in Gothic-minim strokes (m=iii, n/u=ii, a=ci, o=cd ...){" + 18% OCR channel" if "n18" in k else ""}; ASEMIC class (CS vol 1+2) must NOT absorb them',
              f"ASEMIC posterior of minim texts: " + ', '.join(f"{t.replace('M_', '').replace(':n18', '')} {v:.2f}" for t, v in a.items()) + f"; absorbed (ASEMIC top) {sum(max(post[t], key=post[t].get) == 'ASEMIC' for t in a)}/{len(a)}; Voynich ASEMIC {post['V_ZL'].get('ASEMIC', 0):.2f}",
              'kill test passed: the bond is not a stroke-transliteration artefact' if sum(max(post[t], key=post[t].get) == 'ASEMIC' for t in a) <= 1 else 'KILL: stroke transliteration alone puts texts into ASEMIC')


if __name__ == '__main__':
    main()
