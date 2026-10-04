"""v42 cycle 2: the full battery (v42_battery.py) -- Zipf, Heaps, TTR, hapax, word-length CV, entropy ratios,
junction strength, v23 glyph and word-length arrows, line-initial / line-final effects, slot dependence, v33 gap
ratio -- on languages, invented languages, ciphers (Copiale, Borg), generators, short gibberish, the Codex
Seraphinianus (raw and repeats collapsed) and the Voynich.

Placement: (a) per metric, which class range the Voynich and the CS fall in; 'shared departures' = metrics where
both leave the language range on the same side; (b) LOCO logistic over corpora on the battery (classes LANG,
INVENT, CIPH, GEN; + GIBB at the SHORT scale), Voynich and CS never trained on; (c) a 6th class ASEMIC = CS volume 1
and volume 2 as two corpora: held-out volume must be recognised (control), then the Voynich posterior ASEMIC vs GEN;
kill control: ASEMIC also absorbs random other test objects (Voynich forgeries excluded; Martian glossolalia).
All in the n18 world (every non-CS text through the OCR-like channel), and in the clean world for reference.
"""
import os, sys, json
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L
from sklearn.linear_model import LogisticRegression

FN = 'v42_cycle2.txt'


def load():
    R = L.load('battery.json')
    return R


def mat(R, keys):
    X = np.array([[r['F'].get(k, np.nan) for k in keys] for r in R], float)
    return X


def fit(X, y):
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
    Z = np.nan_to_num((X - mu) / sd)
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Z, y)
    return (lambda W: m.predict_proba(np.nan_to_num((W - mu) / sd))), list(m.classes_)


def main():
    R = load()
    out = {}; rows = []
    for scale in ('LONG', 'SHORT'):
        keys = sorted({k for r in R if r['scale'] == scale for k in r['F']})
        if scale == 'SHORT': keys = [k for k in keys if k != 'gap']
        for world in ('n18', 'clean'):
            def sel(r):
                if r['scale'] != scale: return False
                if r['corpus'].startswith('S_'): return r['var'] in ('clean', 'col')
                return r['var'] == world
            Rs = [r for r in R if sel(r)]
            grp = lambda r: (r['corpus'] + (':col' if r['var'] == 'col' else '')) if r['corpus'].startswith(('S_', 'V_')) else r['cls']
            X = mat(Rs, keys); G = np.array([grp(r) for r in Rs]); corp = np.array([r['corpus'] for r in Rs]); cls = np.array([r['cls'] for r in Rs])
            train_cls = ['LANG', 'INVENT', 'CIPH', 'GEN'] + (['GIBB'] if scale == 'SHORT' else [])
            # ---- per-metric table
            med = {}
            for g in sorted(set(G)):
                med[g] = {k: float(np.nanmedian(X[G == g][:, i])) for i, k in enumerate(keys)}
            lo = {k: np.nanpercentile(X[cls == 'LANG'][:, i], 5) for i, k in enumerate(keys)}
            hi = {k: np.nanpercentile(X[cls == 'LANG'][:, i], 95) for i, k in enumerate(keys)}
            def side(g, k):
                v = med[g][k]; return 0 if lo[k] <= v <= hi[k] else (1 if v > hi[k] else -1)
            vz = 'V_ZL'; cs = 'S_CS'; csc = 'S_CS:col'
            shared = [k for k in keys if side(vz, k) != 0 and side(vz, k) == side(cs, k)]
            shared_c = [k for k in keys if side(vz, k) != 0 and side(vz, k) == side(csc, k)]
            vdep = [k for k in keys if side(vz, k) != 0]
            # which class median is nearest per metric (in LANG sd units)
            sdk = {k: np.nanstd(X[:, i]) or 1 for i, k in enumerate(keys)}
            cl_list = train_cls + ['S_CS', 'S_CS:col']
            nearest = {}
            for k in keys:
                d = sorted((abs(med[vz][k] - med[c][k]) / sdk[k], c) for c in cl_list if c in med)
                nearest[k] = d[0][1]
            # ---- LOCO classifier over corpora
            tr = np.isin(cls, train_cls) & ~np.char.startswith(corp.astype(str), ('S_', 'V_'))
            per = defaultdict(list)
            for c in sorted(set(corp[tr])):
                te = tr & (corp == c); P, cl = fit(X[tr & ~te], cls[tr & ~te])
                per[cls[te][0]].append(cl[int(P(X[te]).mean(0).argmax())] == cls[te][0])
            ba = float(np.mean([np.mean(v) for v in per.values()])); pc = {k: f'{sum(v)}/{len(v)}' for k, v in per.items()}
            P, cl = fit(X[tr], cls[tr])
            post = {g: dict(zip(cl, P(X[G == g]).mean(0).round(2))) for g in ('V_ZL', 'V_IT', 'V_ZL_A', 'V_ZL_B', 'S_CS', 'S_CS:col', 'S_CS1', 'S_CS2', 'S_CS1:col', 'S_CS2:col') if (G == g).any()}
            # ---- ASEMIC 6th class: CS V1 and V2 as two corpora (raw); held-out-volume control
            asem = {}
            for cs_var in ('', ':col'):
                m1 = G == 'S_CS1' + cs_var; m2 = G == 'S_CS2' + cs_var
                Xa = np.vstack([X[tr], X[m1], X[m2]]); ya = np.r_[cls[tr], ['ASEMIC'] * (m1.sum() + m2.sum())]
                ctl = []
                for a, b in ((m1, m2), (m2, m1)):
                    Pq, clq = fit(np.vstack([X[tr], X[a]]), np.r_[cls[tr], ['ASEMIC'] * a.sum()])
                    ctl.append(clq[int(Pq(X[b]).mean(0).argmax())])
                Pa, cla = fit(Xa, ya)
                pv = {g: dict(zip(cla, Pa(X[G == g]).mean(0).round(2))) for g in ('V_ZL', 'V_IT', 'V_ZL_A', 'V_ZL_B') if (G == g).any()}
                # kill control: how often do held-out TRAINING corpora get absorbed by ASEMIC? (LOCO with ASEMIC present)
                absorbed = 0; tot = 0
                for c in sorted(set(corp[tr])):
                    te = tr & (corp == c)
                    keep = tr & ~te
                    Pk, clk = fit(np.vstack([X[keep], X[m1], X[m2]]), np.r_[cls[keep], ['ASEMIC'] * (m1.sum() + m2.sum())])
                    absorbed += clk[int(Pk(X[te]).mean(0).argmax())] == 'ASEMIC'; tot += 1
                asem[cs_var or ':raw'] = {'heldout_vol': ctl, 'voy': pv, 'absorbed': f'{absorbed}/{tot}'}
            out[f'{scale}|{world}'] = {'keys': keys, 'med': med, 'shared': shared, 'shared_col': shared_c, 'vdep': vdep,
                                       'nearest': nearest, 'loco': ba, 'per': pc, 'post': post, 'asem': asem,
                                       'lang_lo': lo, 'lang_hi': hi}
            print(scale, world, 'LOCO', round(ba, 2), pc)
            print('  Voynich departs on', vdep, '; shared with CS raw', shared, '; with CS col', shared_c)
            print('  nearest class per metric', nearest)
            for g, p in post.items(): print('  post', g, p)
            print('  ASEMIC', asem, flush=True)
    L.save('cycle2.json', out)
    # ---------- rows
    n = 1
    show = ['LANG', 'INVENT', 'CIPH', 'GEN', 'GIBB', 'S_CS', 'S_CS:col', 'V_ZL', 'V_ZL_A', 'V_ZL_B']
    for scale in ('LONG', 'SHORT'):
        o = out[f'{scale}|n18']
        tab = '; '.join(f"{k}: " + ' '.join(f"{g.replace('S_CS', 'CS').replace('V_ZL', 'V')} {o['med'][g][k]:.2f}" for g in show if g in o['med']) for k in o['keys'])
        L.row(FN, f'V-42.2.{n}', f'{scale} scale ({"3000" if scale == "LONG" else "250"}-token blocks), n18 world: class medians per metric (LANG/INVENT/CIPH/GEN/GIBB classes; CS raw / collapsed; Voynich ZL, A, B)', tab, 'descriptive'); n += 1
        for world in ('n18', 'clean'):
            o = out[f'{scale}|{world}']
            L.row(FN, f'V-42.2.{n}', f'{scale}/{world}: metrics where the Voynich median leaves the 5-95% language range, and where the CS leaves it on the same side; nearest class median per metric',
                  f"Voynich departs on {len(o['vdep'])}/{len(o['keys'])} ({', '.join(o['vdep'])}); same side as CS raw {len(o['shared'])} ({', '.join(o['shared'])}); as CS collapsed {len(o['shared_col'])} ({', '.join(o['shared_col'])}); nearest per metric: {dict(Counter(o['nearest'].values()))}",
                  'see verdict'); n += 1
            L.row(FN, f'V-42.2.{n}', f'{scale}/{world}: LOCO logistic on the battery (control: held-out corpora), Voynich and CS never trained on',
                  f"LOCO {o['loco']:.2f} {o['per']}; " + '; '.join(f"{g}: " + ', '.join(f'{c} {v:.2f}' for c, v in sorted(p.items(), key=lambda x: -x[1]) if v >= 0.05) for g, p in o['post'].items()),
                  'see verdict'); n += 1
            a = o['asem']
            L.row(FN, f'V-42.2.{n}', f'{scale}/{world}: 6th class ASEMIC = CS vol 1 + vol 2. Controls: the held-out volume must be classed ASEMIC; kill control = share of held-out training corpora absorbed by ASEMIC',
                  '; '.join(f"CS{v}: held-out volumes -> {a[v]['heldout_vol']}, training corpora absorbed {a[v]['absorbed']}, Voynich " + ', '.join(f"{g} " + '/'.join(f'{c} {p:.2f}' for c, p in sorted(pp.items(), key=lambda x: -x[1])[:2]) for g, pp in a[v]['voy'].items()) for v in a),
                  'see verdict'); n += 1


if __name__ == '__main__':
    main()
