"""v42 cycle 2b: robustness of the cycle-2 placement.
(1) Line metrics (linit, lfin) dropped: line breaks are real in the Voynich, the CS and the gibberish samples but
    editorial in many training corpora (the v31 caveat), so they could fake a CS-Voynich bond.
(2) Specificity of 'shared departures': for EVERY corpus, the number of metrics on which it leaves the 5-95%
    language range on the same side as the Voynich; rank of the CS among all corpora.
(3) Out-of-distribution kill control for the ASEMIC class: objects never trained on that are neither CS nor
    Voynich (Martian glossolalia, Lingua Ignota list, Steganographia conjurations at SHORT scale; v21 forgers
    excluded since they copy the Voynich) must NOT be absorbed by ASEMIC.
"""
import os, sys, json, random, zlib
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v42_battery as B, v42_cycle2 as C2

FN = 'v42_cycle2.txt'
LINE = {'linit', 'lfin'}


def ood_rows():
    import v31_lib as L31
    C = json.load(open(L31.CORPORA))
    out = []
    for k in ('T_Martian', 'T_LinguaIgnota', 'T_Stegano'):
        d = C[k]['docs']
        for var, dd in (('clean', d), ('n18', L.noise_docs(d, 0.18, seed=7))):
            toks = [w for x in dd for l in x for w in l]
            n = min(250, len(toks))
            if n < 150: continue
            bl = B.blocks(dd, n, 1)
            for i, b in enumerate(bl):
                F = B.battery(b, random.Random(1), 'SHORT')
                out.append({'corpus': k, 'cls': 'TEST', 'var': var, 'scale': 'SHORT', 'i': i, 'F': F, 'ntok': n})
    return out


def main():
    R = L.load('battery.json')
    O = ood_rows()
    R2 = R + O
    res = {}
    for scale in ('LONG', 'SHORT'):
        for world in ('n18', 'clean'):
            def sel(r):
                if r['scale'] != scale: return False
                if r['corpus'].startswith('S_'): return r['var'] in ('clean', 'col')
                return r['var'] == world
            Rs = [r for r in R2 if sel(r)]
            keys = sorted({k for r in Rs for k in r['F']} - LINE - ({'gap'} if scale == 'SHORT' else set()))
            X = C2.mat(Rs, keys)
            grp = np.array([(r['corpus'] + (':col' if r['var'] == 'col' else '')) for r in Rs])
            corp = np.array([r['corpus'] for r in Rs]); cls = np.array([r['cls'] for r in Rs])
            isT = np.array([str(c).startswith(('S_', 'V_', 'T_')) for c in corp])
            train_cls = ['LANG', 'INVENT', 'CIPH', 'GEN'] + (['GIBB'] if scale == 'SHORT' else [])
            tr = np.isin(cls, train_cls) & ~isT
            # (2) shared departures, all corpora, line metrics excluded
            langm = cls == 'LANG'
            lo = np.nanpercentile(X[langm], 5, axis=0); hi = np.nanpercentile(X[langm], 95, axis=0)
            def sides(m):
                v = np.nanmedian(X[m], 0); return np.where(v > hi, 1, np.where(v < lo, -1, 0))
            sv = sides(grp == 'V_ZL')
            shared = {}
            for g in sorted(set(grp)):
                if g.startswith('V_'): continue
                s = sides(grp == g); shared[g] = int(np.sum((sv != 0) & (s == sv)))
            vals = sorted(shared.values(), reverse=True)
            rank_cs = {g: 1 + sum(v > shared[g] for v in shared.values() if True) for g in ('S_CS', 'S_CS:col') if g in shared}
            top = sorted(shared.items(), key=lambda x: -x[1])[:8]
            # (1)+(3) ASEMIC class without line metrics
            m1 = grp == 'S_CS1'; m2 = grp == 'S_CS2'
            ctl = []
            for a, b in ((m1, m2), (m2, m1)):
                P, cl = C2.fit(np.vstack([X[tr], X[a]]), np.r_[cls[tr], ['ASEMIC'] * a.sum()])
                ctl.append(cl[int(P(X[b]).mean(0).argmax())])
            P, cl = C2.fit(np.vstack([X[tr], X[m1], X[m2]]), np.r_[cls[tr], ['ASEMIC'] * (m1.sum() + m2.sum())])
            post = {g: dict(zip(cl, P(X[grp == g]).mean(0).round(2))) for g in ('V_ZL', 'V_IT', 'V_ZL_A', 'V_ZL_B', 'T_Martian', 'T_LinguaIgnota', 'T_Stegano') if (grp == g).any()}
            absorbed = 0; tot = 0
            for c in sorted(set(corp[tr])):
                te = tr & (corp == c); keep = tr & ~te
                Pk, clk = C2.fit(np.vstack([X[keep], X[m1], X[m2]]), np.r_[cls[keep], ['ASEMIC'] * (m1.sum() + m2.sum())])
                absorbed += clk[int(Pk(X[te]).mean(0).argmax())] == 'ASEMIC'; tot += 1
            # 4-class (no ASEMIC) LOCO without line metrics
            per = defaultdict(list)
            for c in sorted(set(corp[tr])):
                te = tr & (corp == c); Pq, clq = C2.fit(X[tr & ~te], cls[tr & ~te])
                per[cls[te][0]].append(clq[int(Pq(X[te]).mean(0).argmax())] == cls[te][0])
            ba = float(np.mean([np.mean(v) for v in per.values()]))
            res[f'{scale}|{world}'] = dict(nkeys=len(keys), vdep=int(np.sum(sv != 0)), shared_top=top, rank_cs=rank_cs,
                                          ncorp=len(shared), heldout=ctl, post=post, absorbed=f'{absorbed}/{tot}', loco=ba)
            print(scale, world, res[f'{scale}|{world}'], flush=True)
    L.save('cycle2b.json', res)
    n = 20
    for k, r in res.items():
        L.row(FN, f'V-42.2.{n}', f'{k}, LINE METRICS DROPPED: (a) for every corpus, metrics leaving the language 5-95% range on the same side as the Voynich, rank of the CS; (b) ASEMIC class (CS vol 1 + 2) control = held-out volume, kill = held-out training corpora and never-trained odd texts (Martian, Lingua Ignota, Steganographia) must not be absorbed',
              f"Voynich departs on {r['vdep']}/{r['nkeys']}; most shared: {[(g.replace('S_CS', 'CS'), v) for g, v in r['shared_top'][:6]]}; CS rank {r['rank_cs']} of {r['ncorp']}; held-out volume -> {r['heldout']}; training corpora absorbed {r['absorbed']}; posteriors " +
              '; '.join(f"{g}: " + '/'.join(f'{c} {p:.2f}' for c, p in sorted(pp.items(), key=lambda x: -x[1])[:2]) for g, pp in r['post'].items()) + f"; 4-class LOCO {r['loco']:.2f}",
              'see verdict'); n += 1


if __name__ == '__main__':
    main()
