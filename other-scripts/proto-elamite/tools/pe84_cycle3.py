"""pe84 cycle 3: does the impression meter see anything we KNOW is there? Within-tablet calibration in two scripts.
Known physical fact (not a PE interpretation): numerals in both proto-cuneiform and PE are impressed with the round
stylus end (large, deep pits); signs are drawn/impressed with the stylus tip (thin lines). So, inside one tablet, the face
carrying the larger share of numeral marks should show wider, deeper impressions (wid, G3+G4 share, depth).
Test: across tablets with both faces inscribed, x = numeral-mark share difference (reverse - obverse), y = physical
difference (reverse - obverse); partial on ink and brightness differences; stratified permutation; halves.
Then the clay-clock primary (cycle 1) is repeated on proto-cuneiform: total-only reverse vs continuation reverse.
usage: python3 pe84_cycle3.py pe <feats.json> | pc <feats.json> <pc_corpus.json>"""
import sys, os, json, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe84_common as C

rng = np.random.default_rng(843)
KEYS = ('wid', 'depth', 'Gbig', 'E16', 'E04', 'ink')


def facetext(lines, surf):
    L = [l for l in lines if l['surface'] == surf]
    marks = sum((n or 0) for l in L for n, c in l['numerals'])
    signs = sum(1 for l in L for g in l['signs'] if g != 'x')
    return marks, signs, L


def phys(fe, face):
    f = fe.get(face)
    if not f:
        return None
    g = [f.get('G%d' % k) for k in range(1, 5)]
    d = dict(f)
    d['Gbig'] = (g[2] + g[3]) / sum(g) if None not in g else None
    return d


def rows_generic(PH, texts):
    out = []
    for pid, lines in texts.items():
        fe = PH.get(pid)
        if not fe or 'err' in fe:
            continue
        o, r = phys(fe, 'ob'), phys(fe, 'rv')
        if not o or not r:
            continue
        mo, so, Lo = facetext(lines, 'obverse'); mr, sr, Lr = facetext(lines, 'reverse')
        if not Lr or not Lo or (mo + so) < 2 or (mr + sr) < 1:
            continue
        x = mr / max(1, mr + sr) - mo / max(1, mo + so)
        row = dict(id=pid, x=x, ndr=len(Lr), tot=float(len(Lr) <= 2 and all(l['numerals'] for l in Lr)),
                   cont=float(len(Lr) >= 3), bright=o['bright'], rvb=r['bright'], so=o['s_orig'])
        for k in KEYS:
            row['d_' + k] = (r[k] - o[k]) if (r.get(k) is not None and o.get(k) is not None) else np.nan
        row['ob_ink'] = o['ink']; row['rv_ink'] = r['ink']; row['amt'] = np.log1p(mr + sr) - np.log1p(mo + so)
        out.append(row)
    return out


def test(R, xk, yk, B=2000, extra=('d_ink', 'amt')):
    x = np.array([r[xk] for r in R], float); y = np.array([r[yk] for r in R], float)
    Z = np.column_stack([np.ones(len(R))] + [np.array([r[k] for r in R], float) for k in ('bright', 'rvb', 'so') + tuple(extra)])
    Z[~np.isfinite(Z)] = 0
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, Z = x[ok], y[ok], Z[ok]
    Q = np.linalg.qr(Z)[0]
    res = lambda v: v - Q @ (Q.T @ v)
    a = res(rankdata(x)); b = res(rankdata(y))
    r0 = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    cnt = 0; ry = rankdata(y)
    for _ in range(B):
        bp = res(rng.permutation(ry))
        cnt += abs(a @ bp / np.linalg.norm(a) / np.linalg.norm(bp)) >= abs(r0)
    return dict(r=round(r0, 3), n=int(ok.sum()), p=round((cnt + 1) / (B + 1), 4))


def main():
    which, featf = sys.argv[1], sys.argv[2]
    PH = json.load(open(featf))
    if which == 'pe':
        import common
        texts = {t['id']: t['lines'] for t in common.load()}
    else:
        texts = {t['id']: t['lines'] for t in json.load(open(sys.argv[3]))}
    R = rows_generic(PH, texts)
    out = {'script': which, 'n': len(R)}
    for k in KEYS:
        if k == 'ink':
            continue
        out['meter_' + k] = test(R, 'x', 'd_' + k)
        out['meter_noink_' + k] = test(R, 'x', 'd_' + k, extra=())
    # planted: shuffle x (must vanish) is the permutation null itself; split halves by id parity as replication
    for half in (0, 1):
        Rh = [r for r in R if int(r['id'][1:]) % 2 == half]
        out['meter_wid_half%d' % half] = test(Rh, 'x', 'd_wid', B=500)
        out['meter_Gbig_half%d' % half] = test(Rh, 'x', 'd_Gbig', B=500)
    S = [r for r in R if r['tot'] or r['cont']]
    out['n_total_only'] = int(sum(r['tot'] for r in S)); out['n_cont'] = int(sum(r['cont'] for r in S))
    for k in ('depth', 'E04', 'wid'):
        out['clock_' + k] = test(S, 'tot', 'd_' + k, extra=('d_ink', 'amt', 'x'))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(C.CK, 'c3_%s.json' % which), 'w'), indent=1)


if __name__ == '__main__':
    main()
