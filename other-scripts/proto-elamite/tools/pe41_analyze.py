"""pe41 analysis: pool votes, run controls.
usage: python3 pe41_analyze.py OUT.json run1.jsonl.gz [run2 ...] [--top Q]  (Q = keep best-Q fraction of populations per target)
"""
import sys, os, json, gzip, math
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import LABELS, STATN, DATA, load_real, corpus_stats

CK = os.path.join(DATA, 'pe41_ckpt')
NL = len(LABELS)
# pre-registered proto-cuneiform calibration lists (standard sign values; control only)
PC_KNOWN = {
    'COMMODITY': ['SZE~a', 'GAR', 'SZE3', '|SZE~a&SZE~a|', 'UDU~a', 'U8', 'UDUNITA~a', 'MASZ', 'MASZ2', 'KISZ', 'UD5~a',
                  'AB2', 'GU4', 'AMAR', 'SILA4~c', 'SILANITA', 'KIR11', 'GUKKAL~a', 'SZEG9', 'SAL', 'KUR~a', 'ERIM~a',
                  'GURUSZ~a', '|SAL.KUR~a|', 'TUG2~a', 'DUG~a', 'DUG~b', 'DUG~c', 'KU6~a', 'GA~a', 'KU3~a', 'URUDU~a',
                  'KASZ~a', 'KASZ~b', 'KASZ~c', 'KASZ~d', 'MUSZEN', 'SIG2~a1', 'SIG2~a2', 'SIG2~a3', 'SIG2~b', 'GADA~a',
                  'SUHUR', 'GARA2~a', 'SILA3~a', 'MUN~a1', 'GISZ'],
    'OFFICE': ['EN~a', 'EN~b', 'SANGA~a', 'SANGA~b', 'SUKKAL', 'KINGAL'],
    'PLACE': ['UNUG~a', 'ADAB', 'SZURUPPAK~a', 'ZABALAM~a', 'DILMUN', 'IDIGNA', 'URI3~a', 'KALAM~a', 'KALAM~b'],
    'ACTION': ['GU7', 'BA', 'DU', 'RU', 'GI4~a'],
}


def load(files):
    runs = []
    for fn in files:
        for l in gzip.open(fn, 'rt'):
            r = json.loads(l)
            if 'res' in r:
                runs.append(r)
    return runs


def votes(runs, tname, nsig, keep=None):
    V = np.zeros((nsig, NL))
    for k, r in enumerate(runs):
        if keep is not None and not keep[k]:
            continue
        v = np.array(r['res'][tname]['v'])
        ok = v >= 0
        np.add.at(V, (np.where(ok)[0], v[ok]), 1)
    return V


def keepmask(runs, tname, q):
    if q >= 1:
        return np.ones(len(runs), bool)
    c = np.array([r['res'][tname]['cost'] for r in runs])
    return c <= np.quantile(c, q)


def stable(runs, tname, nsig, q, zmin=3.0, minv=20):
    """split populations by seed parity; a vote is stable if the same label is enriched z>=zmin in both halves."""
    km = keepmask(runs, tname, q)
    par = np.array([r['seed'] % 2 for r in runs])
    out = []
    Vs = [votes(runs, tname, nsig, km & (par == h)) for h in (0, 1)]
    Vall = Vs[0] + Vs[1]
    base = Vall.sum(0) / max(Vall.sum(), 1)
    for i in range(nsig):
        zs = []
        for V in Vs:
            n = V[i].sum()
            if n < minv:
                zs.append(None); continue
            sh = V[i] / n
            z = (sh - base) / np.sqrt(base * (1 - base) / n + 1e-12)
            zs.append(z)
        if zs[0] is None or zs[1] is None:
            continue
        zz = np.minimum(zs[0], zs[1])
        l = int(np.argmax(zz))
        if zz[l] >= zmin:
            n = Vall[i].sum()
            out.append(dict(i=i, label=LABELS[l], zmin=float(zz[l]), share=float(Vall[i, l] / n), base=float(base[l]), n=int(n)))
    return out, Vall, base


def heldout_acc(runs, meta, q):
    res = []
    for t in sorted(k for k in meta if k.startswith('HELD')):
        truth = np.array(meta[t]['truth'])
        km = keepmask(runs, t, q)
        V = votes(runs, t, len(truth), km)
        ok = (V.sum(1) > 0) & (truth >= 0)
        pred = V.argmax(1)
        acc = float(np.mean(pred[ok] == truth[ok]))
        rng = np.random.default_rng(0)
        perm = [np.mean(pred[ok] == rng.permutation(truth[ok])) for _ in range(500)]
        maj = float(np.max(np.bincount(truth[ok], minlength=NL)) / ok.sum())
        # enrichment prediction (share / pooled base) scored by balanced accuracy
        base = V.sum(0) / max(V.sum(), 1)
        pe_ = (V / np.maximum(V.sum(1, keepdims=True), 1) / np.maximum(base, 1e-9)).argmax(1)
        def bal(pr, tr):
            ls = np.unique(tr); return float(np.mean([np.mean(pr[tr == l] == l) for l in ls]))
        b = bal(pe_[ok], truth[ok])
        bperm = [bal(pe_[ok], rng.permutation(truth[ok])) for _ in range(500)]
        res.append(dict(t=t, n=int(ok.sum()), acc=acc, perm_mean=float(np.mean(perm)), p=float(np.mean(np.array(perm) >= acc)), majority=maj,
                        bal=b, bal_perm=float(np.mean(bperm)), bal_p=float(np.mean(np.array(bperm) >= b))))
    return res


def pc_control(runs, meta, q):
    signs = meta['PC']['signs']
    km = keepmask(runs, 'PC', q)
    V = votes(runs, 'PC', len(signs), km)
    share = V / np.maximum(V.sum(1, keepdims=True), 1)
    out = {}
    rng = np.random.default_rng(1)
    for cat, lst in PC_KNOWN.items():
        idx = [signs.index(s) for s in lst if s in signs]
        if not idx:
            continue
        l = LABELS.index(cat)
        obs = float(share[idx, l].mean())
        null = [share[rng.choice(len(signs), len(idx), replace=False), l].mean() for _ in range(2000)]
        top = float(np.mean(share[idx].argmax(1) == l))
        out[cat] = dict(n=len(idx), mean_share=obs, null_mean=float(np.mean(null)), p=float(np.mean(np.array(null) >= obs)), top_frac=top)
    # frequency-matched null for COMMODITY (frequency is a big feature): compare with signs of similar rank
    idx = [signs.index(s) for s in PC_KNOWN['COMMODITY'] if s in signs]
    l = LABELS.index('COMMODITY')
    obs = share[idx, l].mean(); nulls = []
    for _ in range(2000):
        pick = [min(len(signs) - 1, max(0, i + int(rng.integers(-8, 9)))) for i in idx]
        nulls.append(share[pick, l].mean())
    out['COMMODITY_rankmatched'] = dict(obs=float(obs), null_mean=float(np.mean(nulls)), p=float(np.mean(np.array(nulls) >= obs)))
    return out


def econ_fit(runs, tname='PE'):
    c = np.array([r['res'][tname]['cost'] for r in runs])
    keys = [k for k, v in runs[0]['p'].items() if isinstance(v, (int, float))]
    out = {}
    best = c <= np.quantile(c, 0.05)
    for k in keys + ['L'] + ['pol_' + x for x in runs[0]['pol']]:
        if k == 'L':
            x = np.array([r['L'] for r in runs], float)
        elif k.startswith('pol_'):
            x = np.array([r['pol'][k[4:]] for r in runs], float)
        else:
            x = np.array([r['p'][k] for r in runs], float)
        rho, p = spearmanr(x, c)
        out[k] = dict(rho=float(rho), p=float(p), best5=float(x[best].mean()), all=float(x.mean()))
    for j, n in enumerate(['ration', 'delivery', 'labor', 'inventory', 'debt']):
        x = np.array([r['p']['mix'][j] for r in runs])
        rho, p = spearmanr(x, c)
        out['mix_' + n] = dict(rho=float(rho), p=float(p), best5=float(x[best].mean()), all=float(x.mean()))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    q = 1.0
    if '--top' in args:
        i = args.index('--top'); q = float(args[i + 1]); del args[i:i + 2]
    outf = args[0]; files = args[1:]
    runs = load(files)
    meta = json.load(open(os.path.join(CK, 'targets_meta.json')))
    R = dict(n_pop=len(runs), q=q)
    for t in ['PE', 'SHUF0', 'SHUF1', 'SHUF2', 'PEA', 'PEB', 'PC']:
        st, V, base = stable(runs, t, len(meta[t]['signs']), q)
        for s in st:
            s['sign'] = meta[t]['signs'][s['i']]
        R[t] = dict(n_stable=len(st), base={LABELS[i]: float(b) for i, b in enumerate(base)}, stable=st,
                    cost_mean=float(np.mean([r['res'][t]['cost'] for r in runs])))
    # tablet-split replication: same sign, top-label agreement PEA vs PEB vs chance
    sa, sb = meta['PEA']['signs'], meta['PEB']['signs']
    VA = votes(runs, 'PEA', len(sa), keepmask(runs, 'PEA', q)); VB = votes(runs, 'PEB', len(sb), keepmask(runs, 'PEB', q))
    common = [s for s in sa if s in sb]
    ta = np.array([VA[sa.index(s)].argmax() for s in common if VA[sa.index(s)].sum() >= 20 and VB[sb.index(s)].sum() >= 20])
    tb = np.array([VB[sb.index(s)].argmax() for s in common if VA[sa.index(s)].sum() >= 20 and VB[sb.index(s)].sum() >= 20])
    rng = np.random.default_rng(2)
    perm = [np.mean(ta == rng.permutation(tb)) for _ in range(1000)]
    R['split_agree'] = dict(n=int(len(ta)), agree=float(np.mean(ta == tb)), perm_mean=float(np.mean(perm)), p=float(np.mean(np.array(perm) >= np.mean(ta == tb))))
    R['heldout'] = heldout_acc(runs, meta, q)
    R['pc'] = pc_control(runs, meta, q)
    R['econ_PE'] = econ_fit(runs, 'PE'); R['econ_SHUF0'] = econ_fit(runs, 'SHUF0'); R['econ_PC'] = econ_fit(runs, 'PC')
    json.dump(R, open(outf, 'w'), indent=1)
    # print summary
    print('pops', len(runs), 'q', q)
    for t in ['PE', 'SHUF0', 'SHUF1', 'SHUF2', 'PEA', 'PEB', 'PC']:
        print(t, 'stable', R[t]['n_stable'], 'cost', round(R[t]['cost_mean'], 3), {k: round(v, 3) for k, v in R[t]['base'].items()})
    print('split', R['split_agree'])
    ha = R['heldout']
    print('heldout mean acc %.3f perm %.3f majority %.3f; p<0.05 in %d/%d' % (np.mean([h['acc'] for h in ha]), np.mean([h['perm_mean'] for h in ha]), np.mean([h['majority'] for h in ha]), sum(h['p'] < 0.05 for h in ha), len(ha)))
    print('heldout balanced acc (enrichment vote) %.3f perm %.3f; p<0.05 in %d/%d' % (np.mean([h['bal'] for h in ha]), np.mean([h['bal_perm'] for h in ha]), sum(h['bal_p'] < 0.05 for h in ha), len(ha)))
    for k, v in R['pc'].items():
        print('PC', k, v)
    for s in sorted(R['PE']['stable'], key=lambda s: -s['zmin'])[:40]:
        print('PE', s['sign'], s['label'], round(s['zmin'], 1), round(s['share'], 2), round(s['base'], 2), s['n'])
    for t in ['SHUF0']:
        for s in sorted(R[t]['stable'], key=lambda s: -s['zmin'])[:10]:
            print(t, s['sign'], s['label'], round(s['zmin'], 1), round(s['share'], 2), s['n'])
    e = R['econ_PE']
    for k in sorted(e, key=lambda k: e[k]['p'])[:14]:
        print('econPE', k, round(e[k]['rho'], 3), '%.1e' % e[k]['p'], round(e[k]['best5'], 3), round(e[k]['all'], 3), ' shuf rho', round(R['econ_SHUF0'][k]['rho'], 3), ' pc rho', round(R['econ_PC'][k]['rho'], 3))
