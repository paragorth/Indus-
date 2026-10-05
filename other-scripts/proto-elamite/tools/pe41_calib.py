"""pe41 cycle 3: let a real, read script choose which machine societies are trustworthy.
Each invented system (population) is scored on how well its alignment labels proto-cuneiform signs of known class
(grain/animal/worker/product = COMMODITY; titles = OFFICE; towns = PLACE; GU7/BA/DU/RU/GI4 = ACTION).
Cross-validation: choose populations on half of the known PC signs, test the pooled vote on the other half (50 splits),
against the unselected pool and random pools of the same size. Then the PC-selected pool votes on PE, with the
PE-half / shuffled-half replication control.
usage: python3 pe41_calib.py OUT.json META.json runs.jsonl.gz [...]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import LABELS
from pe41_analyze import load, votes, PC_KNOWN
from pe41_analyze2 import replicate

NL = len(LABELS)


def known(meta, tname='PC'):
    sg = meta[tname]['signs']
    idx, lab = [], []
    for c, l in PC_KNOWN.items():
        for s in l:
            if s in sg:
                idx.append(sg.index(s)); lab.append(LABELS.index(c))
    return np.array(idx), np.array(lab)


def pop_matrix(runs, tname, idx):
    return np.array([[r['res'][tname]['v'][i] for i in idx] for r in runs])  # pops x known signs


def bal_acc(pred, truth):
    return float(np.mean([np.mean(pred[truth == l] == l) for l in np.unique(truth)]))


def pooled_pred(M, keep, truth_classes=None):
    """enrichment vote over the kept populations: share / pooled base share."""
    sub = M[keep]
    V = np.zeros((M.shape[1], NL))
    for j in range(M.shape[1]):
        v = sub[:, j]; v = v[v >= 0]
        V[j] = np.bincount(v, minlength=NL)
    base = V.sum(0) / max(V.sum(), 1)
    E = V / np.maximum(V.sum(1, keepdims=True), 1) / np.maximum(base, 1e-9)
    return E.argmax(1), V


def pop_score(M, truth, cols):
    """per-population balanced accuracy on the given known-sign columns (unmatched = wrong)."""
    S = np.zeros(M.shape[0])
    for l in np.unique(truth[cols]):
        c = cols[truth[cols] == l]
        S += (M[:, c] == l).mean(1)
    return S / len(np.unique(truth[cols]))


if __name__ == '__main__':
    outf, metaf = sys.argv[1], sys.argv[2]
    runs = load(sys.argv[3:]); meta = json.load(open(metaf))
    idx, truth = known(meta)
    M = pop_matrix(runs, 'PC', idx)
    rng = np.random.default_rng(7)
    R = dict(n_pop=len(runs), n_known=len(idx))
    for q in (0.05, 0.1, 0.2):
        res = []
        for rep in range(50):
            # stratified half split of known signs
            cols1, cols2 = [], []
            for l in np.unique(truth):
                c = rng.permutation(np.where(truth == l)[0]); h = len(c) // 2
                cols1 += list(c[:h]); cols2 += list(c[h:])
            cols1, cols2 = np.array(cols1), np.array(cols2)
            sc = pop_score(M, truth, cols1)
            keep = sc >= np.quantile(sc, 1 - q)
            pred, _ = pooled_pred(M[:, cols2], keep)
            sel = bal_acc(pred, truth[cols2])
            pred0, _ = pooled_pred(M[:, cols2], np.ones(len(runs), bool))
            allp = bal_acc(pred0, truth[cols2])
            rnd = []
            for _ in range(20):
                k = np.zeros(len(runs), bool); k[rng.choice(len(runs), keep.sum(), replace=False)] = True
                pr, _ = pooled_pred(M[:, cols2], k); rnd.append(bal_acc(pr, truth[cols2]))
            # per-population transfer: score on half 1 vs score on half 2
            sc2 = pop_score(M, truth, cols2)
            res.append((sel, allp, float(np.mean(rnd)), float(np.mean(np.array(rnd) >= sel)), float(np.corrcoef(sc, sc2)[0, 1])))
        a = np.array(res)
        R['cv_q%.2f' % q] = dict(selected=float(a[:, 0].mean()), unselected=float(a[:, 1].mean()), random_pool=float(a[:, 2].mean()),
                                  frac_splits_sel_beats_all=float(np.mean(a[:, 0] > a[:, 1])), mean_p_vs_random=float(a[:, 3].mean()),
                                  pop_score_transfer_r=float(a[:, 4].mean()), chance=1.0 / len(np.unique(truth)))
        print('q', q, {k: round(v, 3) for k, v in R['cv_q%.2f' % q].items()})
    # final selection on all known signs, then PE
    sc = pop_score(M, truth, np.arange(len(idx)))
    for q in (0.1, 0.2):
        keep = sc >= np.quantile(sc, 1 - q)
        sub = [r for r, k in zip(runs, keep) if k]
        out = {}
        for a, b in [('PEA', 'PEB'), ('SHUF0A', 'SHUF0B'), ('PCA', 'PCB'), ('PCSHA', 'PCSHB')]:
            rr = replicate(sub, meta, a, b, minv=max(10, int(30 * q)))
            out[a[:-1]] = rr
            print('q', q, 'replication', a[:-1], 'n', rr['n'], 'agree %.3f perm %.3f p %.3f meancor %.3f nrep %d' % (rr['agree'], rr['perm'], rr['p'], rr['mean_cor'], rr['n_rep']))
        # PE full-corpus votes from the selected pool
        sg = meta['PE']['signs']
        V = votes(sub, 'PE', len(sg))
        base = V.sum(0) / V.sum()
        rep = {r['sign']: r for r in out['PE']['rep']}
        rows = []
        for s, r in rep.items():
            i = sg.index(s) if s in sg else None
            if i is None: continue
            l = LABELS.index(r['label'])
            rows.append(dict(sign=s, label=r['label'], zmin_halves=r['zmin'], share=float(V[i, l] / V[i].sum()), base=float(base[l]), n=int(V[i].sum())))
        rows.sort(key=lambda r: -r['zmin_halves'])
        out['PE_votes'] = rows
        # economy of PC-calibrated societies
        eco = {}
        for k, v in runs[0]['p'].items():
            if isinstance(v, (int, float)):
                x = np.array([r['p'][k] for r in runs], float)
                eco[k] = dict(sel=float(x[keep].mean()), all=float(x.mean()), z=float((x[keep].mean() - x.mean()) / (x.std() / np.sqrt(keep.sum()) + 1e-12)))
        for k in runs[0]['pol']:
            x = np.array([r['pol'][k] for r in runs], float)
            eco['pol_' + k] = dict(sel=float(x[keep].mean()), all=float(x.mean()), z=float((x[keep].mean() - x.mean()) / (x.std() / np.sqrt(keep.sum()) + 1e-12)))
        x = np.array([r['L'] for r in runs], float)
        eco['L'] = dict(sel=float(x[keep].mean()), all=float(x.mean()), z=float((x[keep].mean() - x.mean()) / (x.std() / np.sqrt(keep.sum()) + 1e-12)))
        out['econ'] = eco
        R['sel_q%.2f' % q] = out
        print('q', q, 'econ of PC-calibrated pops:', [(k, round(eco[k]['sel'], 3), round(eco[k]['all'], 3), round(eco[k]['z'], 1)) for k in sorted(eco, key=lambda k: -abs(eco[k]['z']))[:10]])
        print('q', q, 'PE votes:', [(r['sign'], r['label'], round(r['zmin_halves'], 1)) for r in rows[:30]])
    json.dump(R, open(outf, 'w'), indent=1)
