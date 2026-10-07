"""v96 cycle 1: select views on train folios (plants only, no Voynich, no fresh plants), freeze views + thresholds +
predictions (sha256), then (argument 'report') summarise the held-out run."""
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L

TOP = 20
FZ = os.path.join(L.DATA, 'v96_frozen_c1.json')


def load(name, h):
    p = os.path.join(L.CK, 'c1_%s_h%d.json' % (name, h))
    return json.load(open(p))['res'] if os.path.exists(p) else None


def select():
    import v96_c1 as C
    rng = np.random.default_rng(96001)
    views = [C.strata_view(rng, sorted(set('abcdefghijklmnopqrstuvwxyzCSTKPF'))) for _ in range(C.NV)]
    plants = [n for n in L.all_names() if L.world(n) in ('W0', 'W1', 'W2', 'W12')]
    R = {n: load(n, 0) for n in plants}
    scored = []
    for i, v in enumerate(views):
        pos = [R[n][i]['a'] for n in plants if L.world(n) in ('W2', 'W12') and R[n][i]]
        neg = [R[n][i]['a'] for n in plants if L.world(n) in ('W0', 'W1') and R[n][i]]
        if len(pos) < 25 or len(neg) < 9: continue
        auc = np.mean([[p > q for q in neg] for p in pos])
        cut = 0.5 * (min(pos) + max(neg)) if min(pos) > max(neg) else float(np.median(pos + neg))
        marg = (min(pos) - max(neg)) / (np.std(pos + neg) + 1e-9)
        scored.append((auc, marg, i, cut, max(neg)))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    sel = scored[:TOP]
    fz = dict(views=[views[i] for _, _, i, _, _ in sel], idx=[i for _, _, i, _, _ in sel],
              cut=[c for *_, c, _ in sel], train_auc=[a for a, *_ in sel], train_margin=[m for _, m, *_ in sel],
              n_scored=len(scored),
              predictions=[
                  'P1: in each of ZL3b, IT2a, GC2a the held-out median over the 20 frozen views of aC-0.5 exceeds the largest '
                  'held-out median of every W1 plant, trained and fresh (P_*, F_MACER, F_PLINY, E_KONRAD, E_CIRCA, L_KONRAD, L_CIRCA)',
                  'P2: in each Voynich transcription at least 15 of 20 frozen views put held-out aC above their frozen cut (key present)',
                  'P3: with candidate pages within 4 positions excluded (aC_far), P1 still holds',
                  'P4 (method check): held-out, the frozen cuts classify >= 90% of planted W0/W1 vs W2/W12 corpora by majority vote'])
    s = json.dumps(fz, sort_keys=True, default=str)
    open(FZ, 'w').write(s)
    open(FZ.replace('.json', '.sha256'), 'w').write(hashlib.sha256(s.encode()).hexdigest() + '  v96_frozen_c1.json\n')
    print('scored', len(scored), 'top auc', [round(a, 3) for a, *_ in sel][:5], 'margin', [round(m, 2) for _, m, *_ in sel][:5])
    print('sha256', hashlib.sha256(s.encode()).hexdigest())


def report():
    fz = json.load(open(FZ)); cut = np.array(fz['cut'])
    names = L.all_names() + L.FRESH_W1
    out = {}
    for n in names:
        r = load(n, 1)
        if r is None: continue
        a = np.array([x['a'] if x else np.nan for x in r]); af = np.array([x['aC_far'] - 0.5 if x else np.nan for x in r])
        ar = np.array([x['aR'] - 0.5 if x else np.nan for x in r])
        g = lambda k: float(np.nanmedian([x[k] - 0.5 if x and x.get(k) is not None else np.nan for x in r]))
        out[n] = dict(w=L.world(n), a=float(np.nanmedian(a)), a_far=float(np.nanmedian(af)), aR=float(np.nanmedian(ar)),
                      votes=int(np.nansum(a > cut)), n=int(np.nanmedian([x['n'] for x in r if x])),
                      a_core=g('aC_core'), aR_core=g('aR_core'), a_nf=g('aC_nf'))
    for n, d in sorted(out.items(), key=lambda kv: (kv[1]['w'], kv[0])):
        print('%-14s %-4s aC-.5 %.3f far %.3f aR-.5 %.3f votes %2d/20 pages %d | core aC %.3f aR %.3f | no-first-para aC %.3f' % (
            n, d['w'], d['a'], d['a_far'], d['aR'], d['votes'], d['n'], d['a_core'], d['aR_core'], d['a_nf']))
    w1 = [d for d in out.values() if d['w'] in ('W1', 'W1F')]
    mx = max(d['a'] for d in w1); mxf = max(d['a_far'] for d in w1)
    cls = [(d['votes'] > 10) == (d['w'] in ('W2', 'W12')) for d in out.values() if d['w'] in ('W0', 'W1', 'W2', 'W12', 'W1F')]
    P = {}
    for v in L.VOY:
        d = out[v]
        P[v] = dict(P1=d['a'] > mx, P2=d['votes'] >= 15, P3=d['a_far'] > mxf)
    print('W1 max a %.3f far %.3f; plant classification %d/%d' % (mx, mxf, sum(cls), len(cls)))
    print(P)
    json.dump(dict(out=out, P=P, w1max=mx, w1max_far=mxf, cls=[sum(cls), len(cls)]), open(os.path.join(L.CK, 'c1_report.json'), 'w'))


if __name__ == '__main__':
    report() if sys.argv[1:] == ['report'] else select()
