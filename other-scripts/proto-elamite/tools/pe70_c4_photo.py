"""pe70 cycle 4 (photo audit): does the text predict an UNRECORDED seal impression?

The sealed profile (cycle 1 logistic model, trained leave-one-volume-out so no tablet scores itself) is applied to
tablets with NO seal recorded anywhere (ATF, catalogue, pe33/pe35).  Out-of-corpus prediction: high-scoring
'unsealed' tablets show rolled seal impressions on their CDLI photographs more often than low-scoring ones.
Blind design: 15 top and 15 bottom tablets (same volumes, not fragments, photo available) plus 10 recorded-sealed
tablets (sensitivity of the coder) are shuffled under random names; the key is written to the checkpoint and read
only after every image has been coded.
usage: pe70_c4_photo.py select  -> scratchpad/pe70_photos/blind/*.jpg, data/pe70_ckpt/c4_key.json
       pe70_c4_photo.py score CODES.json -> data/pe70_ckpt/c4.json
"""
import json, os, sys, subprocess, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from pe70_common import get, CK, SCR
from pe70_feats import matrix
import warnings
warnings.filterwarnings('ignore')
OUT = os.path.join(SCR, 'pe70_photos', 'blind')


def lovo_scores(R):
    Xb, B, Xn, NN = matrix(R, 'pe')
    X = np.hstack([Xb, Xn]); y = np.array([r['sealed'] for r in R], int)
    vol = np.array([r['vol'] for r in R]); sc = np.full(len(R), np.nan)
    for v in np.unique(vol):
        te = vol == v
        m = LogisticRegression(C=0.1, max_iter=400, class_weight='balanced').fit(X[~te], y[~te])
        sc[te] = m.decision_function(X[te])
    return sc


def fetch(pid, dest):
    url = 'https://cdli.earth/dl/photo/%s.jpg' % pid
    r = subprocess.run(['curl', '-sS', '-L', '-o', dest, '-w', '%{http_code}', url], capture_output=True, text=True)
    return r.stdout.strip() == '200' and os.path.getsize(dest) > 20000


def select():
    rng = np.random.default_rng(706)
    R = get('pe')
    sc = lovo_scores(R)
    vols = ('MDP 06', 'MDP 17', 'MDP 26', 'MDP 26S', 'MDP 31', 'TCL 32')
    ok = [i for i, r in enumerate(R) if not r['sealed'] and r['vol'] in vols and r['pres'] != 'fragment' and r['n_lines'] >= 2]
    order = sorted(ok, key=lambda i: -sc[i])
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(SCR, 'pe70_photos', 'tmp.jpg')
    picked = {'top': [], 'bottom': [], 'sealed': []}

    def take(lst, grp, n):
        for i in lst:
            if len(picked[grp]) >= n:
                break
            if fetch(R[i]['id'], tmp):
                picked[grp].append(i)
                os.replace(tmp, os.path.join(SCR, 'pe70_photos', R[i]['id'] + '.jpg'))
    take(order, 'top', 15)
    take(order[::-1], 'bottom', 15)
    sealed = [i for i, r in enumerate(R) if r['sealed'] and r['pres'] != 'fragment' and r['vol'] in vols]
    take(list(rng.permutation(sealed)), 'sealed', 10)
    allp = [(g, i) for g, L in picked.items() for i in L]
    perm = rng.permutation(len(allp))
    from PIL import Image
    key = {}
    for k, j in enumerate(perm):
        g, i = allp[j]
        name = 'img%02d' % k
        im = Image.open(os.path.join(SCR, 'pe70_photos', R[i]['id'] + '.jpg'))
        im.thumbnail((800, 1500))
        im.convert('RGB').save(os.path.join(OUT, name + '.jpg'), quality=88)
        key[name] = dict(group=g, id=R[i]['id'], score=float(sc[i]), vol=R[i]['vol'])
    json.dump(key, open(os.path.join(CK, 'c4_key.json'), 'w'), indent=1)
    print({g: len(L) for g, L in picked.items()}, 'images', len(key))


def score(codes_path):
    from scipy.stats import fisher_exact
    key = json.load(open(os.path.join(CK, 'c4_key.json')))
    codes = json.load(open(codes_path))
    tab = collections.defaultdict(collections.Counter)
    for name, c in codes.items():
        if c != 'unclear':
            tab[key[name]['group']][c] += 1
    out = {g: dict(v) for g, v in tab.items()}
    t, b = tab['top'], tab['bottom']
    yes = lambda c: c['clear'] + c['possible']
    odds, p = fisher_exact([[yes(t), sum(t.values()) - yes(t)], [yes(b), sum(b.values()) - yes(b)]], alternative='greater')
    oddc, pc = fisher_exact([[t['clear'], sum(t.values()) - t['clear']], [b['clear'], sum(b.values()) - b['clear']]], alternative='greater')
    out['p_any'] = p; out['p_clear'] = pc
    out['rows'] = {n: dict(key[n], code=codes[n]) for n in codes}
    json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
    print({k: v for k, v in out.items() if k != 'rows'})


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select()
    else:
        score(sys.argv[2])
