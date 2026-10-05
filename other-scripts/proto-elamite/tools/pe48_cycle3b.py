"""pe48 cycle 3b: where do the Yahya-homed signs live at Susa? ('the Yahya desk at Susa')

Set Y = signs homed at Yahya by the cycle-2 survivors (P >= 0.5). Among Susa tablets:
 (1) co-occurrence: number of Susa tablets carrying 2+ signs of Y, vs 2,000 frequency-matched random sets
     (each sign replaced by a sign of similar Susa tablet frequency);
 (2) date/format proxy: mean tablet length (sign tokens) and share of single-sign entries in Susa tablets
     carrying a Y sign vs tablets carrying the matched random sets;
 (3) publication volume concentration (designation prefix) vs matched sets (chi-square-like L1 distance);
 (4) the hXRF import ST-11 = P009157 (Yahya clay found at Susa): does it carry Y signs?
"""
import json, os, random, re
import numpy as np
from collections import Counter
import pe48_lib as L

if __name__ == '__main__':
    tab = json.load(open(os.path.join(L.CK, 'c2_signs.json')))
    Y = [x['sign'] for x in tab if x['P']['Yahya'] >= 0.5]
    docs = L.pe_docs()
    sus = [d for d in docs if d['site'] == 'Susa']
    df = Counter(t for d in sus for t in set(d['toks']))
    vocab = [x['sign'] for x in tab]
    order = sorted(vocab, key=lambda s: df[s])
    rank = {s: i for i, s in enumerate(order)}
    sets = [set(d['toks']) for d in sus]
    lens = np.array([len(d['toks']) for d in sus])
    solo = np.array([sum(1 for _, r in d['roles'] if r == 'SOLO') / max(1, len(d['roles'])) for d in sus])
    from common import load
    des = {t['id']: t['designation'] for t in load()}
    vol = np.array([re.split(r'[ ,]', des[d['id']])[0] + ' ' + (re.split(r'[ ,]+', des[d['id']])[1] if len(re.split(r'[ ,]+', des[d['id']])) > 1 else '') for d in sus])

    def stats(S):
        S = set(S)
        k = np.array([len(S & s) for s in sets])
        m = k >= 1
        vc = Counter(vol[m]); allc = Counter(vol)
        n = m.sum()
        l1 = sum(abs(vc[v] / max(1, n) - allc[v] / len(sus)) for v in allc)
        return dict(co2=int((k >= 2).sum()), n=int(n), len=float(lens[m].mean()) if n else 0,
                    solo=float(solo[m].mean()) if n else 0, l1=float(l1), vol=vc.most_common(5))
    real = stats(Y)
    rng = random.Random(3)
    null = []
    for _ in range(2000):
        S = []
        for s in Y:
            i = rank[s]
            cand = [order[j] for j in range(max(0, i - 4), min(len(order), i + 5)) if order[j] not in Y and order[j] not in S]
            S.append(rng.choice(cand))
        null.append(stats(S))
    out = dict(Y=Y, real=real)
    for k in ('co2', 'len', 'solo', 'l1'):
        v = np.array([x[k] for x in null])
        out[k] = dict(real=real[k], null_mean=float(v.mean()), p_hi=float(np.mean(v >= real[k])),
                      p_lo=float(np.mean(v <= real[k])))
    # ST-11
    st = [d for d in docs if d['id'] == 'P009157']
    out['ST11'] = dict(found=bool(st), toks=st[0]['toks'] if st else None,
                       Y_signs=sorted(set(st[0]['toks']) & set(Y)) if st else None)
    # base-rate: share of Susa tablets carrying any Y sign
    out['susa_frac_any_Y'] = real['n'] / len(sus)
    json.dump(out, open(os.path.join(L.CK, 'c3b_summary.json'), 'w'), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
