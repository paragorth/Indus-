"""pe76 cycle 3.  'freeze': write data/pe76_frozen_span.json (+ sha256) from the cycle 1/2 results, with the
pre-registered outside predictions.  'outside': only then read excavation data (CDLI stratigraphic levels via
data/pe19_hidden.json, museum numbers and publication volumes via data/pe8_meta.json) and score them."""
import os, sys, json, collections, hashlib
import numpy as np
import pe76_common as P, common
from pe76_est import vshuffle

FROZEN = os.path.join(P.DATA, 'pe76_frozen_span.json')
T = common.load(); n = len(T); ids = [t['id'] for t in T]

if sys.argv[1] == 'freeze':
    c1 = json.load(open(os.path.join(P.CKPT, sys.argv[2])))
    c2a = json.load(open(os.path.join(P.CKPT, 'cycle2_varA.json')))
    c2b = json.load(open(os.path.join(P.CKPT, 'cycle2_varB.json')))
    fr = dict(
        span_full={k: v['real'] for k, v in c1.items() if k.startswith('seed')},
        span_halfA=c2a['real'], span_halfB=c2b['real'],
        prereg=[
            'P1 levels: if pooled S >= 2 careers, tablet pairs from DIFFERENT CDLI stratigraphic levels (Susa Acropole I, Malyan) share fewer variant habits than same-level pairs, beyond the variant-shuffle null; if S < 1 no difference.',
            'P2 museum/volume distance: if S >= 2, habit-sharing excess over the variant-shuffle null declines from near-Sb-number pairs (|d| <= 10, same volume) to same-volume far pairs to different-volume pairs, and the different-volume excess is < half the same-volume-far excess; if S < 1, same-volume-far ~ different-volume (flat beyond the lot).',
        ])
    s = json.dumps(fr, sort_keys=True, default=float)
    fr['sha256'] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(fr, open(FROZEN, 'w'), indent=1, default=float)
    print(fr['sha256'])
    sys.exit()

# ---------------- outside
fr = json.load(open(FROZEN)); h = fr.pop('sha256')
assert hashlib.sha256(json.dumps(fr, sort_keys=True, default=float).encode()).hexdigest() == h
names, sets = P.variant_markers(T)


def pair_share(sets):
    tab = collections.defaultdict(set)
    for j, s in enumerate(sets):
        for i in s:
            tab[i].add(j)
    return tab


def share_matrix(sets):
    X = P.to_matrix(sets, n)
    return (X @ X.T).tocsr()


real = share_matrix(sets)
nulls = [share_matrix(vshuffle(T, np.random.default_rng(3000 + k))) for k in range(50)]

# tablets that carry any variant token at all (eligible pairs)
hasv = np.zeros(n, bool)
for i, t in enumerate(T):
    hasv[i] = any(common.is_sign(s) and '~' in s and not s.startswith('|') for l in t['lines'] for s in l['signs'])


def score(pairs):
    pairs = [(a, b) for a, b in pairs if hasv[a] and hasv[b]]
    if not pairs:
        return dict(n=0)
    A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
    r = np.asarray(real[A, B]).ravel() > 0
    nl = np.array([(np.asarray(M[A, B]).ravel() > 0).mean() for M in nulls])
    return dict(n=len(pairs), real=float(r.mean()), null=float(nl.mean()), excess=float(r.mean() - nl.mean()),
                z=float((r.mean() - nl.mean()) / (nl.std() + 1e-9)), p=float((np.sum(nl >= r.mean()) + 1) / (len(nl) + 1)))


out = {'frozen_sha256': h}
idx = {p: i for i, p in enumerate(ids)}
hid = json.load(open(os.path.join(P.DATA, 'pe19_hidden.json')))
for key in ('susa_level', 'malyan_level'):
    L = {idx[p]: v for p, v in hid[key].items() if p in idx}
    ks = sorted(L)
    same = [(a, b) for x, a in enumerate(ks) for b in ks[x + 1:] if L[a] == L[b]]
    diff = [(a, b) for x, a in enumerate(ks) for b in ks[x + 1:] if L[a] != L[b]]
    out[key] = dict(n_tablets=len(ks), n_with_variants=int(sum(hasv[k] for k in ks)), same=score(same), diff=score(diff))

meta = json.load(open(os.path.join(P.DATA, 'pe8_meta.json')))
vol = np.array([meta[p]['pub_vol'] if p in meta else '' for p in ids])
mus = np.array([meta[p]['mus_num'] if (p in meta and meta[p].get('mus_prefix') == 'Sb' and meta[p].get('mus_num')) else -1 for p in ids], float)
rng = np.random.default_rng(76)
el = np.where(hasv & (vol != ''))[0]
near, far, other = [], [], []
for x, a in enumerate(el):
    for b in el[x + 1:]:
        if vol[a] == vol[b]:
            if mus[a] > 0 and mus[b] > 0 and abs(mus[a] - mus[b]) <= 10:
                near.append((a, b))
            else:
                far.append((a, b))
        else:
            other.append((a, b))
far = [far[i] for i in rng.choice(len(far), min(len(far), 60000), replace=False)]
other = [other[i] for i in rng.choice(len(other), min(len(other), 60000), replace=False)]
out['museum'] = dict(near=score(near), same_vol_far=score(far), diff_vol=score(other))
# per volume pair breakdown for the big Susa volumes
big = ['MDP 06', 'MDP 17', 'MDP 26', 'MDP 26S', 'TCL 32', 'MDP 31']
vv = {}
for i, a in enumerate(big):
    for b in big[i:]:
        A = [k for k in el if vol[k] == a]; B = [k for k in el if vol[k] == b]
        pr = [(x, y) for x in A for y in B if x < y] if a == b else [(x, y) for x in A for y in B]
        if len(pr) > 20000:
            pr = [pr[j] for j in rng.choice(len(pr), 20000, replace=False)]
        vv[a + '|' + b] = score(pr)
out['volume_pairs'] = vv
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(P.CKPT, 'cycle3_outside.json'), 'w'), indent=1)
