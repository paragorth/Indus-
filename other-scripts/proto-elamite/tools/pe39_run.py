"""pe39 single run: python3 pe39_run.py SETTING SEED [arch] [steps] [tag]"""
import sys, os, json, random, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import *  # noqa

SET, SEED = sys.argv[1], int(sys.argv[2])
ARCH = sys.argv[3] if len(sys.argv) > 3 else 'gru'
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
TAG = sys.argv[5] if len(sys.argv) > 5 else 'c1'
out_fn = os.path.join(CK, '%s_%s_%s_%d.json' % (TAG, SET, ARCH, SEED))
if os.path.exists(out_fn):
    sys.exit(0)
t0 = time.time()
mode = 'abs' if ('UR3' in SET or 'ABS' in SET) else 'ncode'
pe = load_script('pe')
pc = load_script('pc')
gold = None
if SET.startswith('PLANT'):
    tabs = sorted({l['tab'] for l in pc}); r = random.Random(1234); r.shuffle(tabs)
    h1 = set(tabs[:len(tabs) // 2])
    A_all = [l for l in pc if l['tab'] in h1]
    B_all = [l for l in pc if l['tab'] not in h1]
    A_all = subsample_tablets(A_all, len(pe), seed=SEED)
    La, Lb = 'pq', 'pc'
    gold = 'identity'
elif SET.startswith('UR3'):
    A_all = load_ur3(max_lines=len(pe), seed=SEED)
    B_all = pc
    La, Lb = 'ur', 'pc'
    gold = json.load(open(os.path.join(CK, 'gold_ur3_pc.json')))
    gold.pop('_note', None)
else:
    A_all, B_all, La, Lb = pe, pc, 'pe', 'pc'
A_all = numeral_tokens(A_all, mode)
B_all = numeral_tokens(B_all, mode)
if SET.endswith('SHUF'):
    A_all = shuffle_numerals(A_all, seed=1000 + SEED)
A_tr, A_ho = tablet_split(A_all, seed=0)
B_tr, B_ho = tablet_split(B_all, seed=0)
V = Vocab({La: A_tr, Lb: B_tr})
Sa = [V.enc(La, l) for l in A_tr]; Sb = [V.enc(Lb, l) for l in B_tr]
Ha = [V.enc(La, l) for l in A_ho]; Hb = [V.enc(Lb, l) for l in B_ho]
P = numeral_profiles(V, {La: A_tr, Lb: B_tr})
plex = profile_lexicon(V, P, La, Lb)
cfg = {'seed': SEED, 'arch': ARCH, 'steps': STEPS, 'ae_steps': STEPS // 4,
       'akw': {'prof': P} if ARCH == 'prof' else {}}
model = train(cfg, {'A': (La, Sa), 'B': (Lb, Sb)}, V)
res = {'set': SET, 'seed': SEED, 'arch': ARCH, 'steps': STEPS, 'mode': mode,
       'nA': len(A_all), 'nB': len(B_all), 'V': len(V.itos)}
res['rt_ho_A'] = roundtrip_nll(model, V, Ha, La, Lb)
res['rt_ho_B'] = roundtrip_nll(model, V, Hb, Lb, La)
# lexicons from translating every A line (train + held-out) into B and back
allA = Sa + Ha; allB = Sb + Hb
outA = translate(model, V, allA, Lb)
outB = translate(model, V, allB, La)
lexAB = lexicon(V, allA, outA)
lexBA = lexicon(V, allB, outB)
res['lexAB'] = lexAB; res['plex'] = plex; res['lexBA'] = lexBA
# share of translated tokens that are numerals copied exactly
def numcopy(src, out):
    k = n = 0
    for a, b in zip(src, out):
        na = [i for i in a if i >= V.num_start]; nb = [i for i in b if i >= V.num_start]
        n += 1; k += (na == nb)
    return k / max(n, 1)
res['numcopy_AB'] = numcopy(allA, outA)
# known-answer scoring
def correct(s, t):
    a = s.split('|', 1)[1]; b = t.split('|', 1)[1]
    if gold == 'identity':
        return a == b
    if isinstance(gold, dict) and a in gold:
        return b in gold[a]
    return None
if gold is not None:
    sc = [(s, t[0], correct(s, t[0])) for s, t in lexAB.items() if correct(s, t[0]) is not None]
    res['known'] = sc
    res['prec_AB'] = sum(c for _, _, c in sc) / max(len(sc), 1)
    # reverse direction
    sc2 = []
    for s, t in lexBA.items():
        c = correct(t[0], s)
        if c is not None:
            sc2.append((s, t[0], c))
    res['prec_BA'] = sum(c for _, _, c in sc2) / max(len(sc2), 1)
    res['n_known'] = len(sc); res['n_known_BA'] = len(sc2)
    nsA = {s: t[4] for s, t in lexAB.items()}
    pk = [(s, t, correct(s, t), nsA.get(s, 0)) for s, t in plex.items() if correct(s, t) is not None]
    res['profile_known'] = pk
# held-out pair gains for the 30 most frequent source signs
fB = collections.Counter(V.itos[i] for s in Sb for i in s if V.is_sign(i))
rng = random.Random(SEED)
gains = {}
for s, t in sorted(lexAB.items(), key=lambda kv: -kv[1][4])[:30]:
    g = pair_gain(model, V, Ha, La, Lb, s, t[0], fB, rng)
    if g is not None:
        gains[s] = (t[0],) + g
res['gainAB'] = gains
res['sec'] = time.time() - t0
jdump(res, out_fn)
print(SET, SEED, ARCH, 'rtA %.3f rtB %.3f' % (res['rt_ho_A'], res['rt_ho_B']), 'prec', res.get('prec_AB'), res.get('prec_BA'),
      'numcopy %.2f' % res['numcopy_AB'], '%.0fs' % res['sec'])
