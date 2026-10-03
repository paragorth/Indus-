"""S-DARK-66 cycle 3: calibration of the class-combination statistics on name systems whose elements DO combine by class.
Corpora: Chinese given names (loop63 cn_given, 60,000 distinct 2-character names; classes = Kangxi radical of the character
from Unihan kRSUnicode grouped into 10 semantic radical groups fixed in this file), Chinese historical full names (cn_ancient,
3+ characters: non-adjacent pairs exist), Japanese given names (jp_given, kanji, same radical classes), Ur III seal-owner names
(loop56 ur3_names_elem, greedy-chunk elements; classes theophoric / kin / predicate-verb / other by a fixed list), and the Indus
middle (DESC classes; non-adjacent pairs as in cycles 1-2, and ALL within-middle pairs for comparability with 2-element names).
Statistics per corpus, same code: class x class table of element pairs within one name; Cramer's V of the table against the
expectation from elements permuted among names within strata (gender x length for Chinese, length for Ur III, site x type x
length for Indus; 300x) and V of the null tables themselves; same-class share O/E; 2-fold held-out class-pair gain in bits per
token (class of the earlier element -> class of the later one) vs 100 random partitions of the same elements; symmetry
corr(z(A->B), z(B->A)).  Chinese/Japanese capped at 20,000 names for speed; Indus at full size.
Usage: python3 tools/dark_loop66_c3.py <seq_raw|seq_strong|seq_all>
"""
import sys, json, collections, random, math, time, re, os
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop66_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 300; NRP = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rnd = random.Random(663); rng = np.random.default_rng(663); T0 = time.time()
out = [f'# S-DARK-66 cycle 3 ({LV}) {time.strftime("%Y-%m-%dT%H:%M")}']
def P(s): out.append(s); print(s, flush=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'

# ---------------------------------------------------------------- radical classes (Kangxi numbers), fixed before any table was seen
RGROUP = {}
def _g(name, nums):
    for n in nums: RGROUP[n] = name
_g('human', [9, 10, 29, 30, 33, 37, 38, 39, 44, 61, 64, 77, 109, 128, 130, 132, 157, 158, 181, 185, 188, 190, 48, 59])
_g('animal', [93, 94, 123, 141, 142, 152, 153, 187, 195, 196, 198, 208, 212, 213, 124, 148, 87, 114])
_g('plant', [75, 115, 118, 119, 140, 97, 179, 199, 202, 45, 127, 156])
_g('nature', [85, 72, 74, 86, 173, 15, 46, 112, 32, 170, 47, 36, 84, 103, 1, 2, 3, 4])
_g('precious', [96, 167, 154, 65])
_g('textile', [120, 145, 50, 177, 178])
_g('speech', [149, 67, 73, 180, 129, 117, 113, 26, 68])
_g('object', [18, 19, 57, 62, 69, 70, 108, 111, 121, 134, 137, 159, 169, 40, 53, 63, 116, 56, 110, 151, 193, 24, 17, 22, 23, 31, 41, 51, 58, 71, 78, 79, 80, 81, 82, 90, 91, 98, 99, 100, 101, 102, 104, 105, 106, 107, 122, 125, 126, 131, 133, 135, 136, 138, 139, 143, 144, 146, 147, 150, 155, 160, 161, 162, 163, 164, 165, 166, 168, 171, 172, 174, 175, 176])
# everything else (numerals, radicals of position, rare) -> 'other'
def load_radicals():
    rad = {}
    for l in open(SCR + 'Unihan_IRGSources.txt'):
        if '\tkRSUnicode\t' in l:
            cp, _, v = l.rstrip('\n').split('\t')
            r = int(re.match(r'(\d+)', v.split()[0]).group(1))
            rad[chr(int(cp[2:], 16))] = r
    return rad
RAD = load_radicals()
def han_class(ch):
    r = RAD.get(ch)
    if r is None: return None
    return RGROUP.get(r, 'other')

# ---------------------------------------------------------------- Ur III element classes (fixed list)
UR3_DEITY = {'en-lil2', 'nanna', 'inanna', 'utu', 'szara2', 'nin-gir2-su', 'ba-ba6', 'suen', 'iszkur', 'dumu-zi', 'nin-szubur', 'nin-urta', 'en-ki', 'nin-hur-sag',
             'da-mu', 'nisaba', 'szul-gi', 'amar-suen', 'szu-suen', 'ur-namma', 'i-bi2-suen', 'lugal', 'nin', 'en', 'dingir', 'd', 'szu-{d}suen', 'szara', 'nin-mar{ki}',
             'nin-gesz-zi-da', 'ha-ia3', 'asznan', 'ig-alim', 'nin-dar-a', 'nin-a-zu', 'nergal', 'isztaran', 'hendur-sag', 'nin-ildu3', 'me-slam-ta-e3', 'ereszkigal'}
UR3_KIN = {'dumu', 'ama', 'ad-da', 'a-a', 'szesz', 'nin9', 'ur', 'lu2', 'geme2', 'arad2', 'ir11', 'dam', 'ibila', 'a-ba', 'a-bi', 'a-hu', 'ah'}
UR3_PRED = {'kal-la', 'zi', 'gi-na', 'ki-ag2', 'mah', 'gal', 'sa6', 'sa6-ga', 'du10', 'du10-ga', 'dadag', 'sig5', 'la2', 'mu', 'ba', 'i3', 'ib2', 'he2', 'ha', 'na', 'ma',
            'zu', 'e3', 'a', 'ga2', 'gu10', 'ka', 'kam', 'kam2', 'sa2', 'da', 'ta', 'ni', 'bi', 'me', 'ra', 'sze3', 'bad3', 'an-dul3', 'he2-gal2', 'tum', 'gin7', 'il2', 'li2', 'ti'}
def ur3_class(e):
    e2 = e.replace('{d}', 'd-').replace('{', '').replace('}', '')
    if e.startswith('{d}') or e in UR3_DEITY or e2 in UR3_DEITY or e.startswith('d-'): return 'theophoric'
    if e in UR3_KIN: return 'kin'
    if e in UR3_PRED: return 'predicate'
    return 'other'

# ---------------------------------------------------------------- generic statistics
def pairs_in(seq, pos, nonadj_only):
    out = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if (not nonadj_only or pos[j] - pos[i] >= 2) and seq[pos[i]] != seq[pos[j]]: out.append((seq[pos[i]], seq[pos[j]]))
    return out

def analyse(label, names, cls, strata, nonadj_only=False):
    """names: list of dict(seq, pos, strat); cls: element -> class"""
    classes = sorted(set(cls.values())); ci = {c: i for i, c in enumerate(classes)}; k = len(classes)
    def table(seqs):
        M = np.zeros((k, k))
        for nm, s in zip(names, seqs):
            for a, b in pairs_in(s, nm['pos'], nonadj_only):
                if a in cls and b in cls: M[ci[cls[a]], ci[cls[b]]] += 1
        return M
    O = table([nm['seq'] for nm in names])
    groups = collections.defaultdict(list)
    for i, nm in enumerate(names): groups[nm['strat']].append(i)
    def permuted():
        new = [list(nm['seq']) for nm in names]
        for idx in groups.values():
            pool = [names[i]['seq'][p] for i in idx for p in names[i]['pos']]; rnd.shuffle(pool); q = 0
            for i in idx:
                for p in names[i]['pos']: new[i][p] = pool[q]; q += 1
        return new
    N = np.array([table(permuted()) for _ in range(NPERM)])
    E = N.mean(0); SD = N.std(0) + 1e-9
    tot = O.sum()
    def V(M):
        m = E >= 1
        chi = (((M - E) ** 2 / np.where(m, E, 1))[m]).sum()
        return math.sqrt(chi / (M.sum() * (k - 1))) if M.sum() else float('nan')
    v = V(O); vn = [V(n) for n in N]
    same = float(np.trace(O)); samen = [float(np.trace(n)) for n in N]
    Z = (O - E) / SD
    iu = np.triu_indices(k, 1); m = (E[iu] >= 3) & (E.T[iu] >= 3)
    r = float(np.corrcoef(Z[iu][m], Z.T[iu][m])[0, 1]) if m.sum() > 3 else float('nan')
    Zs = (Z + Z.T) / 2
    cells = sorted([(classes[i], classes[j], O[i, j] + (O[j, i] if i != j else 0), E[i, j] + (E[j, i] if i != j else 0), Zs[i, j] * (math.sqrt(2) if i != j else 1)) for i, j in zip(*np.triu_indices(k))], key=lambda c: -abs(c[4]))
    # held-out class gain, 2-fold, earlier -> later class
    idx = list(range(len(names))); rnd.shuffle(idx); half = len(idx) // 2
    A = [names[i] for i in idx[:half]]; B = [names[i] for i in idx[half:]]
    def events(ns): return [(a, b) for nm in ns for a, b in pairs_in(nm['seq'], nm['pos'], nonadj_only)]
    def gain(trn, tst, c):
        pairs = [(c[a], c[b]) for a, b in trn if a in c and b in c]; unis = [c[b] for a, b in trn if b in c]
        big = collections.defaultdict(collections.Counter)
        for a, b in pairs: big[a][b] += 1
        uni = collections.Counter(unis); Nn = sum(uni.values()); Tt = len(uni); lam0 = Nn / (Nn + Tt)
        def pu(b): return lam0 * uni.get(b, 0) / Nn + (1 - lam0) / k
        g = []
        for a, b in tst:
            if a in c and b in c:
                cc = big.get(c[a]);
                if not cc: g.append(0.0); continue
                n = sum(cc.values()); t = len(cc); lam = n / (n + t)
                g.append(math.log2(lam * cc.get(c[b], 0) / n + (1 - lam) * pu(c[b])) - math.log2(pu(c[b])))
        return g
    eA, eB = events(A), events(B)
    g = gain(eA, eB, cls) + gain(eB, eA, cls)
    gm = float(np.mean(g)); gb = [np.mean(np.random.default_rng(i).choice(g, len(g))) for i in range(300)]
    keys = list(cls); labs = [cls[w] for w in keys]; rp = []
    for _ in range(NRP):
        rnd.shuffle(labs); rc = dict(zip(keys, labs)); rp.append(np.mean(gain(eA, eB, rc) + gain(eB, eA, rc)))
    P(f'\n## {label}: {len(names)} names, {int(tot)} classed pairs ({"non-adjacent only" if nonadj_only else "all pairs"}), {k} classes {dict(collections.Counter(cls.values()).most_common())}')
    P(f'   Cramer V vs permutation expectation: {v:.3f} (null tables {np.mean(vn):.3f} [{np.percentile(vn, 2.5):.3f}, {np.percentile(vn, 97.5):.3f}], P = {pval(v, vn):.3f})')
    P(f'   same-class pairs {same:.0f} / {tot:.0f} = {same / tot:.3f}; expected {np.mean(samen) / tot:.3f}; O/E {same / np.mean(samen):.2f}; P(hi) = {pval(same, samen):.3f} P(lo) = {pval(same, samen, "lo"):.3f}')
    P(f'   class-pair gain over unigram (2-fold held-out): {gm:+.4f} bits/token [{np.percentile(gb, 2.5):+.4f}, {np.percentile(gb, 97.5):+.4f}] on {len(g)} tokens; random partitions {np.mean(rp):+.4f} [{np.percentile(rp, 2.5):+.4f}, {np.percentile(rp, 97.5):+.4f}] P = {pval(gm, rp):.3f}')
    P(f'   symmetry corr(z(A->B), z(B->A)) = {r:+.2f} over {int(m.sum())} cell pairs')
    P('   strongest symmetric cells: ' + '; '.join(f'{a}x{b} O {o:.0f} E {e:.0f} z {z:+.1f}' for a, b, o, e, z in cells[:8]))
    return dict(label=label, n=len(names), pairs=int(tot), k=k, V=v, V_null=float(np.mean(vn)), V_P=pval(v, vn), same=same / tot, same_E=float(np.mean(samen)) / tot,
                same_OE=same / float(np.mean(samen)), same_Phi=pval(same, samen), same_Plo=pval(same, samen, 'lo'), gain=gm, gain_ci=[float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
                gain_rand=float(np.mean(rp)), gain_P=pval(gm, rp), sym_r=r, cells=[(a, b, float(o), float(e), float(z)) for a, b, o, e, z in cells[:8]])

RES = []
# ---------------------------------------------------------------- Indus
T = parse_all(load_wells(LV))
CJ = json.load(open(DARK + f'loop66_classes_{LV}.json'))
DESC = {int(w): c for w, c in CJ['DESC'].items()}; WB = {int(w): c for w, c in CJ['WBLOCK'].items()}
ind = [dict(seq=t['seq'], pos=t['midpos'], strat=(t['site'], t['ot'], len(t['midpos']))) for t in T if len(t['midpos']) >= 2]
RES.append(analyse('Indus middle, DESC classes, non-adjacent pairs', ind, DESC, None, True))
RES.append(analyse('Indus middle, DESC classes, all pairs', ind, DESC, None, False))
RES.append(analyse('Indus middle, WBLOCK classes, all pairs', ind, WB, None, False))
# ---------------------------------------------------------------- Chinese / Japanese
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
def load_names(fn, cap=None, minlen=2, skip_first=False):
    rows = [json.loads(l) for l in open(DARK + 'loop63_corpora/' + fn)]
    rows = [r for r in rows if len(r['seq']) >= minlen]
    rnd.shuffle(rows); rows = rows[:(cap or CAP)]
    out = []
    for r in rows:
        pos = list(range(1 if skip_first else 0, len(r['seq'])))
        out.append(dict(seq=r['seq'], pos=pos, strat=(r.get('g', 'U'), len(r['seq']))))
    return out
def han_cls(names):
    els = set(x for nm in names for p in nm['pos'] for x in [nm['seq'][p]])
    return {e: han_class(e) for e in els if han_class(e) is not None}
cn = load_names('cn_given.jsonl'); RES.append(analyse('Chinese given names (2 characters), radical classes', cn, han_cls(cn), None, False))
cna = load_names('cn_ancient.jsonl', minlen=3, skip_first=True)
RES.append(analyse('Chinese historical names (given part, >= 2 chars after surname), radical classes, non-adjacent pairs', cna, han_cls(cna), None, True))
RES.append(analyse('Chinese historical names, radical classes, all pairs', cna, han_cls(cna), None, False))
jp = load_names('jp_given.jsonl'); RES.append(analyse('Japanese given names (kanji), radical classes, all pairs', jp, han_cls(jp), None, False))
# ---------------------------------------------------------------- Ur III
ur = []
for l in open(DARK + 'loop56_corpora/ur3_names_elem.jsonl'):
    s = json.loads(l)['seq']
    if len(s) < 2 or '($' in s[0] or any('x' == e for e in s): continue
    ur.append(dict(seq=s, pos=list(range(len(s))), strat=(len(s),)))
urc = {e: ur3_class(e) for nm in ur for e in nm['seq']}
RES.append(analyse('Ur III owner names (greedy-chunk elements), theophoric/kin/predicate/other, all pairs', ur, urc, None, False))
ur3 = [nm for nm in ur if len(nm['seq']) >= 3]
if len(ur3) > 200: RES.append(analyse('Ur III owner names (>= 3 elements), non-adjacent pairs', ur3, urc, None, True))

json.dump(RES, open(DARK + f'loop66_c3_{LV}.json', 'w'))
open(DARK + f'loop66_c3_{LV}.txt', 'w').write('\n'.join(out) + '\n')
P(f'done {time.time() - T0:.0f}s')
