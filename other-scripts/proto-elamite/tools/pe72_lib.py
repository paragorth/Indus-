"""pe72 CRACK ATTEMPT v2: shared library.

The pe59 working reading (pe59-R1) updated with everything that survived pe60-pe70 and nothing else:
  dropped  : every guess the pe66 kill sweep killed (M388 capacity opener, name-frame elements, 48/75 per unit,
             team rule (pe69), M056 = hidden M288 (pe69)), plus v1 C items never confirmed (M136 capacity opener).
  added    : the per-head 60 rule scoped as pe69 found it (bare 'M288 n' after a PERSON-final count line),
             the pe63 dossiers (headers and number system fixed inside a series), the pe70 sealed document type
             (as a tablet-type prior and a gloss), the pe52/pe58 class-sign weights that survive a re-test on the
             TRAINING half only, and the four clean pe66 survivors (Yahya lexicon C+, [M327+M342] header slot B,
             grain-office signs C+, M005~a header slot C+).
Decoder: the pe59 decoder (C1 system, C2 role/numeral checks, C3 totals) plus new falsifiable checks
(header slot, dossier header/system, weight direction, scoped per-head rule).  Every element carries a grade.
Same tablets, same split ('pe59' hash split) as pe59, so v1 and v2 are scored on the same held-out half.
"""
import os, sys, json, math, random, hashlib, copy, re
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe59_lib as P  # noqa: E402
from pe59_models import Grammar  # noqa: E402
from common import load as load_raw, is_sign  # noqa: E402

PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe72_ckpt')
os.makedirs(CK, exist_ok=True)

sha = P.sha
seed = P.seed

# ------------------------------------------------------------------ data
DOSSIERS = {  # pe63 (B): template series found on the whole corpus
    'D1': ['P008723', 'P008724', 'P008725', 'P008726', 'P008727', 'P008728', 'P008729', 'P008730', 'P008731'],
    'D2': ['P008796', 'P008797', 'P008798', 'P008799', 'P008800', 'P008801', 'P008802'],
    'D3': ['P009190', 'P009211', 'P009220', 'P009237', 'P009238', 'P009286', 'P009309'],
    'D4': ['P009056', 'P009137', 'P009138', 'P009140'],
    'D1b': ['P008717', 'P008718', 'P008719', 'P008720'],
    'D5': ['P008790', 'P008791', 'P008792', 'P008794'],
    'D6': ['P008100', 'P008125', 'P008193', 'P368479'],
    'D7': ['P393079', 'P393080', 'P393082'],
}
TEMPLATE_SEALS = {'PES0329': 'griffin seal template series (|M153+M342|, M340)',
                  'PES0334': 'boat seal template series (|M153+X|, M054)'}


def pe_tablets():
    """pe59 tablets + raw (variant-bearing) signs per line, sealed flag and seal ids."""
    fn = os.path.join(CK, 'pe_tabs72.json')
    if os.path.exists(fn):
        T = json.load(open(fn))
        prov = {t['id']: t['provenience'] for t in load_raw()}
        for t in T:   # modern site (pe59 'site' keeps only the ancient name, outposts are 'uncertain')
            m = re.search(r'mod\. ([^)]*)\)', prov.get(t['id'], ''))
            t['msite'] = m.group(1) if m else t['site']
        return T
    T = P.build_pe()
    raw = {t['id']: t for t in load_raw()}
    import pe70_common as C70
    seal = {r['id']: r for r in C70.get('pe')}
    for t in T:
        rl = raw[t['id']]['lines']
        assert len(rl) == len(t['lines'])
        for l, r in zip(t['lines'], rl):
            l['rsigns'] = [s for s in r['signs'] if is_sign(s)]
        s = seal.get(t['id'])
        t['sealed'] = bool(s and s['sealed'])
        t['seals'] = s['seals'] if s else []
    json.dump(T, open(fn, 'w'))
    return pe_tablets()


def pc_tablets():
    fn = os.path.join(CK, 'pc_tabs72.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = P.build_pc()
    import pe70_common as C70
    seal = {r['id']: r for r in C70.get('pc')}
    for t in T:
        for l in t['lines']:
            l['rsigns'] = list(l['signs'])
        s = seal.get(t['id'])
        t['sealed'] = bool(s and s['sealed'])
        t['seals'] = s['seals'] if s else []
    json.dump(T, open(fn, 'w'))
    return T


def pc_dossiers(T):
    """Proto-cuneiform analogue of the pe63 dossiers (calibration only): runs of tablets with P-numbers within 3
    of each other, the same header line and sign-set Jaccard >= 0.5; groups of >= 3."""
    def num(t):
        return int(t['id'][1:])
    def hdr(t):
        h = [l for l in t['lines'] if l['role'] == 'H' and l['signs']]
        return tuple(h[0]['signs']) if h else None
    def ss(t):
        return set(s for l in t['lines'] for s in l['signs'])
    TT = sorted(T, key=num)
    groups, cur = [], []
    for t in TT:
        if cur and num(t) - num(cur[-1]) <= 3 and hdr(t) is not None and hdr(t) == hdr(cur[-1]):
            a, b = ss(t), ss(cur[-1])
            if len(a & b) / max(1, len(a | b)) >= 0.5:
                cur.append(t); continue
        if len(cur) >= 3:
            groups.append([x['id'] for x in cur])
        cur = [t]
    if len(cur) >= 3:
        groups.append([x['id'] for x in cur])
    return {'PCD%d' % i: g for i, g in enumerate(groups)}


# ------------------------------------------------------------------ weights: re-test on the training half only
def entry_logvals(t, capmap, cntmap):
    tau = P.tablet_type(t) or 'CNTT'
    out = []
    for l in t['lines']:
        if l['role'] != 'E' or not l['numclean']:
            continue
        c = P.ncls(l['nums'])
        if c == 'CAP' or (c == 'AMB' and tau == 'CAPT'):
            v, grp = P.value(l['nums'], capmap), 'cap'
        elif c in ('AMB', 'FRAC', 'BIS'):
            v, grp = P.value(l['nums'], cntmap), 'cnt'
        else:
            continue
        if v and v > 0:
            out.append((grp, math.log(float(v)), set(l['signs'])))
    return out


def weight_survival(train, cands, capmap, cntmap, nperm=400, rng=None):
    """For each candidate sign with a frozen direction, mean within-tablet log shift (entry minus mean of its
    same-system siblings) on TRAINING tablets; one-sided within-tablet permutation p in the frozen direction."""
    rng = rng or np.random.default_rng(seed('pe72w'))
    groups = []
    for t in train:
        ev = entry_logvals(t, capmap, cntmap)
        for g in ('cap', 'cnt'):
            e = [(v, s) for gg, v, s in ev if gg == g]
            if len(e) >= 2:
                groups.append(e)
    gsh = []
    for e in groups:
        vals = np.array([v for v, _ in e])
        gsh.append(vals - (vals.sum() - vals) / (len(vals) - 1))
    res = {}
    for s, d in cands.items():
        occ = [(gi, j) for gi, e in enumerate(groups) for j, (_, ss) in enumerate(e) if s in ss]
        n = len(occ)
        if n < 5:
            res[s] = {'n': n, 'mean': None, 'p': None, 'survives': False, 'dir': d}
            continue
        obs = float(np.mean([gsh[gi][j] for gi, j in occ]))
        L = np.array([len(gsh[gi]) for gi, _ in occ])
        flat = np.concatenate([gsh[gi] for gi, _ in occ])
        off = np.concatenate([[0], np.cumsum(L)[:-1]])
        dr = (rng.random((nperm, n)) * L).astype(int) + off
        m = flat[dr].mean(1)
        p = (int((m * d >= obs * d).sum()) + 1) / (nperm + 1)
        res[s] = {'n': n, 'mean': round(obs, 3), 'p': round(p, 4), 'dir': d,
                  'survives': bool(obs * d > 0 and p < 0.05)}
    return res


# ------------------------------------------------------------------ the reading v2
def derive_weights(train, capmap, cntmap, min_n=15, nperm=300, alpha=0.01):
    """Proto-cuneiform calibration: weights discovered AND tested on the training half (direction = sign of the
    observed mean shift; kept if the one-sided permutation p < alpha)."""
    f = Counter(s for t in train for l in t['lines'] if l['role'] == 'E' and l['numclean'] for s in set(l['signs']))
    cands = {s: 1 for s, n in f.items() if n >= min_n}
    r1 = weight_survival(train, cands, capmap, cntmap, nperm=1)
    cands = {s: (1 if v['mean'] > 0 else -1) for s, v in r1.items() if v['mean'] is not None and abs(v['mean']) > 0.05}
    r = weight_survival(train, cands, capmap, cntmap, nperm=nperm)
    return {s: v['dir'] for s, v in r.items() if v['survives'] and v['p'] < alpha}, r


def build_reading2(train_ids_sha, weight_res):
    R = copy.deepcopy(P.READING)
    R['name'] = 'pe72-R2 working reading of the Proto-Elamite administrative system (v2 of pe59-R1)'
    R['header'] = {
        'OPEN_GEN': {'signs': ['M157'], 'grade': 'A', 'src': 'test d; pe66 F13 (M157 is the header sign)', 'gloss': 'general opener'},
        'OPEN_327': {'signs': ['|M327+M342|', 'M327'], 'prefix_match': '|M327+', 'grade': 'B',
                     'src': 'pe66 F9 (header SLOT B; 86-95% in header line, decoys 0/40); meaning "office" C, not used',
                     'gloss': 'M327-family header-slot sign'},
        'OPEN_005a': {'raw_signs': ['M005~a'], 'grade': 'C+', 'src': 'pe66 F16 (16/52 header vs M005 0/18; > 42/42 decoys)',
                      'gloss': 'M005~a header-slot variant'},
        'OPEN_OTHER': {'signs': ['|M377+M320+M377|', 'M305', 'M247'], 'grade': 'B', 'src': 'test d', 'gloss': 'other opener'},
    }
    R['header_slot_check'] = {'grade': 'B (|M327+M342|) / C+ (M005~a)',
                              'rule': '|M327+M342| and M005~a stand in the header line; one in an entry line is a conflict'}
    R['roles']['MEASURED']['signs'] = R['roles']['MEASURED']['signs'] + ['M286', 'M248']
    R['roles']['MEASURED']['grain_office_Cplus'] = ['M081', 'M265', 'M266', 'M296', 'M112', 'M286', 'M248']
    R['roles']['MEASURED']['src'] += '; pe66 F10 grain-office signs C+ (35-61% capacity, decoys 1-3/30)'
    R['roles']['ALLOT'] = {'grade': 'B-', 'src': 'pe27, pe28, pe63 D3 (6/6), pe66 F6 (60 is the top real rate), pe69 F2',
                           'signs': ['M288'],
                           'rule': "a bare 'M288 n' line directly after a count line ending in a PERSON-class sign holds "
                                   "60 N39C = 2(N39B) 1(N24) (sometimes 120 N39C) per unit of that line (34/49 vs 5.5 by "
                                   "chance); 'signs + M288' lines follow no per-unit rule (6/52 vs 5.1)",
                           'gloss': 'standard allotment line'}
    R['roles'].pop('NAMEFRAME', None)
    R['weights'] = {'grade': 'B', 'src': 'pe52 (both-mode weights) and pe58 factors, kept only if the frozen direction '
                    'replicates on the pe59 TRAINING half (within-tablet permutation p < 0.05)',
                    'retest': weight_res,
                    'signs': {s: v['dir'] for s, v in weight_res.items() if v['survives']},
                    'check': 'an entry with a weighted sign lies above (+) / below (-) the mean of its same-system siblings'}
    R['sealed'] = {'grade': 'B', 'src': 'pe70 F1 (cross-volume AUC 0.667 vs 0.524)',
                   'rule': 'sealed tablets are a distinct document type: short, headed (M157), with M288 lines, fewer '
                           'capacity lines; used as a tablet-type prior feature (fitted on training tablets)',
                   'meaning': 'signed-for per-head disbursement (C, gloss only, not scored)',
                   'template_seals': TEMPLATE_SEALS}
    R['dossiers'] = {'grade': 'B', 'src': 'pe63 (8 series, 42 tablets; planted 30/30)', 'members': DOSSIERS,
                     'rule': 'inside a series the header and the number system stay fixed; identifiers and counts vary; '
                             'predictions use the series siblings in the TRAINING half only'}
    R['outposts'] = {'grade': 'C+', 'src': 'pe48, pe66 F4 (held-out half selects exactly these 4, p 0.024; decoys 1/30)',
                     'Yahya': ['M056', 'M044', 'M219', 'M136']}
    R['dropped'] = {
        'OPEN_CAP M388': 'killed pe66 F1 (M388 tablets close as counts)',
        'OPEN_CAP M136': 'v1 C, never confirmed (stickiness killed pe66 F17); M136 kept only as a Yahya term (C+)',
        'OPEN_OTHER M005 (plain)': 'replaced by M005~a (pe66 F16: plain M005 0/18 in header)',
        'NAMEFRAME': 'killed pe66 F13',
        'M288 120k as a separate rate': 'killed pe59 F6; 120 kept only as the pe69 occasional variant of the scoped rule',
        'team rule': 'killed pe69 F1', '48 / 75 per unit': 'killed pe66 F6-F7 (48 survives only as a dossier-bound C, not in the reading)',
        'M056 = hidden M288': 'killed pe69 F3', 'M297 running sums': 'killed pe66 F3',
        'commodity candidates pe61': 'killed pe66 F11'}
    R['train_ids_sha'] = train_ids_sha
    return R


def roles_v2(R2):
    base_roles = P.pe_roles()
    S = base_roles['sets']
    S['MEASURED'] = set(R2['roles']['MEASURED']['signs'])
    S['H_GEN'] = set(R2['header']['OPEN_GEN']['signs'])
    S['H_327'] = set(R2['header']['OPEN_327']['signs'])
    S['H_CAP'] = set()
    S['H_OTH'] = set(R2['header']['OPEN_OTHER']['signs'])
    S['H_RAW'] = set(R2['header']['OPEN_005a']['raw_signs'])
    S['HSLOT'] = {'|M327+M342|'}
    S['OUT_Yahya'] = set(R2['outposts']['Yahya'])
    S['GRAIN_OFFICE'] = set(R2['roles']['MEASURED']['grain_office_Cplus'])
    S['ALLOT_PREV'] = set(R2['roles']['PERSON']['signs'])
    return {'sets': S, 'weights': base_roles['weights'], 'h327_prefix': '|M327+', 'allot_sign': 'M288',
            'wdir': dict(R2['weights']['signs']), 'version': 'v2', 'dossiers': DOSSIERS}


def roles_v1():
    r = P.pe_roles()
    r['version'] = 'v1'
    return r


# ------------------------------------------------------------------ proto-cuneiform calibration readings (answer key)
PC_EXT = {  # true Sumerological readings ADDED in v2 (calibration only)
    'COUNTED': {'SUHUR', 'MUSZEN', 'NUNUZ', 'GADA', 'GU4', 'SZAH2', 'MUNUS'},
    'ALLOT': set(),     # no proto-cuneiform analogue of the scoped per-head rule is known
    'H_GEN': set(),     # no sign is known to be positionally a header-slot sign (EN/SANGA tried: they stand in
                        # entries as often as in headers, so a header-slot claim for them would be a WRONG reading)
}


def pc_roles_v1():
    r = P.pc_roles(); r['version'] = 'v1'; return r


def pc_roles_v2(wdir, doss):
    r = P.pc_roles()
    S = r['sets']
    S['COUNTED'] = S['COUNTED'] | PC_EXT['COUNTED']
    S['ALLOT'] = S['ALLOT'] | PC_EXT['ALLOT']
    S['H_GEN'] = set(PC_EXT['H_GEN'])
    S['HSLOT'] = set()
    S['H_RAW'] = set()
    S['GRAIN_OFFICE'] = set()
    S['ALLOT_PREV'] = set()   # no scoped per-head rule known for proto-cuneiform
    r.update({'wdir': dict(wdir), 'version': 'v2', 'dossiers': doss})
    return r


# ------------------------------------------------------------------ grammar with the sealed prior
class Grammar2(Grammar):
    """pe59 grammar; for v2 the header-role feature of the tablet-type classifier also carries the sealed flag
    (fitted on training tablets), and dossier siblings in the training half fix the tablet type."""
    def tau_features(self, t, upto=None, with_nums=True):
        h, lr, nc, site = super().tau_features(t, upto, with_nums)
        if self.roles.get('version') == 'v2':
            for l in t['lines']:
                if l['role'] == 'H' and l.get('rsigns'):
                    if l['rsigns'][0] in self.roles['sets'].get('H_RAW', ()):
                        h = 'H_OTH'
                    break
            if t.get('sealed_x', t.get('sealed')):
                h = h + '|S'
        return h, lr, nc, site


def fit_grammar(train, roles, maps, corpus):
    return Grammar2(train, roles, maps, corpus=corpus)


def dossier_context(train, roles):
    """For each dossier: majority header first sign and tablet type over TRAINING members."""
    trid = {t['id']: t for t in train}
    ctx = {}
    for d, mem in roles.get('dossiers', {}).items():
        hs, ts = Counter(), Counter()
        for m in mem:
            t = trid.get(m)
            if not t:
                continue
            h = [l for l in t['lines'] if l['role'] == 'H' and l['signs']]
            if h:
                hs[h[0]['signs'][0]] += 1
            tt = P.tablet_type(t)
            if tt:
                ts[tt] += 1
        ctx[d] = {'header': hs.most_common(1)[0][0] if hs else None, 'type': ts.most_common(1)[0][0] if ts else None,
                  'n_train': sum(1 for m in mem if m in trid)}
    member = {m: d for d, mem in roles.get('dossiers', {}).items() for m in mem}
    return ctx, member


# ------------------------------------------------------------------ decoder
ROLE_GLOSS = {'MEASURED': 'measured (grain-class) commodity', 'COUNTED': 'counted item', 'PERSON': 'person-unit',
              'ALLOT': 'standard allotment', 'FRACLINE': 'fraction line', 'MEASURED_IN': 'string with a measured-class sign',
              'COUNTED_IN': 'string with a counted-class sign', 'OTHER': 'designation (unclassified string)',
              'BARE': 'bare number'}
ROLE_GRADE = {'MEASURED': 'A/B', 'COUNTED': 'A/B', 'PERSON': 'B', 'ALLOT': 'B', 'FRACLINE': 'B', 'MEASURED_IN': 'B-',
              'COUNTED_IN': 'B-', 'OTHER': 'B', 'BARE': 'A'}
HDR_GLOSS = {'H_GEN': ('general opener M157', 'A'), 'H_327': ('M327-family header-slot sign', 'B'),
             'H_CAP': ('capacity-account opener', 'C'), 'H_OTH': ('other opener', 'B'),
             'H_X': ('unclassified header', '-'), 'NONE': ('no header', '-')}


class Ctx:
    def __init__(self, corpus, roles, maps, train, sealed_override=None, dossier_override=None):
        self.corpus = corpus
        self.roles = roles
        if sealed_override is not None:
            train = [dict(t, sealed_x=sealed_override.get(t['id'], False)) for t in train]
        self.g = fit_grammar(train, roles, maps, corpus)
        self.capmap, self.cntmaps = maps
        self.cnt_main = 'sex2' if corpus == 'PE' else 'S'
        r2 = dict(roles)
        if dossier_override is not None:
            r2['dossiers'] = dossier_override
        self.dctx, self.dmember = dossier_context(train, r2) if roles.get('version') == 'v2' else ({}, {})
        self.sealed_override = sealed_override


def _val(nums, m):
    return P.value(nums, m)


def fmt_q(nums, tau, C):
    c = P.ncls(nums)
    if c == 'CAP' or (c == 'AMB' and tau == 'CAPT'):
        v = _val(nums, C.capmap)
        if v is None:
            return '?', None
        if C.corpus == 'PE':
            return '%s N39C (%.3g-%.3g l, litres C)' % (v, float(v) * 0.6, float(v) * 0.8), v
        return '%s N01-grain' % v, v
    v = _val(nums, C.cntmaps[C.cnt_main])
    if v is None:
        return '?', None
    return '%s units' % v, v


def decode(t, C, numerals_from=None, want_gloss=False):
    """Decode one tablet. Returns metrics dict (+ gloss with grades if want_gloss)."""
    v2 = C.roles.get('version') == 'v2'
    lines = [dict(l) for l in t['lines']]
    if numerals_from is not None:
        E = [i for i, l in enumerate(lines) if l['role'] in ('E', 'T')]
        for i, nums in zip(E, numerals_from):
            lines[i]['nums'], lines[i]['numclean'] = nums[0], nums[1]
    tt = dict(t); tt['lines'] = lines
    if C.sealed_override is not None:
        tt['sealed_x'] = C.sealed_override.get(t['id'], False)
    sealed = tt.get('sealed_x', t.get('sealed', False))
    ents = [l for l in lines if l['role'] == 'E']
    clean = [l for l in ents if l['numclean']]
    if len(clean) < 2:
        return None
    S = C.roles['sets']
    ptau = C.g.tau_post(tt, with_nums=False)
    dos = C.dmember.get(t['id']) if v2 else None
    dinfo = C.dctx.get(dos) if dos else None
    dos_type_used = False
    if v2 and dinfo and dinfo['type']:
        ptau = 0.95 if dinfo['type'] == 'CAPT' else 0.05
        dos_type_used = True
    tau = 'CAPT' if ptau > 0.5 else 'CNTT'
    G = []   # (text, grade, status)
    tchecks = []  # tablet-level (name, ok)
    h = [l for l in lines if l['role'] == 'H']
    hr = P.header_role(h[0]['signs'][0], C.roles) if h and h[0]['signs'] else 'NONE'
    if v2 and h and h[0].get('rsigns') and h[0]['rsigns'][0] in S.get('H_RAW', ()):
        G.append(('header [%s]: M005~a header-slot variant' % ' '.join(h[0]['rsigns']), 'C+', 'PASS'))
        tchecks.append(('hslot', True))
    else:
        txt, gr = HDR_GLOSS[hr]
        st = '-'
        if v2 and h and h[0]['signs'] and h[0]['signs'][0] in S.get('HSLOT', ()):
            st = 'PASS'; tchecks.append(('hslot', True))
        G.append(('header%s: %s' % ((' [' + ' '.join(h[0]['signs']) + ']') if h and h[0]['signs'] else '', txt), gr, st))
    if v2:
        bad = [l for l in ents if any(s in S.get('HSLOT', ()) for s in l['signs'])
               or any(s in S.get('H_RAW', ()) for s in l.get('rsigns', []))]
        if bad:
            tchecks.append(('hslot', False))
            G.append(('header-slot sign inside an entry line', 'B/C+', 'FAIL'))
    if any(l['role'] == 'G' for l in lines):
        G.append(('edge tag ' + ' '.join('%d(%s)' % (n, c) for l in lines if l['role'] == 'G' for n, c in l['nums'])
                  + ': document-type tag, never a sum', 'B', '-'))
    if v2 and sealed:
        sl = [s for s in t.get('seals', []) if s in TEMPLATE_SEALS]
        G.append(('sealed tablet: distinct short, headed M288 document type%s' %
                  ('; ' + '; '.join(TEMPLATE_SEALS[s] for s in sl) if sl else ''), 'B', '-'))
        G.append(('meaning of the seal: someone signed for a per-head disbursement', 'C', '-'))
    if v2 and dinfo:
        txt = 'template series %s (%d training siblings)' % (dos, dinfo['n_train'])
        G.append((txt, 'B', '-'))
        if dinfo['header'] and h and h[0]['signs']:
            ok = h[0]['signs'][0] == dinfo['header']
            tchecks.append(('dossier_header', ok))
            G.append(('series header predicted %s' % dinfo['header'], 'B', 'PASS' if ok else 'FAIL'))
    obs = P.tablet_type(tt)
    G.append(('document: %s account (P=%.2f%s)' % ('capacity' if tau == 'CAPT' else 'count',
                                                   ptau if tau == 'CAPT' else 1 - ptau,
                                                   ', from the series siblings' if dos_type_used else ', from signs alone'),
              'B', '-'))
    # C1 system
    c1ok = True
    for l in clean:
        c = P.ncls(l['nums'])
        if tau == 'CAPT':
            if c in ('FRAC', 'BIS'):
                c1ok = False
            if c == 'AMB':
                d = Counter()
                for n, cc in l['nums']:
                    d[cc] += n
                if d['N01'] >= 6:
                    c1ok = False
        elif c == 'CAP':
            c1ok = False
    tchecks.append(('system', c1ok))
    # weight-check context: log values of clean entries by system group
    ev = []
    for l in lines:
        if l['role'] == 'E' and l['numclean']:
            c = P.ncls(l['nums'])
            if c == 'CAP' or (c == 'AMB' and tau == 'CAPT'):
                v, g = _val(l['nums'], C.capmap), 'cap'
            elif c in ('AMB', 'FRAC', 'BIS'):
                v, g = _val(l['nums'], C.cntmaps[C.cnt_main]), 'cnt'
            else:
                v, g = None, None
            ev.append((id(l), g, math.log(float(v)) if v and v > 0 else None))
    evd = {k: (g, v) for k, g, v in ev}
    wdir = C.roles.get('wdir', {}) if v2 else {}
    LS = []  # per clean entry: 'PASS'/'FAIL'/'-'
    comp = Counter()
    prev = None
    k = 0
    yahya = 'Yahya' in t.get('msite', '')
    for l in lines:
        if l['role'] != 'E':
            continue
        k += 1
        sg = l['signs']
        role = P.line_role(sg, C.roles)
        pre = sg[0] if (len(sg) > 1 and sg[0] in S['PREFIX']) else None
        cls_s = sg[-1] if (sg and role in ('MEASURED', 'COUNTED', 'PERSON', 'ALLOT', 'FRACLINE')) else None
        mid = sg[(1 if pre else 0):(-1 if cls_s else len(sg))]
        q, v = fmt_q(l['nums'], tau, C) if l['numclean'] else ('[damaged]', None)
        parts, notes = [], []
        if pre:
            parts.append('qualifier %s (A)' % pre)
        if mid:
            parts.append('designation [%s] (B: tablet-pool string)' % ' '.join(mid))
        oks = []
        if l['numclean']:
            c = P.ncls(l['nums'])
            ok = None
            if role == 'MEASURED':
                ok = c == 'CAP' or (c == 'AMB' and tau == 'CAPT')
            elif role in ('COUNTED', 'PERSON'):
                ok = c != 'CAP'
                if role == 'PERSON' and ok and v is not None and c == 'AMB' and tau == 'CNTT':
                    ok = v <= 60
            elif role == 'FRACLINE':
                ok = c != 'CAP'
            elif role == 'ALLOT' and prev is not None and prev['numclean'] and P.ncls(prev['nums']) == 'AMB':
                scoped = (not v2) or (not S.get('ALLOT_PREV')) or (
                    sg == [C.roles['allot_sign']] and prev['signs'] and prev['signs'][-1] in S['ALLOT_PREV'])
                if scoped:
                    ku = _val(prev['nums'], C.cntmaps[C.cnt_main])
                    vv = _val(l['nums'], C.capmap)
                    if ku and vv is not None and ku <= 60:
                        ok = vv in (60 * ku, 120 * ku)
                        notes.append(('allotment for %s units: %s' % (ku, '= 60/120 N39C per unit' if ok else 'off-rate'),
                                      'B-' if v2 else 'B'))
            if ok is not None:
                oks.append(ok)
                comp[('allot_' if role == 'ALLOT' else 'role_') + ('ok' if ok else 'bad')] += 1
            if wdir and evd.get(id(l), (None, None))[1] is not None:
                g0, lv = evd[id(l)]
                sib = [vv for kk, (gg, vv) in evd.items() if kk != id(l) and gg == g0 and vv is not None]
                ws = [(s, wdir[s]) for s in set(sg) if s in wdir]
                if len(sib) >= 1 and ws:
                    d = sum(dd for _, dd in ws)
                    diff = lv - float(np.mean(sib))
                    if d != 0 and abs(diff) > 1e-9:
                        okw = (diff > 0) == (d > 0)
                        oks.append(okw)
                        comp['w_' + ('ok' if okw else 'bad')] += 1
                        notes.append(('weight sign %s: entry %s its siblings as predicted' %
                                      (','.join(s for s, _ in ws), 'above' if diff > 0 else 'below') if okw else
                                      'weight sign %s: entry on the wrong side of its siblings' % ','.join(s for s, _ in ws), 'B'))
        st = '-' if not oks else ('PASS' if all(oks) else 'FAIL')
        if l['numclean']:
            LS.append(st)
        clsg = ROLE_GRADE[role]
        if v2 and cls_s and cls_s in S.get('GRAIN_OFFICE', ()):
            clsg = 'C+'
        if v2 and yahya:
            ys = [s for s in sg if s in S.get('OUT_Yahya', ())]
            if ys:
                notes.append(('Yahya-local term %s' % ','.join(ys), 'C+'))
        txt = 'entry %d: %s%s %s of %s%s' % (k, ' + '.join(parts) + (' ' if parts else ''),
                                            'receives' if role == 'ALLOT' else '-', q, ROLE_GLOSS[role],
                                            (' ' + cls_s) if cls_s else '')
        G.append((txt, clsg, st))
        for nt, ng in notes:
            G.append(('   ' + nt, ng, '-'))
        prev = l
    c3 = C.g.closes(tt, tau)
    tot = [l for l in lines if l['role'] == 'T']
    if tot:
        G.append(('total: ' + ('not checkable (damage)' if c3 is None else 'closes in the %s system' %
                               ('capacity' if tau == 'CAPT' else 'count') if c3 else 'does not close'),
                  'A', '-' if c3 is None else ('PASS' if c3 else 'FAIL')))
    else:
        G.append(('total: none written', '-', '-'))
    tfail = any(not ok for _, ok in tchecks)
    npass = LS.count('PASS'); nfail = LS.count('FAIL')
    full = (not tfail) and nfail == 0 and (c3 is not False)
    out = {'id': t['id'], 'full': full, 'strict': full and npass >= 1 and c3 is True,
           'dense': full and npass >= 0.5 * len(LS), 'npass': npass, 'nfail': nfail, 'nlines': len(LS),
           'C1': c1ok, 'C3': c3, 'roles_checked': npass + nfail > 0,
           'tfail': [n for n, ok in tchecks if not ok], 'tpass': [n for n, ok in tchecks if ok and n != 'system'],
           'comp': dict(comp)}
    if want_gloss:
        out['gloss'] = G
    return out


def summarise(R):
    R = [r for r in R if r]
    n = len(R)
    lines = sum(r['nlines'] for r in R)
    return {'n': n, 'full': sum(r['full'] for r in R), 'strict': sum(r['strict'] for r in R),
            'dense': sum(r['dense'] for r in R), 'C1': sum(r['C1'] for r in R),
            'C3_n': sum(r['C3'] is not None for r in R), 'C3': sum(r['C3'] is True for r in R),
            'lines': lines, 'pass': sum(r['npass'] for r in R), 'fail': sum(r['nfail'] for r in R),
            'net': sum(r['npass'] - r['nfail'] for r in R),
            'tab_fail': sum(bool(r['tfail']) for r in R),
            'comp': dict(sum((Counter(r['comp']) for r in R), Counter())
                         + Counter('t_' + x + '_bad' for r in R for x in r['tfail'])
                         + Counter('t_' + x + '_ok' for r in R for x in r['tpass']))}


# ------------------------------------------------------------------ twins
def freq_vocab(T, raw=False):
    key = 'rsigns' if raw else 'signs'
    f = Counter(s for t in T for l in t['lines'] for s in l.get(key, l['signs']))
    return f


def matched(sign, freq, used, rng):
    vocab = sorted(freq, key=lambda s: -freq[s])
    rank = {s: i for i, s in enumerate(vocab)}
    r = rank.get(sign, len(vocab) - 1)
    for d in range(len(vocab)):
        for j in (r + d, r - d):
            if 0 <= j < len(vocab):
                c = vocab[j]
                if c not in used and rng.random() < 0.5:
                    used.add(c); return c
    return vocab[rng.randrange(len(vocab))]


def shuffle_sealed(T, rng):
    """sealed flags permuted among tablets within line-count bands."""
    def band(t):
        n = len(t['lines'])
        return 0 if n <= 3 else 1 if n <= 7 else 2 if n <= 15 else 3
    out = {}
    by = defaultdict(list)
    for t in T:
        by[band(t)].append(t)
    for b, ts in by.items():
        fl = [bool(t.get('sealed')) for t in ts]
        rng.shuffle(fl)
        for t, f in zip(ts, fl):
            out[t['id']] = f
    return out


def random_dossiers(T, doss, rng):
    ids = [t['id'] for t in T]
    used = set()
    out = {}
    for d, mem in doss.items():
        pick = []
        while len(pick) < len(mem):
            c = ids[rng.randrange(len(ids))]
            if c not in used:
                used.add(c); pick.append(c)
        out[d] = pick
    return out


def twin_roles(roles, T, rng, scope='all', v1roles=None):
    """scope 'all': every role set, header-slot sign, weight sign, the ALLOT_PREV condition, sealed flags and dossier
    memberships replaced by frequency-matched random ones.  scope 'ext': the v1 part kept TRUE, only the v2
    additions replaced (new MEASURED signs, header-slot signs, weights, scoped ALLOT condition, sealed, dossiers)."""
    freq = freq_vocab(T)
    rfreq = freq_vocab(T, raw=True)
    used = set()
    S = roles['sets']
    new = copy.deepcopy(roles)
    NS = new['sets']
    if scope == 'all':
        sh = P.shuffle_roles({'sets': {k: v for k, v in S.items() if k in ('MEASURED', 'COUNTED', 'FRACLINE', 'ALLOT',
                                                                            'PERSON', 'PREFIX', 'H_GEN', 'H_327',
                                                                            'H_CAP', 'H_OTH', 'OUT_Yahya')},
                              'weights': roles['weights'], 'h327_prefix': roles['h327_prefix'],
                              'allot_sign': roles['allot_sign']}, T, rng)
        for k2, v in sh['sets'].items():
            NS[k2] = v
        new['allot_sign'] = sh['allot_sign']; new['h327_prefix'] = None; new['weights'] = sh['weights']
        used = set().union(*sh['sets'].values())
        for k2 in ('HSLOT', 'GRAIN_OFFICE', 'ALLOT_PREV'):
            NS[k2] = {matched(s, freq, used, rng) for s in S.get(k2, ())}
        if 'GRAIN_OFFICE' in NS:
            NS['MEASURED'] |= NS['GRAIN_OFFICE']
    else:
        used = set().union(*[v for k2, v in S.items()])
        v1S = v1roles['sets']
        added = S['MEASURED'] - v1S['MEASURED']
        NS['MEASURED'] = set(v1S['MEASURED']) | {matched(s, freq, used, rng) for s in added}
        NS['GRAIN_OFFICE'] = {s for s in NS['MEASURED'] - v1S['MEASURED']} | (S.get('GRAIN_OFFICE', set()) & v1S['MEASURED'])
        NS['HSLOT'] = {matched(s, freq, used, rng) for s in S.get('HSLOT', ())}
        addh = S['H_GEN'] - v1S['H_GEN']
        NS['H_GEN'] = set(v1S['H_GEN']) | {matched(s, freq, used, rng) for s in addh}
        if addh:   # proto-cuneiform: the added header signs ARE the header-slot set
            NS['HSLOT'] = NS['H_GEN'] - v1S['H_GEN']
        NS['ALLOT_PREV'] = {matched(s, freq, used, rng) for s in S.get('ALLOT_PREV', ())}
        for k2 in ('COUNTED', 'ALLOT'):
            add = S[k2] - v1S[k2]
            NS[k2] = set(v1S[k2]) | {matched(s, freq, used, rng) for s in add}
    if S.get('H_RAW'):
        rv = [s for s in rfreq if '~' in s and s not in S['H_RAW']]
        rv.sort(key=lambda s: abs(math.log(rfreq[s] + 1) - math.log(rfreq[list(S['H_RAW'])[0]] + 1)))
        NS['H_RAW'] = {rv[rng.randrange(min(20, len(rv)))]}
    if roles.get('wdir'):
        new['wdir'] = {matched(s, freq, used, rng): d for s, d in roles['wdir'].items()}
    new['dossiers'] = random_dossiers(T, roles.get('dossiers', {}), rng)
    return new


def random_units(maps, rng):
    cap, cnt = maps
    codes = sorted(cap, key=lambda c: cap[c])
    v = Fr(1); new = {}
    for c in codes:
        new[c] = v; v *= rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    cnt2 = {}
    for nm, m in cnt.items():
        cs = sorted(m, key=lambda c: m[c]); vv = Fr(1); mm = {}
        for c in cs:
            mm[c] = vv if m[c] >= 1 else m[c]
            if m[c] >= 1:
                vv *= rng.choice([2, 3, 5, 6, 10, 12])
        cnt2[nm] = mm
    return new, cnt2


def transplant(ho, rng):
    byn = defaultdict(list)
    for t in ho:
        E = [(l['nums'], l['numclean']) for l in t['lines'] if l['role'] in ('E', 'T')]
        byn[len(E)].append(E)
    out = []
    for t in ho:
        n = sum(1 for l in t['lines'] if l['role'] in ('E', 'T'))
        pool = byn.get(n, [])
        out.append(pool[rng.randrange(len(pool))] if len(pool) >= 2 else None)
    return out
