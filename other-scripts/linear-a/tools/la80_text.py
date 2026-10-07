"""LA-80 text side: tablet-level text features and the random hypothesis pool (no readings used)."""
import json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L

def tablet_of(i):
    return re.sub(r'(?<=\d)[a-z]$', '', i)

def load_tablets(ids):
    C = {d['id']: d for d in json.load(open(os.path.join(L.DATA, 'corpus_ra.json')))}
    T = collections.OrderedDict()
    for i in ids:
        T.setdefault(tablet_of(i), []).append(C[i])
    return T

def base_stats(docs):
    toks = [t for d in docs for t in d['tokens']]
    words = [t for t in toks if t['t'] == 'word']
    nums = [t for t in toks if t['t'] == 'num']
    logos = [t for t in toks if t['t'] == 'logo']
    signs = [s for w in words for s in w['s']]
    nl = sum(1 for t in toks if t['t'] == 'nl') + len(docs)
    edge = sum(1 for t in toks if set(t.get('fl', [])) & {'edgeL', 'edgeR', 'part'})
    worn = sum(1 for t in toks if 'worn' in t.get('fl', []))
    real = [t for t in toks if t['t'] in ('word', 'num', 'logo', 'unk')]
    return toks, words, nums, logos, signs, nl, edge, worn, real

def features(T):
    """returns covariates X (n x 3), fixed scalar text features F (dict name->array), sign sets, logo sets."""
    ids = list(T)
    cov, F, S, G, first = [], collections.defaultdict(list), [], [], []
    for k in ids:
        toks, words, nums, logos, signs, nl, edge, worn, real = base_stats(T[k])
        n_sign = len(signs) + len(logos) + len(nums)
        cov.append([np.log1p(n_sign), np.log1p(nl), edge / max(1, len(real))])
        vals = [t['v'] for t in nums if isinstance(t.get('v'), (int, float))]
        F['words_per_line'].append(len(words) / nl)
        F['nums_per_line'].append(len(nums) / nl)
        F['mean_wlen'].append(np.mean([len(w['s']) for w in words]) if words else 0)
        F['share_1sign'].append(np.mean([len(w['s']) == 1 for w in words]) if words else 0)
        F['frac_share'].append(np.mean([bool(t.get('frac')) for t in nums]) if nums else 0)
        F['log_max_num'].append(np.log1p(max(vals)) if vals else 0)
        F['log_sum_num'].append(np.log1p(sum(vals)) if vals else 0)
        F['n_logo_types'].append(len(set(t.get('v', '').split('+')[0] for t in logos)))
        F['logo_share'].append(len(logos) / max(1, len(real)))
        F['has_total'].append(int(any(w['s'][:2] == ['KU', 'RO'] or w['s'][-2:] == ['KU', 'RO'] for w in words)))
        F['worn_share'].append(worn / max(1, len(real)))
        F['word_then_num'].append(np.mean([real[i + 1]['t'] == 'num' for i in range(len(real) - 1) if real[i]['t'] == 'word']) if words else 0)
        F['n_faces'].append(len(T[k]))
        F['first_is_word'].append(int(bool(real) and real[0]['t'] == 'word'))
        F['round_nums'].append(np.mean([v % 10 == 0 for v in vals]) if vals else 0)
        S.append(set(signs)); G.append(set(t.get('v', '').split('+')[0] for t in logos))
        fw = next((w for w in words), None)
        first.append(set(fw['s']) if fw else set())
    return ids, np.array(cov), {k: np.array(v, float) for k, v in F.items()}, S, G, first

def random_text_feature(rng, Fnames, sign_pool, logo_pool):
    r = rng.random()
    if r < 0.35:
        return ('scalar', rng.choice(Fnames))
    if r < 0.75:
        k = int(rng.integers(1, 5))
        return ('signs', tuple(sorted(rng.choice(sign_pool, k, replace=False))))
    if r < 0.9:
        k = int(rng.integers(1, 3))
        return ('logos', tuple(sorted(rng.choice(logo_pool, k, replace=False))))
    k = int(rng.integers(1, 4))
    return ('first', tuple(sorted(rng.choice(sign_pool, k, replace=False))))

def eval_text_feature(h, F, S, G, first):
    kind, arg = h
    if kind == 'scalar': return F[arg]
    if kind == 'signs': return np.array([len(s & set(arg)) > 0 for s in S], float)
    if kind == 'logos': return np.array([len(g & set(arg)) > 0 for g in G], float)
    return np.array([len(s & set(arg)) > 0 for s in first], float)
