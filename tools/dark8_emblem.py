"""S-DARK-8.3: 'emblem first' - does prepending the field symbol as a token tighten the grammar?
Seals with a recorded symbol (canonical 'symbol' field, coarse class before ':'). Trigram model on
MD + Harappa, nll of the SIGN tokens on held-out sites, with the emblem token (a) prepended, (b)
appended (emblem predicted from the last two signs; reported as nll of the emblem token), (c) absent.
Control: emblem labels shuffled across texts (200 reps). Also MI(emblem; first sign) vs MI(emblem;
last sign) with shuffle nulls, all texts. Run on seq_raw, seq_strong, seq_all."""
import json, math, random, collections, sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark8_permute import Tri
OUT = '/home/user/Indus-/data/derived/dark/'
rng = random.Random(83)

def nll_signs(m, s, prefix):
    s = ('<s>', prefix) + s + ('</s>',)
    return sum(-m.lp(s[i-2], s[i-1], s[i]) for i in range(2, len(s)))

def mi(pairs):
    n = len(pairs); ca = collections.Counter(a for a, b in pairs); cb = collections.Counter(b for a, b in pairs); cab = collections.Counter(pairs)
    return sum(c / n * math.log2(c * n / (ca[a] * cb[b])) for (a, b), c in cab.items())

def run(variant, lines):
    d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
    tx = []
    for t in d:
        s = t[variant]
        sym = t['symbol'].split(':')[0]
        if not s or 0 in s or len(s) < 2 or not t['type'].startswith('SEAL') or sym in ('', '-'): continue
        tx.append((t['site'], 'E_' + sym, tuple(s)))
    big = ('Mohenjo-daro', 'Harappa')
    train = [x for x in tx if x[0] in big]; test = [x for x in tx if x[0] not in big]
    V = len({x for _, _, s in tx for x in s}) + len({e for _, e, _ in tx}) + 1
    def fit_eval(train, test):
        m0 = Tri([s for _, _, s in train], V)
        m1 = Tri([(e,) + s for _, e, s in train], V)
        m2 = Tri([s + (e,) for _, e, s in train], V)
        ntok = sum(len(s) + 1 for _, _, s in test)
        base = sum(nll_signs(m0, s, '<s>') for _, _, s in test) / ntok
        pre = sum(nll_signs(m1, s, e) for _, e, s in test) / ntok
        # emblem-last: nll of the emblem token given last two signs, per text
        last = sum(-m2.lp(s[-2] if len(s) > 1 else '<s>', s[-1], e) for _, e, s in test) / len(test)
        # emblem-first analogue: nll of emblem token from nothing (unigram) vs. first sign predicted given emblem
        return base, pre, last
    base, pre, last = fit_eval(train, test)
    nulls = []
    for _ in range(200):
        es = [e for _, e, _ in tx]; rng.shuffle(es)
        sh = [(st, e, s) for (st, _, s), e in zip(tx, es)]
        b, p, l = fit_eval([x for x in sh if x[0] in big], [x for x in sh if x[0] not in big])
        nulls.append((b - p, l))
    g = base - pre; gn = sorted(x[0] for x in nulls); ln = sorted(x[1] for x in nulls)
    counts = collections.Counter(e for _, e, _ in tx)
    lines.append(f'{variant}: {len(tx)} seals with emblem ({len(counts)} classes; top {counts.most_common(4)}); train {len(train)}, test {len(test)}. '
                 f'Held-out nll/token of signs: no emblem {base:.4f}, emblem prepended {pre:.4f} (gain {g:+.4f}; shuffled-emblem null median {gn[100]:+.4f}, 95th {gn[190]:+.4f}, P = {sum(1 for x in gn if x >= g)/200:.3f}). '
                 f'Emblem predicted from last two signs: {last:.3f} nats (null median {ln[100]:.3f}, 5th {ln[10]:.3f}; P = {sum(1 for x in ln if x <= last)/200:.3f}).')
    # MI first vs last, all texts, length >= 3
    tx3 = [x for x in tx if len(x[2]) >= 3]
    mf = mi([(e, s[0]) for _, e, s in tx3]); ml = mi([(e, s[-1]) for _, e, s in tx3]); mm = mi([(e, s[len(s)//2]) for _, e, s in tx3])
    nf = []; nl = []
    for _ in range(200):
        es = [e for _, e, _ in tx3]; rng.shuffle(es)
        nf.append(mi(list(zip(es, [s[0] for _, _, s in tx3])))); nl.append(mi(list(zip(es, [s[-1] for _, _, s in tx3]))))
    nf.sort(); nl.sort()
    lines.append(f'{variant}: MI(emblem; first sign) {mf:.3f} bits (null med {nf[100]:.3f}, 95th {nf[190]:.3f}); MI(emblem; last sign) {ml:.3f} (null med {nl[100]:.3f}, 95th {nl[190]:.3f}); MI(emblem; middle sign) {mm:.3f}. n = {len(tx3)}')
    return dict(n=len(tx), base=base, pre=pre, gain=g, null95=gn[190], last=last, mi_first=mf, mi_last=ml, mi_mid=mm, null_first95=nf[190], null_last95=nl[190])

if __name__ == '__main__':
    lines = []; res = {}
    for v in ('seq_raw', 'seq_strong', 'seq_all'):
        res[v] = run(v, lines)
    json.dump(res, open(OUT + 'loop8_cycle3_emblem.json', 'w'), indent=1)
    open(OUT + 'loop8_cycle3_log.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
