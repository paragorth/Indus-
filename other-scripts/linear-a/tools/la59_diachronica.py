#!/usr/bin/env python3
"""LA-59: a second human key from attested sound changes (Searchable Index Diachronica 10.2,
chridd.nfshost.com/diachronica/all; downloaded to data/la59_ckpt/idx.html).
Only unconditioned or conditioned one-to-one segment rules 'X Y Z -> X' Y' Z'' with equal counts are used.
Output: symmetric change-count matrices over the 16 Miller-Nicely consonants and the 5 vowels, and the
similarity S = (c_ab + 0.5) / sqrt(n_a n_b) rescaled to the Miller-Nicely off-diagonal range."""
import re, html, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import CK, MN_LABELS, VOWELS, mn_matrix

IPA_C = dict(p='p', t='t', k='k', f='f', θ='T', s='s', ʃ='S', b='b', d='d', ɡ='g', g='g', v='v', ð='D', z='z', ʒ='Z',
             m='m', n='n')
IPA_V = dict(a='a', ɑ='a', æ='a', e='e', ɛ='e', i='i', ɪ='i', o='o', ɔ='o', u='u', ʊ='u')

def seg(x):
    x = x.strip('*').replace('ː', '').replace('ʰ', '').replace('ʲ', '').replace('ʷ', '')
    if x in IPA_C:
        return 'C', IPA_C[x]
    if x in IPA_V:
        return 'V', IPA_V[x]
    return None

def build():
    t = open(os.path.join(CK, 'idx.html'), encoding='utf8').read()
    t = html.unescape(re.sub(r'<[^>]+>', '', t))
    cc = np.zeros((16, 16)); vv = np.zeros((5, 5)); nrule = 0
    for l in t.split('\n'):
        if '→' not in l or l.count('→') != 1:
            continue
        lhs, rhs = l.split('→')
        rhs = rhs.split('/')[0].split('(')[0]
        if any(ch in lhs + rhs for ch in '{}[]'):
            continue
        L = lhs.strip().lstrip('—').split(); R = rhs.strip().split()
        if not L or len(L) != len(R):
            continue
        for a, b in zip(L, R):
            sa, sb = seg(a), seg(b)
            if not sa or not sb or sa[0] != sb[0] or sa[1] == sb[1]:
                continue
            nrule += 1
            if sa[0] == 'C':
                i, j = MN_LABELS.index(sa[1]), MN_LABELS.index(sb[1]); cc[i, j] += 1
            else:
                i, j = VOWELS.index(sa[1]), VOWELS.index(sb[1]); vv[i, j] += 1
    return cc, vv, nrule

def sim(c, ref_off):
    s = c + c.T; n = s.sum(1) + 1
    S = (s + 0.5) / np.sqrt(np.outer(n, n))
    off = ~np.eye(len(S), dtype=bool)
    r = np.argsort(np.argsort(S[off]))
    T = np.eye(len(S)); T[off] = np.quantile(ref_off, np.linspace(0, 1, off.sum()))[r]
    return (T + T.T) / 2

if __name__ == '__main__':
    cc, vv, n = build()
    mn = mn_matrix(); off = mn[~np.eye(16, dtype=bool)]
    SCd = sim(cc, off)
    from la59_common import vowel_key
    SV, _ = vowel_key(); SVd = sim(vv, SV[~np.eye(5, dtype=bool)])
    iu = np.triu_indices(16, 1)
    from scipy.stats import spearmanr
    print('rules used', n, 'consonant changes', cc.sum(), 'vowel changes', vv.sum())
    print('rho(diachronica C, Miller-Nicely)', spearmanr(SCd[iu], mn[iu]))
    iv = np.triu_indices(5, 1); print('rho(diachronica V, formant key)', spearmanr(SVd[iv], SV[iv]))
    top = sorted(((cc[i, j] + cc[j, i], MN_LABELS[i], MN_LABELS[j]) for i, j in zip(*iu)), reverse=True)[:12]
    print('top consonant changes', top)
    json.dump(dict(SC=SCd.tolist(), SV=SVd.tolist(), cc=cc.tolist(), vv=vv.tolist(), nrule=n), open(os.path.join(CK, 'diachronica_key.json'), 'w'))
