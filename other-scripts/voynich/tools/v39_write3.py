"""v39 cycle 3 rows: massive random merge guessing."""
import numpy as np
from scipy.stats import spearmanr
import v39_lib as L
from v39_cycle3 import NSET, halves

FN = 'v39_cycle3.txt'
FAM = [set('ktpf'), set('KTPF'), set('CS')]


def is_twin(name, a, b, planted):
    if name == 'Pla': return frozenset((a, b)) in planted
    return any(a in s and b in s for s in FAM)


def summary(name):
    halves(name)
    planted = halves.__globals__['_B'].get('planted', set())
    R = {s: L.load(f'c3_{name}|d|{s}.json') for s in range(NSET[name])}
    R = {s: r for s, r in R.items() if r}
    base = L.load(f'c3_{name}|d|-1.json')
    sel = L.load(f'c3_{name}_sel.json')
    S = np.array([R[s]['S'] for s in R])
    top, rnd = sel['top'], sel['rnd']
    allp = [p for s in R for p in R[s]['pairs']]
    topp = [p for s in top for p in R[s]['pairs']]
    tw_all = np.mean([is_twin(name, a, b, planted) for a, b in allp])
    tw_top = np.mean([is_twin(name, a, b, planted) for a, b in topp])
    # S by number of twin pairs in the set
    nt = np.array([sum(is_twin(name, a, b, planted) for a, b in R[s]['pairs']) for s in R])
    rho_t = spearmanr(nt, S).correlation
    out = dict(n=len(R), base=base['S'], mu=S.mean(), sd=S.std(), max=S.max(), tw_all=tw_all, tw_top=tw_top, rho_t=rho_t,
               basepLI=base['pLI'], basepG=base['pG'], maxpLI=max(R[s]['pLI'] for s in R))
    for part in ('t', 'ti'):
        H = {s: L.load(f'c3_{name}|{part}|{s}.json') for s in set(top) | set(rnd) | {-1}}
        if not any(H.values()): continue
        hr = np.array([H[s]['S'] for s in rnd if H.get(s)])
        q95 = np.quantile(hr, 0.95)
        surv = [s for s in top if H.get(s) and H[s]['S'] > max(q95, H[-1]['S'] + 0.10)]
        both = [s for s in set(top) | set(rnd) if H.get(s)]
        rho = spearmanr([R[s]['S'] for s in both], [H[s]['S'] for s in both]).correlation
        out[part] = dict(base=H[-1]['S'], rnd_mu=hr.mean(), q95=q95, surv=surv, rho=rho,
                         top_mu=np.mean([H[s]['S'] for s in top if H.get(s)]),
                         surv_pairs=[R[s]['pairs'] for s in surv],
                         surv_pLI=[H[s]['pLI'] for s in surv], surv_pG=[H[s]['pG'] for s in surv],
                         surv_arrow=[H[s]['arrow'] for s in surv], surv_gap=[H[s]['gap'] for s in surv])
    return out


def esc(p):
    return '+'.join(''.join(c if c.isascii() else '#' for c in a) + '=' + ''.join(c if c.isascii() else '#' for c in b) for a, b in p)


if __name__ == '__main__':
    with open(f'{L.LOOPS}/{FN}', 'w') as f:
        f.write('# v39 cycle 3 - MASSIVE RANDOM MERGE GUESSING (4 Oct 2026): random merge sets (1-8 pairs of units with >= 100 tokens) scored on a discovery half by S = P(language or conlang) - P(generator) (v31 classifier), top 12 and 40 random sets rescored on the held-out half (and on IT2a for the Voynich). Tools v39_cycle3.py, v39_write3.py; checkpoints data/v39_ckpt/c3_*.json\n')
        f.write('| row | method and control | result | verdict |\n|---|---|---|---|\n')
    res = {}
    for name, lab in (('Pla', 'POSITIVE control: Latin herbal with 5 planted free twin splits'), ('V', 'Voynich ZL (held-out: ZL other half, IT2a same pages)')):
        o = summary(name); res[name] = o
        txt = (f"{o['n']} sets; discovery S unmerged {o['base']:+.2f} (pLI {o['basepLI']:.2f}, pG {o['basepG']:.2f}), random sets {o['mu']:+.2f}+-{o['sd']:.2f}, best {o['max']:+.2f} (max pLI {o['maxpLI']:.2f}); "
               f"twin/planted share of pairs: all sets {o['tw_all']:.3f}, top-12 sets {o['tw_top']:.3f}; Spearman(S, number of twin pairs in set) {o['rho_t']:+.2f}")
        for part in ('t', 'ti'):
            if part in o:
                h = o[part]
                txt += (f" // held-out {'ZL' if part == 't' else 'IT2a'}: unmerged {h['base']:+.2f}, random {h['rnd_mu']:+.2f} (95th pct {h['q95']:+.2f}), top-12 mean {h['top_mu']:+.2f}, "
                        f"discovery-vs-held-out rho {h['rho']:+.2f}, survivors (held-out S > random 95th pct AND > unmerged + 0.10) {len(h['surv'])}/12" +
                        (': ' + '; '.join(f"{esc(p)} (pLI {a:.2f}, pG {g:.2f}, arrow {int(ar)}, gap {gp:.2f})" for p, a, g, ar, gp in zip(h['surv_pairs'], h['surv_pLI'], h['surv_pG'], h['surv_arrow'], h['surv_gap'])) if h['surv'] else ''))
        L.row(FN, f'V-39.3.{1 if name == "Pla" else 2}', lab, txt, 'see 3.V')
    print(open(f'{L.LOOPS}/{FN}').read())
