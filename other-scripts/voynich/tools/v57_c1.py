"""v57 cycle 1: THE INK CLOCK. Was the Voynich written in reading order?
Ink fades within a pen load, so ink similarity is a clock. Three tests per corpus:
 D1 cross-diagonal: last word of line L vs first of L+1 (adjacent in reading time) against
    first of L vs last of L+1 (same spatial offsets, ~2 lines apart in time). Sign-flip null.
 D2 sawtooth arrow: skewness of within-line darkness increments, left-to-right
    (re-dip = sudden darkening, fade = slow). Null: within-line shuffles.
 D3 writing-order search: ~5,000 hypotheses (line order x direction x pass structure x
    seeds) scored by ink continuity on half the pages, re-scored on the other half.
Controls: Latin (CREMMA) must prefer reading order; PLANTED Latin with ink re-laid in
an alternative writing order must be recovered; shuffled-line null."""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v57_lib import Corpus, CK

RNG = np.random.default_rng(57)


def page_lines(C):
    out = {}
    for L in C.lines:
        out.setdefault(int(C.page[L[0]]), []).append(L)
    return out


# ---------------- writing-order hypotheses
def make_order(lines, h, seed):
    rng = np.random.default_rng(seed)
    n = len(lines)
    lo, dr, ps = h
    idx = list(range(n))
    if lo == 'rev':
        idx = idx[::-1]
    elif lo == 'oddeven':
        idx = idx[0::2] + idx[1::2]
    elif lo == 'pairswap':
        idx = [j for i in range(0, n, 2) for j in idx[i:i + 2][::-1]]
    elif lo == 'halves':
        idx = idx[n // 2:] + idx[:n // 2]
    elif lo == 'blockrev':
        b = 2 + seed % 5
        idx = [j for i in range(0, n, b) for j in idx[i:i + b][::-1]]
    elif lo == 'perm':
        idx = list(rng.permutation(n))
    elif lo == 'localperm':
        key = np.arange(n) + rng.normal(0, 1 + seed % 4, n)
        idx = list(np.argsort(key))
    seqs = []
    for t, i in enumerate(idx):
        L = list(lines[i])
        if dr == 'rtl' or (dr == 'alt' and t % 2 == 1):
            L = L[::-1]
        seqs.append(L)
    if ps == 'none':
        return np.array([j for s in seqs for j in s])
    if ps.startswith('init') or ps.startswith('final'):
        m = int(ps[-1])
        first, rest = [], []
        for s in seqs:
            s0 = sorted(s)  # reading order inside line
            sel = s0[:m] if ps.startswith('init') else s0[-m:]
            first += [j for j in s if j in sel]
            rest += [j for j in s if j not in sel]
        return np.array(first + rest)
    if ps == 'col':
        mx = max(len(s) for s in seqs)
        return np.array([sorted(s)[k] for k in range(mx) for s in seqs if k < len(s)])
    if ps == 'halfline':  # left halves of all lines, then right halves
        a, b = [], []
        for s in seqs:
            s0 = sorted(s); h2 = len(s0) // 2
            a += [j for j in s if j in s0[:h2]]; b += [j for j in s if j in s0[h2:]]
        return np.array(a + b)
    raise ValueError(ps)


def hypotheses():
    H = []
    los = ['id', 'rev', 'oddeven', 'pairswap', 'halves']
    drs = ['ltr', 'rtl', 'alt']
    pss = ['none', 'init1', 'init2', 'final1', 'final2', 'col', 'halfline']
    for lo, dr, ps in itertools.product(los, drs, pss):
        H.append(((lo, dr, ps), 0))
    for s in range(1, 230):
        for dr, ps in itertools.product(drs, pss):
            H.append((('blockrev' if s < 8 else ('localperm' if s < 120 else 'perm'), dr, ps), s))
    return H


def cont_score(r, order, clip=3.0):
    d = np.abs(np.diff(r[order]))
    return np.minimum(d, clip).mean()


def page_ref(r, idx, clip=3.0):
    v = r[idx]
    d = np.abs(v[:, None] - v[None, :])
    return np.minimum(d, clip)[np.triu_indices(len(v), 1)].mean()


def search(C, r, H, pl):
    pages = sorted(pl)
    S = np.zeros((len(H), len(pages)))
    for j, p in enumerate(pages):
        lines = pl[p]
        allidx = np.concatenate(lines)
        ref = page_ref(r, allidx)
        for i, (h, s) in enumerate(H):
            o = make_order(lines, h, s * 1000 + p)
            S[i, j] = 1 - cont_score(r, o) / ref
    return S, pages


# ---------------- D1, D2
def diag(C, r, pl):
    st = []
    for p, lines in pl.items():
        for L1, L2 in zip(lines[:-1], lines[1:]):
            if C.li[L2[0]] != C.li[L1[0]] + 1 or len(L1) < 3 or len(L2) < 3:
                continue
            a, b, c, d = r[L1[-1]], r[L2[0]], r[L1[0]], r[L2[-1]]
            st.append(abs(c - d) - abs(a - b))
    st = np.array(st)
    obs = st.mean()
    signs = RNG.choice([-1, 1], size=(20000, len(st)))
    null = (signs * st).mean(1)
    return dict(n=len(st), mean=float(obs), z=float(obs / null.std()), p=float((null >= obs).mean()))


def skew(x):
    x = x - x.mean()
    return (x ** 3).mean() / (x ** 2).mean() ** 1.5


def arrow(C, r, pl, nsur=500):
    def incs(rr):
        return np.concatenate([np.diff(rr[L]) for ls in pl.values() for L in ls if len(L) > 2])
    obs = skew(incs(r))
    sur = []
    for _ in range(nsur):
        rr = r.copy()
        for ls in pl.values():
            for L in ls:
                rr[L] = r[RNG.permutation(L)]
        sur.append(skew(incs(rr)))
    sur = np.array(sur)
    # also: mean darkness trend within line (slope per word, LTR)
    sl = [np.polyfit(np.arange(len(L)), r[L], 1)[0] for ls in pl.values() for L in ls if len(L) >= 5]
    return dict(skew=float(obs), sur_mu=float(sur.mean()), sur_sd=float(sur.std()),
                z=float((obs - sur.mean()) / sur.std()), slope=float(np.mean(sl)),
                slope_se=float(np.std(sl) / np.sqrt(len(sl))))


def planted(C, pl, h, seed):
    """Lay each page's reading-order ink sequence down in writing order h."""
    r2 = C.r.copy()
    for p, lines in pl.items():
        read = np.concatenate(lines)
        o = make_order(lines, h, seed * 1000 + p)
        r2[o] = C.r[read]
    return r2


def run(C, r, tag, H, pl, out):
    S, pages = search(C, r, H, pl)
    A = np.array([j for j, p in enumerate(pages) if j % 2 == 0]); B = np.array([j for j, p in enumerate(pages) if j % 2 == 1])
    sa, sb, sall = S[:, A].mean(1), S[:, B].mean(1), S.mean(1)
    ia = np.argsort(-sa)
    read = H.index((('id', 'ltr', 'none'), 0))
    rank_all = int((sall > sall[read]).sum()) + 1
    top = [(str(H[i][0]) + ('#%d' % H[i][1] if H[i][1] else ''), round(float(sa[i]), 4), round(float(sb[i]), 4),
            int((sb > sb[i]).sum()) + 1) for i in ia[:5]]
    # wins over reading order on pages: per page, does the best non-reading hypothesis beat reading?
    res = dict(tag=tag, n_hyp=len(H), read_score=round(float(sall[read]), 4), read_rank=rank_all,
               best=str(H[int(np.argmax(sall))][0]), best_score=round(float(sall.max()), 4),
               top_on_A_with_B=top, diag=diag(C, r, pl), arrow=arrow(C, r, pl))
    out.append(res)
    print(json.dumps(res), flush=True)
    return S


if __name__ == '__main__':
    H = hypotheses()
    print('hypotheses', len(H), flush=True)
    out = []
    for which in ['L', 'V']:
        for meas in ['top', 'med']:
            C = Corpus(which, meas)
            pl = page_lines(C)
            run(C, C.r, f'{which}_{meas}', H, pl, out)
            if meas == 'top':
                # shuffled-line null: lines' ink rows permuted within page (keeps within-line structure)
                r0 = C.r.copy()
                for p, lines in pl.items():
                    perm = RNG.permutation(len(lines))
                    for L, j in zip(lines, perm):
                        src = lines[j]
                        v = C.r[src]
                        r0[L] = np.interp(np.linspace(0, 1, len(L)), np.linspace(0, 1, len(v)), v) if len(v) > 1 else v[0]
                run(C, r0, f'{which}_top_LINESHUF', H, pl, out)
                if which == 'L':
                    for h, s in [(('id', 'ltr', 'init1'), 0), (('oddeven', 'ltr', 'none'), 0),
                                 (('id', 'rtl', 'none'), 0), (('perm', 'ltr', 'none'), 150), (('id', 'ltr', 'col'), 0)]:
                        run(C, planted(C, pl, h, s), f'L_PLANT_{h}', H, pl, out)
    json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
