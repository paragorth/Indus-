"""pe43: random-phylogeny search over root signs.

A hypothesis = a forest over root signs (parent array). Families = root subtrees.
Fitness = modularity of the TRAIN-tablet co-occurrence graph under the family partition.
Massive random start (n_rand random forests), then hill-climb the best n_climb by random
subtree re-parenting (accept if dQ >= 0, small annealing); consensus of the top n_keep.
"""
import numpy as np
from pe43_common import modularity, forest_labels, relabel


def random_forest(n, q, rng):
    order = rng.permutation(n)
    par = -np.ones(n, dtype=int)
    for r, i in enumerate(order):
        if r > 0 and rng.random() < q:
            par[i] = order[rng.integers(r)]
    return par


class Climber:
    def __init__(self, A, par, rng):
        self.A = A; self.n = len(par); self.rng = rng
        self.W2 = A.sum(); self.d = A.sum(1)
        self.par = par.copy()
        self.kids = [[] for _ in range(self.n)]
        for i, p in enumerate(par):
            if p >= 0:
                self.kids[p].append(i)
        self.lab = np.empty(self.n, dtype=int)
        for i in range(self.n):
            j = i
            while self.par[j] >= 0:
                j = self.par[j]
            self.lab[i] = j
        M = np.zeros((self.n, self.n)); M[np.arange(self.n), self.lab] = 1
        self.AM = A @ M            # node -> community weight
        self.dc = M.T @ self.d     # community degree
        self.Q = modularity(A, relabel(self.lab))

    def subtree(self, i):
        out, st = [], [i]
        while st:
            x = st.pop(); out.append(x); st.extend(self.kids[x])
        return out

    def step(self, temp=0.0):
        rng = self.rng; n = self.n
        i = rng.integers(n)
        S = self.subtree(i)
        if rng.random() < 0.1:
            j = -1
        else:
            j = rng.integers(n)
            if j in S:
                return False
        a = self.lab[i]
        b = i if j < 0 else self.lab[j]
        if b == a and (j >= 0 or self.par[i] < 0):
            # same community: just re-wire tree, no change in Q
            if j >= 0 and self.par[i] != j:
                self._rewire(i, j)
            return False
        S = np.array(S)
        wSb = self.AM[S, b].sum() if b != i else 0.0
        wSa = self.AM[S, a].sum()
        wSS = self.A[np.ix_(S, S)].sum()
        dS = self.d[S].sum()
        if b == i:   # new community (i becomes a root; community id i must be empty: it is, since i was not a root)
            db = 0.0
        else:
            db = self.dc[b]
        da = self.dc[a]
        dinn = 2 * (wSb - (wSa - wSS))
        dQ = dinn / self.W2 - ((db + dS) ** 2 + (da - dS) ** 2 - db ** 2 - da ** 2) / self.W2 ** 2
        if dQ >= 0 or (temp > 0 and rng.random() < np.exp(dQ / temp)):
            colsum = self.A[:, S].sum(1)
            self.AM[:, a] -= colsum; self.AM[:, b] += colsum
            self.dc[a] -= dS; self.dc[b] += dS
            self.lab[S] = b
            self._rewire(i, j)
            self.Q += dQ
            return True
        return False

    def _rewire(self, i, j):
        p = self.par[i]
        if p >= 0:
            self.kids[p].remove(i)
        self.par[i] = j
        if j >= 0:
            self.kids[j].append(i)

    def labels(self):
        return relabel(self.lab.copy())


def search(A_tr, rng, n_rand=3000, n_climb=40, n_keep=15, steps=6000):
    n = len(A_tr)
    starts = []
    for _ in range(n_rand):
        q = rng.uniform(0.3, 0.97)
        par = random_forest(n, q, rng)
        starts.append((modularity(A_tr, forest_labels(par)), par))
    starts.sort(key=lambda x: -x[0])
    rand_Q = np.array([s[0] for s in starts])
    climbed = []
    for Q0, par in starts[:n_climb]:
        c = Climber(A_tr, par, rng)
        for s in range(steps):
            c.step(temp=1e-4 * max(0, 1 - s / (0.7 * steps)))
        climbed.append((c.Q, c.labels(), c.par.copy()))
    climbed.sort(key=lambda x: -x[0])
    keep = climbed[:n_keep]
    return keep, rand_Q, starts


def consensus(keep, n, thr=0.5):
    C = np.zeros((n, n))
    for _, lab, _ in keep:
        C += (lab[:, None] == lab[None, :])
    C /= len(keep)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1 - C; np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), 'average')
    lab = fcluster(Z, t=1 - thr, criterion='distance') - 1
    return relabel(lab), C
