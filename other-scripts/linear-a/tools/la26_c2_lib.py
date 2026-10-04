"""LA-26 shared helpers for the word-level ring-path statistic (cycle 2)."""
import itertools, collections
import numpy as np
from la26_common import ring_matrix


def site_sets(docs, key='words'):
    S = collections.defaultdict(set)
    for d in docs:
        for w in d[key]:
            S[w].add(d['site'])
    return {w: s for w, s in S.items() if len(s) >= 2}


def F_stat(sets, Rset):
    num = den = 0
    for w, s in sets.items():
        for a, b in itertools.combinations(sorted(s), 2):
            den += 1; num += (a, b) in Rset
    return num / den if den else np.nan, num, den


def rset_from(inc, nodes):
    R = ring_matrix(inc, nodes)
    return {(a, b) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if R[i, j] > 0} | \
           {(b, a) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if R[i, j] > 0}


