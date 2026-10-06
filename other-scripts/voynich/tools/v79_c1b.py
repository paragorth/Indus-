"""v79 cycle 1b: is the Voynich first-order chain table COUNTER-SHAPED? Same random counter search (line units) on
lift-permuted Markov surrogates: each row keeps its strength of preference and the successor marginal is kept, but
WHICH successors are favoured is scrambled. A counter-shaped table (favoured successors chained into one cycle) beats
these; a table whose favourites form funnels/short loops falls below them. Controls: Dante numbers, dominical letters,
golden numbers, Isidore surface marker."""
import sys, time
import numpy as np
import v79_lib as L, v79_c1 as C1

def main():
    part = int(sys.argv[1]); t0 = time.time()
    names = ['ZL3b', 'IT2a', 'C_DANTE', 'C_DOM', 'C_GOLD', 'N_ISI'][part::2]
    allc = C1.corpora()
    for n in names:
        T, al = L.chain_table(allc[n][1]); T = C1.add_entry_alpha(T)
        rng = np.random.default_rng(9)
        vals = []
        for j in range(8):
            Tn = L.markov_surrogate(T, np.random.default_rng(int(rng.integers(1e9))), liftperm=True)
            r = C1.search(Tn, nhyp=C1.N_HYP // 2, seed=200 + j)
            vals.append({u: r[u]['held_z'] for u in ('line/para', 'line/page', 'line/book')})
        L.psave('c1b_%s.pkl' % n, vals)
        print(n, '%.0fs' % (time.time() - t0), {u: (round(np.mean([v[u] for v in vals]), 1), round(np.std([v[u] for v in vals]), 1)) for u in vals[0]}, flush=True)

if __name__ == '__main__':
    main()
