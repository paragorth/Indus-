"""Loop 41 cycle 2b: S349 P1/P2 under the original null (R fixed), the both-shuffled null and Markov nulls. Appends to loop41_cycle2.txt"""
import sys, random, json
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
out = open(DARK + 'loop41_cycle2.txt', 'a')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical'); rnd = random.Random(413)
# S349 nulls
P('\n# S349 P1/P2 on the die regime: original null (R fixed, held-out shuffled) vs both-shuffled vs Markov order-1 (fitted on held-out, per class)')
for lvl in ('seq_raw', 'seq_all'):
    for name, T in (('FULL', dedup(C, lvl, 'die')), ('defragile', defragile(dedup(C, lvl, 'die')))):
        o, nm, nx, n = p1p2_fixedR(T, rnd, nnull=50)
        h = p1p2(T, rnd, nnull=30)
        om, mm, mx = p2_markov(T, rnd, nnull=30, order=1)
        om2, mm2, mx2 = p2_markov(T, rnd, nnull=30, order=2)
        o1, n1, x1, ns = nest_markov(T, rnd, nnull=20, order=1)
        P(f'{lvl} {name:10s} held-out n={n}: P2 {o:.3f} vs R-fixed shuffle {nm:.3f} (max {nx:.3f}) = {o/nm:.1f}x | both-shuffled {h["P2_null"]:.3f} = {o/h["P2_null"]:.1f}x | Markov-1 {mm:.3f} (max {mx:.3f}) = {om/mm:.1f}x | Markov-2 {mm2:.3f} (max {mx2:.3f}) = {om2/mm2:.1f}x || P1 {h["P1"]:.3f} vs shuffle {h["P1_null"]:.3f} = {h["P1"]/h["P1_null"]:.1f}x || nesting all-corpus {o1:.3f} vs Markov-1 {n1:.3f} (max {x1:.3f}) = {o1/n1:.2f}x')
