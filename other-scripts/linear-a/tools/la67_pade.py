#!/usr/bin/env python3
"""LA-67 kill sweep, test K-PADE: re-run la53 cycle 3 (attention specs chosen on Linear B + Ur III,
applied to Linear A with a within-stratum permutation null) with FRESH seeds and 1,000 permutations,
to re-test the la53 grade-C guess 'PA-DE is a specially marked entry' (p 0.010, n 3). Output in
data/la67_ckpt/la53/c3.json; the decoy comparison (all n = 3 types) is made in la67_c2.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('NPERM', '1000')
import la53_common as C
lb = C.lb_docs(); ur = C.ur3_docs()
C.lb_docs = lambda: lb; C.ur3_docs = lambda n=6000: ur
C.CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', 'la53'); os.makedirs(C.CK, exist_ok=True)
_s = C.seed
C.seed = lambda name: _s('la67-' + name)
import la53_c3 as M
M.main()
