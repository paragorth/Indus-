#!/usr/bin/env python3
"""pe74 cycle 2, items G and H: pe66's own tests (unchanged code) on a filtered corpus.
c1 L = [M327+M342] header slot, c1 O = grain-office signs, c3 D2 = Yahya lexicon (final pe66 metric),
c3 V2 = M005~a header slot.
usage: python3 pe74_c2_pe66.py MODE  -> data/pe74_ckpt/<MODE>/pe66/pe74.rows"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
ROOT = H.activate(MODE)
import pe66_lib
d = H.redirect_ck(pe66_lib, 'pe66')
import pe66_c1 as M1
import pe66_c3 as M3
M1.CK = M3.CK = d
for w in 'LO':
    getattr(M1, 'test_' + w)()
for w in ('D2', 'V2'):
    getattr(M3, 'test_' + w)()
open(os.path.join(d, 'pe74.rows'), 'w').write('\n'.join(M1.OUT + M3.OUT) + '\n')
