#!/usr/bin/env python3
"""pe74 cycle 2, item C: the pe70 sealed document type (pe70_c1b, unchanged code) on a filtered corpus:
sealed-association of each separator in 5 subsets (all, one per seal, no MDP 26S, dims non-fragment, no seal groups;
5,000 permutations within volume x size band) and leave-one-volume-out AUC vs permuted labels.
usage: python3 pe74_c2_pe70.py MODE -> data/pe74_ckpt/<MODE>/pe70/c1b.json"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
H.activate(MODE)
import pe70_common as C70
d = H.redirect_ck(C70, 'pe70')
import pe70_c1b as M
M.CK = d
M.main()
