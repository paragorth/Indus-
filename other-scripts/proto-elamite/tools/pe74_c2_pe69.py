#!/usr/bin/env python3
"""pe74 cycle 2, item B: the per-head 60 rule (pe69 cycle 2, unchanged code) on a filtered corpus.
Headline: bare 'M288 n' lines after PERSON-final count lines, hits (60/120 x last, 60 x team) vs re-deal expectation
(pe69: 34/49 vs 5.5).  Also the pe64/pe27 'after M054/M388/M124' scope.
usage: python3 pe74_c2_pe69.py MODE   -> data/pe74_ckpt/<MODE>/pe69/c2.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
ROOT = H.activate(MODE)
os.environ.setdefault('NREP', '200')
import pe59_lib as P
H.redirect_ck(P, 'pe59', copy=['pc_tabs.json', 'pc_admin_ids.json'])
import pe69_lib
import pe69_c2 as M
d = os.path.join(ROOT, 'pe69')
os.makedirs(d, exist_ok=True)
M.CK = d
M.DATA = d          # decoded-slot table goes to the checkpoint, not data/
pe69_lib.CK = d
M.main()
