#!/usr/bin/env python3
"""LA-67: word-level anchoring z from the fresh-seed iterated-learning chains (la52_c2 analysis, run on
data/la67_ckpt/la52). usage: la67_ilz.py TAG"""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la52_common as C
C.CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', 'la52')
sys.argv = [os.path.join(HERE, 'la52_c2.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
