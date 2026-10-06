#!/usr/bin/env python3
"""LA-67 kill sweep: re-run the la52 iterated-learning chains with a NEW tag (= new seeds) and
checkpoints under data/la67_ckpt/la52, to re-test the grade-C guess 'OLE+RI dies faster than its
frequency predicts'. usage: la67_chains.py TAG COND NCHAINS WORKER G KIND   (same as la52_run.py)"""
import sys, os, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la52_common as C
C.CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', 'la52')
os.makedirs(C.CK, exist_ok=True)
sys.argv = [os.path.join(HERE, 'la52_run.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
