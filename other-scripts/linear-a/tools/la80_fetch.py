import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L
ids = json.load(open(sys.argv[1])); g = L.fetch(ids); print(len(ids), len(g))
