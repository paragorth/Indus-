"""Prepare in-session reading batches (no API key needed).

    python3 prepare_batches.py <wave_name> [--pass nonortho] [--size 60] [--limit 0]

For every screened-in paper not yet read, writes work/claude_in/<key>.txt (header + full text or
abstract) and manifests work/claude_batches/<wave>_NN.txt (one input path per line). Readers follow
work/claude_batches/INSTRUCTIONS.md and write work/claude_out/<wave>_NN.json; then run
`python3 validate_claude.py --ingest`. UK/Ireland and full-text papers come first.
"""
import argparse
import os

import extract
import pipeline

IN_DIR = os.path.join(pipeline.WORK, "claude_in")
BATCH_DIR = os.path.join(pipeline.WORK, "claude_batches")


def fname(key):
    return key.replace(":", "_").replace("/", "_") + ".txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wave")
    ap.add_argument("--pass", dest="pas", default="nonortho")
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    keys, recs, _ = extract._todo(a.pas, 0)
    scr = pipeline.load("screen.json", {})
    import classify
    # closed-loop papers first, then UK/Ireland, then full text available
    hits = pipeline.load(f"hits_{a.pas}.json", {})
    thin = {k for k, h in hits.items() if any(l.startswith("thin ") for l in h.get("labels", []))}
    keys.sort(key=lambda k: (k not in thin,                 # thin specialties first (gap report)
                             scr[k].get("audit_kind") != "closed-loop or re-audit",
                             not classify.uk_ireland(recs[k].get("affiliations")),
                             not os.path.exists(os.path.join(pipeline.FT_DIR, f"{recs[k].get('pmcid')}.xml"))))
    queued = set()                      # already in a manifest (being read now)
    import glob
    for m in glob.glob(os.path.join(BATCH_DIR, "*.txt")):
        queued.update(l.strip() for l in open(m) if l.strip())
    keys = [k for k in keys if os.path.join(IN_DIR, fname(k)) not in queued]
    keys = [k for k in keys if len(recs[k].get("abstract") or "") >= 200]   # title-only records carry no results
    from extra_harvest import CLINICAL, NOT_CLINICAL      # drop financial, IT and research-governance "audits"
    keys = [k for k in keys if CLINICAL.search((recs[k].get("title") or "") + " " + (recs[k].get("abstract") or ""))
            and not NOT_CLINICAL.search(recs[k].get("title") or "")]
    if a.limit:
        keys = keys[:a.limit]
    os.makedirs(IN_DIR, exist_ok=True)
    paths = []
    for k in keys:
        r = recs[k]
        kind, text = extract.build_input(r)
        p = os.path.join(IN_DIR, fname(k))
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"KEY: {k}\nJournal: {r.get('journal', '')} ({r.get('year', '')})\n"
                    f"Affiliations: {'; '.join(r.get('affiliations', [])[:5]) or 'not given'}\n"
                    f"Text supplied: {kind}\nLibrary pass: {a.pas}\n\n{text}\n")
        paths.append(p)
    n = 0
    for i in range(0, len(paths), a.size):
        with open(os.path.join(BATCH_DIR, f"{a.wave}_{n:02d}.txt"), "w") as f:
            f.write("\n".join(paths[i:i + a.size]) + "\n")
        n += 1
    print(f"{len(paths)} papers in {n} batches ({a.wave}_00 … {a.wave}_{n - 1:02d})")


if __name__ == "__main__":
    main()
