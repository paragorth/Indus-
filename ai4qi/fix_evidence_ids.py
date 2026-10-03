"""One-off repair: citations written before ids became permanent pointed at the library version of the
day. For each proposed audit (parts files), topic card and CLAUDE.md, map every [id] above 1145 back to
the paper it meant (via the library at the commit it was written in) and on to that paper's permanent id.

    python3 fix_evidence_ids.py          # dry run: report
    python3 fix_evidence_ids.py --write
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RX = re.compile(r"\[(\d{1,5}(?:\s*[,;]\s*\d{1,5})*)\]")
CARD_ERA = "6cca95c"   # library version (3,713 entries) the draft topic cards were written against

_cache = {}


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=True).stdout


def old_map(commit):
    if commit not in _cache:
        lib = json.loads(git("show", f"{commit}:ai4qi/ai4qi-library.json"))
        _cache[commit] = {e["id"]: (e.get("paper") or {}).get("key") for e in lib["audits"] if e["id"] > 1145}
    return _cache[commit]


idmap = json.load(open(os.path.join(HERE, "work", "id_map.json")))
unresolved = []


def remap_text(s, commit, where):
    if not isinstance(s, str):
        return s
    om = old_map(commit)

    def one(m):
        out = []
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            i = int(n)
            if i <= 1145:
                out.append(str(i))
                continue
            key = om.get(i)
            new = idmap.get(key) if key else None
            if new is None:
                unresolved.append((where, i, key))
                out.append(n)
            else:
                out.append(str(new))
        return "[" + ", ".join(out) + "]"
    return RX.sub(one, s)


def remap_obj(o, commit, where):
    if isinstance(o, str):
        return remap_text(o, commit, where)
    if isinstance(o, list):
        return [remap_obj(x, commit, where) for x in o]
    if isinstance(o, dict):
        return {k: (remap_obj(v, commit, where) if k not in ("template", "standard", "question") else v)
                for k, v in o.items()}
    return o


def first_commit_per_question(path):
    rel = os.path.relpath(path, ROOT)
    seen = {}
    for c in git("log", "--reverse", "--format=%h", "--", rel).split():
        try:
            for a in json.loads(git("show", f"{c}:{rel}")):
                seen.setdefault(a["question"], c)
        except subprocess.CalledProcessError:
            pass
    return seen


def main(write):
    changed = 0
    for path in sorted(glob.glob(os.path.join(HERE, "new_audits", "parts", "*.json")) +
                       glob.glob(os.path.join(HERE, "new_audits", "nonortho_parts", "*.json"))):
        first = first_commit_per_question(path)
        data = json.load(open(path, encoding="utf-8"))
        out = []
        for a in data:
            c = first.get(a["question"], "HEAD")
            b = remap_obj(a, c, f"{os.path.basename(path)}: {a['question'][:50]}")
            changed += b != a
            out.append(b)
        if write:
            json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # topic cards: library copy and source files
    lib_path = os.path.join(HERE, "ai4qi-library.json")
    lib = json.load(open(lib_path, encoding="utf-8"))
    om = old_map(CARD_ERA)
    cards_fixed = 0
    for card in lib.get("topic_knowledge", []):
        if card.get("generated_by") != "pipeline":
            continue
        new = remap_obj({k: v for k, v in card.items() if k != "evidence_ids"}, CARD_ERA, f"card {card['topic']}")
        ev = [idmap.get(om.get(i)) if i > 1145 else i for i in card.get("evidence_ids", [])]
        new["evidence_ids"] = [i for i in ev if i is not None]
        cards_fixed += 1
        card.clear()
        card.update(new)
    for f in glob.glob(os.path.join(HERE, "work", "topics_out", "cards", "*.json")):
        body = json.load(open(f, encoding="utf-8"))
        nb = remap_obj(body, CARD_ERA, f"card file {os.path.basename(f)}")
        if write:
            json.dump(nb, open(f, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    cm = os.path.join(HERE, "CLAUDE.md")
    cm_new = remap_text(open(cm, encoding="utf-8").read(), CARD_ERA, "CLAUDE.md")
    if write:
        json.dump(lib, open(lib_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        open(cm, "w", encoding="utf-8").write(cm_new)
    print(f"proposed audits changed: {changed}; draft cards remapped: {cards_fixed}; unresolved refs: {len(unresolved)}")
    for u in unresolved[:40]:
        print("  unresolved:", u)


if __name__ == "__main__":
    main("--write" in sys.argv)
