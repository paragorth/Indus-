"""Compile the proposed orthopaedic audits (parts/*.json) into deployable outputs.

Outputs (all in ai4qi/new_audits/ unless noted):
  ortho_new_audits.json      every proposed audit, numbered ONA-001 ...
  catalogue.md               one short block per audit (question, standard, change, target)
  templates/ONA-xxx.csv      data-collection sheet: one column per template field, plus a
                             second header row with type/options so it opens straight in Excel
  ../standards/standards.json  exact standard wording cited by the audits, with URLs (dedupe by source)
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["area", "question", "why", "standard", "pass", "population", "sample", "data_source", "template",
        "timeline", "change", "target", "reaudit", "close_loop", "evidence", "novelty", "effort"]
# Near-duplicates across parts found on review; the version listed in the comment is kept.
DROP = [
    "In adults with a new spinal column fracture confirmed on imaging, was the rest of the spine imaged",  # trauma whole-spine
    "In adults with a spinal fracture managed non-operatively, did the notes state",                     # trauma stability/orthosis
    "In adults with an ankle fracture of uncertain stability treated in a cast or boot",                 # trauma weight-bearing X-ray
    "What proportion of systemically well patients with suspected early fracture-related infection",     # trauma FRI consultant review
    "In adults having shoulder replacement under general anaesthesia for over 90 minutes",               # periop upper-limb VTE
    "In adults transfused after primary joint replacement, what proportion have a discharge summary",    # periop transfusion letter
    "In adults with diabetes listed for elective hip or knee replacement, was an HbA1c",                 # elective HbA1c
    "What proportion of adults with suspected non-variceal upper GI bleedin",  # non-ortho near-duplicate
    "What proportion of adults with a new melanoma diagnosis have a vitamin",  # non-ortho near-duplicate
    "In non-bleeding adults in intensive care, what proportion of red cell ",  # non-ortho near-duplicate
    "What proportion of adult medical inpatients at medium risk of sepsis (",  # non-ortho near-duplicate
    "What proportion of adults with confirmed C. difficile infection have t",  # non-ortho near-duplicate
    "What proportion of adults with suspected TIA seen in the ED receive as",  # non-ortho near-duplicate
    "What proportion of infants with bronchiolitis seen in the ED receive n",  # non-ortho near-duplicate
    "What proportion of febrile infants under 3 months seen in the ED have ",  # non-ortho near-duplicate
    "What proportion of adults discharged from the ED or ambulatory care wi",  # non-ortho near-duplicate
    "What proportion of adults given insulin-glucose for hyperkalaemia in t",  # non-ortho near-duplicate
]
ORDER = ["hip", "trauma", "paeds", "limbs_spine", "elective", "periop", "outpatients", "generated"]  # generated = added from live answers; always last so earlier ids never shift


# set name -> (parts folder, id prefix, output json, catalogue, title, area order)
SETS = {
    "ortho": ("parts", "ONA", "ortho_new_audits.json", "catalogue.md", "orthopaedic", None),
    "nonortho": ("nonortho_parts", "NNA", "nonortho_new_audits.json", "catalogue_nonortho.md", "non-orthopaedic",
                 ["surgery", "surgical_specialties", "anaesthesia_icu", "emergency", "medicine", "elderly_neuro_palliative",
                  "psychiatry", "paediatrics", "obs_gynae_sexual", "diagnostics_prescribing", "generated"]),
}


def main():
    std = {}
    for name in SETS:
        build(name, std)
    os.makedirs(os.path.join(HERE, "..", "standards"), exist_ok=True)
    json.dump(sorted(std.values(), key=lambda x: x["source"]),
              open(os.path.join(HERE, "..", "standards", "standards.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"{len(std)} standards in standards/standards.json")


def build(name, std):
    folder, prefix, out_json, catalogue, title, order = SETS[name]
    order = order or ORDER
    audits, problems = [], []
    files = sorted(glob.glob(os.path.join(HERE, folder, "*.json")),
                   key=lambda f: order.index(os.path.basename(f)[:-5]) if os.path.basename(f)[:-5] in order else 99)
    if not files:
        return
    for f in files:
        for a in json.load(open(f, encoding="utf-8")):
            missing = [k for k in KEYS if k not in a]
            if missing:
                problems.append((os.path.basename(f), a.get("question", "?")[:60], missing))
                continue
            audits.append(a)
    seen, unique = set(), []
    dropped = [a for a in audits if any(a["question"].startswith(d) for d in DROP)]
    audits = [a for a in audits if a not in dropped]
    for a in audits:                      # drop near-identical questions across parts
        k = re.sub(r"[^a-z0-9]+", " ", a["question"].lower()).strip()
        if k in seen:
            continue
        seen.add(k)
        unique.append(a)
    # Ids are permanent: id_map_<set>.json maps each question to its id; new questions get the next number.
    map_path = os.path.join(HERE, f"id_map_{name}.json")
    idmap = json.load(open(map_path, encoding="utf-8")) if os.path.exists(map_path) else None
    if idmap is None:                      # first run: seed from the ids already published
        idmap = {}
        if os.path.exists(os.path.join(HERE, out_json)):
            for old in json.load(open(os.path.join(HERE, out_json), encoding="utf-8")):
                idmap[re.sub(r"[^a-z0-9]+", " ", old["question"].lower()).strip()] = old["id"]
    used = set(idmap.values())
    nxt = max([int(v.split("-")[1]) for v in used] + [0]) + 1
    for a in unique:
        k = re.sub(r"[^a-z0-9]+", " ", a["question"].lower()).strip()
        if k not in idmap:
            idmap[k] = f"{prefix}-{nxt:03d}"; nxt += 1
        a["id"] = idmap[k]
    unique.sort(key=lambda a: int(a["id"].split("-")[1]))
    json.dump(idmap, open(map_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    fb_path = os.path.join(HERE, "feedback.json")      # thumbs up/down collected from the app and reviewers
    fb = json.load(open(fb_path, encoding="utf-8")) if os.path.exists(fb_path) else []
    for a in unique:
        mine = [f for f in fb if f.get("id") == a["id"]]
        if mine:
            reasons = {}
            for f in mine:
                for r in f.get("reasons") or []:
                    reasons[r] = reasons.get(r, 0) + 1
            a["feedback"] = {"up": sum(f.get("rating") == "up" for f in mine),
                             "down": sum(f.get("rating") == "down" for f in mine), "reasons": reasons}
        else:
            a.pop("feedback", None)
    os.makedirs(os.path.join(HERE, "templates"), exist_ok=True)
    for a in unique:
        path = os.path.join(HERE, "templates", f"{a['id']}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([t["field"] for t in a["template"]])
            w.writerow([t["type"] + (": " + " / ".join(t.get("options") or []) if t.get("options") else "")
                        + (f" ({t['note']})" if t.get("note") else "") for t in a["template"]])
        a["template_file"] = f"new_audits/templates/{a['id']}.csv"
    json.dump(unique, open(os.path.join(HERE, out_json), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, catalogue), "w", encoding="utf-8") as fh:
        fh.write(f"# Proposed {title} audits ({len(unique)})\n\nDesigned from current standards and gaps in "
                 "the corpus. Each has a data template in `templates/`. Not yet run anywhere.\n")
        area = None
        for a in unique:
            if a["area"] != area:
                area = a["area"]
                fh.write(f"\n## {area}\n")
            s = a["standard"]
            fh.write(f"\n**{a['id']}. {a['question']}** ({a['novelty']})\n"
                     f"- Standard: {s.get('source')}: \"{s.get('wording')}\" {s.get('url')}\n"
                     f"- Pass: {a['pass']}. Target: {a['target']}. Sample: {a['sample']}\n"
                     f"- Change: {a['change']}\n- Template: `{a['template_file']}`\n")
            for k in ("pitfalls", "pearls"):
                if a.get(k):
                    fh.write(f"- {k.capitalize()}: " + "; ".join(a[k]) + "\n")
    for a in unique:
        s = a["standard"]
        k = (s.get("source") or "").strip()
        e = std.setdefault(k, {"source": k, "wording": s.get("wording"), "url": s.get("url"), "used_by": []})
        e["used_by"].append(a["id"])
    by_area = {}
    for a in unique:
        by_area[a["area"]] = by_area.get(a["area"], 0) + 1
    print(f"{name}: {len(unique)} audits ({len(audits) - len(unique) + len(dropped)} duplicates dropped); by area {by_area}")
    for p in problems:
        print("  skipped (missing keys):", p)


if __name__ == "__main__":
    main()
