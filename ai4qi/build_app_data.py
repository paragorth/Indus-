#!/usr/bin/env python3
"""Build the compact data files used by the web app in app/.

Reads (never writes) ai4qi-library.json, new_audits/*_new_audits.json,
new_audits/templates/*.csv, standards/standards.json and figures/.
Writes app/data/*.json, app/data/audits/*.json, app/templates/*.csv, app/figures/.

Standard library only. Re-runnable: output folders are rebuilt each time.

    python3 build_app_data.py
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
DATA = APP / "data"
SHARD = 100  # published audits per detail file

OPEN_LICENCES = {"cc by", "cc0"}
UK_WORDS = re.compile(r"\b(UK|United Kingdom|England|Scotland|Wales|Northern Ireland|Ireland|Irish|British)\b", re.I)
UK_HOSTS = re.compile(r"(\.nhs\.uk|\.nhs\.scot|\.hse\.ie|\.ac\.uk|\.gov\.uk|\.org\.uk|\.co\.uk|\.ie)(/|$)", re.I)

# Editorial wording for the product: the site calls the collection "the library".
WORDING = [
    (re.compile(r"\bin the corpus\b"), "in the library"),
    (re.compile(r"\bin corpus\b"), "in the library"),
    (re.compile(r"\bThe corpus\b"), "The library"),
    (re.compile(r"\bthe corpus\b"), "the library"),
    (re.compile(r"\bcorpus\b"), "library"),
]


def tidy(text):
    if not isinstance(text, str):
        return text
    for pat, rep in WORDING:
        text = pat.sub(rep, text)
    return text


def tidy_all(obj):
    if isinstance(obj, str):
        return tidy(obj)
    if isinstance(obj, list):
        return [tidy_all(x) for x in obj]
    if isinstance(obj, dict):
        return {k: tidy_all(v) for k, v in obj.items()}
    return obj


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))


def clean(value):
    """Drop empty values so the files stay small."""
    if isinstance(value, dict):
        out = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [clean(v) for v in value if v not in (None, "", [], {})]
    return value


def is_uk(a):
    paper = a.get("paper") or {}
    if paper.get("uk_ireland") is True:
        return True
    if paper.get("uk_ireland") is False:
        return False
    country = str((a.get("detail") or {}).get("country") or "")
    if country and country != "not reported":
        return bool(UK_WORDS.search(country))
    src = str(a.get("source") or "")
    host = re.sub(r"^https?://", "", src).split("/")[0]
    return bool(host and UK_HOSTS.search(host + "/"))


def loop_closed(detail):
    v = (detail or {}).get("loop_closed")
    if v is True or (isinstance(v, str) and v.lower().startswith("yes")):
        return 1
    return 0


def spec_group(s):
    return (s or "Other").split(" – ")[0].strip()


def build_library(lib):
    index, shards = [], {}
    figures_needed = set()
    for a in lib["audits"]:
        d = a.get("detail") or {}
        p = a.get("paper") or {}
        c1 = d.get("cycle1") or {}
        c2 = d.get("cycle2") or {}
        country = d.get("country") if d.get("country") not in (None, "not reported") else ""
        if not country and p.get("country_from_affiliation"):
            country = p["country_from_affiliation"]
        row = {
            "id": a["id"],
            "t": a.get("title"),
            "s": a.get("specialty"),
            "g": spec_group(a.get("specialty")),
            "tp": a.get("topic"),
            "st": a.get("standard"),
            "f": a.get("finding"),
            "c": a.get("change"),
            "r1": c1.get("result") if c1.get("result") != a.get("finding") else "",
            "iv": d.get("intervention") if d.get("intervention") != a.get("change") else "",
            "r2": c2.get("result"),
            "co": country,
            "y": p.get("year"),
            "uk": 1 if is_uk(a) else 0,
            "lc": loop_closed(d),
            "dt": 1 if d else 0,
            "fx": d.get("fix_type") if d.get("fix_type") not in ("not reported", "unclassified") else "",
            "ss": a.get("status"),
        }
        index.append(clean(row))

        full = {k: a.get(k) for k in ("id", "title", "specialty", "topic", "status", "standard",
                                       "finding", "change", "next_audit", "source", "figure_link")}
        if d:
            full["detail"] = {k: v for k, v in d.items() if k not in ("fix_type_note",)}
        if p:
            keep = ("authors", "journal", "year", "doi", "pmid", "pmcid", "licence", "uk_ireland", "citation", "title")
            full["paper"] = {k: p.get(k) for k in keep}
            if str(p.get("licence") or "").lower() in OPEN_LICENCES:
                full["paper"]["abstract"] = p.get("abstract")
        if a.get("figures"):
            full["figures"] = [{k: f.get(k) for k in ("label", "caption", "credit", "licence", "source_url", "file")}
                               for f in a["figures"]]
            figures_needed.update(f["file"] for f in a["figures"] if f.get("file"))
        shards.setdefault(a["id"] // SHARD, []).append(clean(full))
    return index, shards, figures_needed


def build_proposed():
    out = []
    for fname, group in (("ortho_new_audits.json", "Orthopaedics"), ("nonortho_new_audits.json", "")):
        for p in json.load(open(HERE / "new_audits" / fname, encoding="utf-8")):
            q = tidy_all(dict(p))
            # Evidence lines are shown only when they cite a library entry.
            q["evidence"] = [e for e in q.get("evidence") or [] if re.search(r"\[\d+", e)]
            q["group"] = group or q.get("area")
            q["template_file"] = "templates/" + Path(p.get("template_file") or (p["id"] + ".csv")).name
            out.append(clean(q))
    return out


def build_standards(proposed):
    stds = json.load(open(HERE / "standards" / "standards.json", encoding="utf-8"))
    by_key = {}
    for s in stds:
        s = dict(s)
        s["used_by"] = list(dict.fromkeys(s.get("used_by") or []))
        by_key.setdefault((s.get("wording") or "").strip(), s)
    result = list(by_key.values())
    for p in proposed:
        st = p.get("standard") or {}
        key = (st.get("wording") or "").strip()
        if not key:
            continue
        s = by_key.get(key)
        if s is None:
            s = {"source": st.get("source"), "wording": st.get("wording"), "url": st.get("url", ""), "used_by": []}
            by_key[key] = s
            result.append(s)
        if p["id"] not in s["used_by"]:
            s["used_by"].append(p["id"])
    result.sort(key=lambda s: (s.get("source") or "").lower())
    return [clean(s) for s in result]


def build_cards(lib):
    cards = []
    for c in lib["topic_knowledge"]:
        card = {k: c.get(k) for k in ("topic", "audits_in_library", "countries", "years_seen", "usual_baseline",
                                      "fix_that_works", "fix_that_fails", "consultant_advice", "evidence_ids",
                                      "status", "revises_seed_card")}
        card["draft"] = 1 if c.get("status") else 0
        cards.append(clean(card))
    return cards


def main():
    lib = json.load(open(HERE / "ai4qi-library.json", encoding="utf-8"))

    for sub in ("data", "templates", "figures"):
        shutil.rmtree(APP / sub, ignore_errors=True)

    index, shards, figs = build_library(lib)
    proposed = build_proposed()
    standards = build_standards(proposed)
    cards = build_cards(lib)

    dump(DATA / "library.json", {"shard": SHARD, "audits": index})
    for key, rows in shards.items():
        dump(DATA / "audits" / f"{key}.json", rows)
    dump(DATA / "proposed.json", proposed)
    dump(DATA / "cards.json", cards)
    dump(DATA / "standards.json", standards)

    (APP / "templates").mkdir(parents=True, exist_ok=True)
    for csv in sorted((HERE / "new_audits" / "templates").glob("*.csv")):
        shutil.copy2(csv, APP / "templates" / csv.name)

    copied = 0
    for rel in sorted(figs):
        src = HERE / rel
        if src.is_file():
            dst = APP / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"library index: {len(index)} audits, {len(shards)} detail files")
    print(f"proposed: {len(proposed)}  cards: {len(cards)}  standards: {len(standards)}")
    print(f"templates: {len(list((APP / 'templates').glob('*.csv')))}  figures copied: {copied}")
    print(f"app/data total: {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
