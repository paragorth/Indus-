"""Ai4Qi audit library pipeline.

    python3 pipeline.py search  --pass ortho|nonortho
    python3 pipeline.py fetch   --pass ortho|nonortho   # metadata, licence, OA flag, full text
    python3 pipeline.py figures --pass ortho|nonortho   # CC BY / CC0 only
    python3 pipeline.py extract --pass ortho|nonortho   # Anthropic API (needs credentials)
    python3 pipeline.py merge                           # dedupe + number into the library
    python3 pipeline.py topics                          # group + draft topic cards (API)
    python3 pipeline.py output                          # JSON, CSV, HTML
    python3 pipeline.py report                          # counts and failures

Every stage caches under work/ and can be re-run; nothing already fetched is
fetched again.  Seed entries (ids 1-1145) are never rewritten.
"""
import argparse
import collections
import csv
import difflib
import glob
import gzip
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zlib

import classify
import sources
import specialties
from queries import broad_query_set, query_set

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sources.WORK
FT_DIR = os.path.join(WORK, "fulltext")
FIG_DIR = os.path.join(HERE, "figures")
LIB = os.path.join(HERE, "ai4qi-library.json")
SEED = os.path.join(HERE, "seed", "ai4qi-library.seed.json")
os.makedirs(FT_DIR, exist_ok=True)


GZ_TRACKED = {"records.json"}
GZ_SHARDS = 6   # large caches kept in git as .json.gz (plain .json is git-ignored)


def load(name, default):
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    shards = sorted(glob.glob(p[:-5] + ".part*.json.gz"))
    if shards:                                   # large caches are split so each git file stays under 50 MB
        out = {}
        for s in shards:
            with gzip.open(s, "rt", encoding="utf-8") as f:
                out.update(json.load(f))
        return out
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return default


def save(name, obj):
    p = os.path.join(WORK, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    if name in GZ_TRACKED:
        parts = [dict() for _ in range(GZ_SHARDS)]
        for k, v in obj.items():                 # stable shard per key, so git diffs stay small
            parts[zlib.crc32(k.encode()) % GZ_SHARDS][k] = v
        for i, part in enumerate(parts):
            sp = f"{p[:-5]}.part{i:02d}.json.gz"
            with gzip.open(sp + ".tmp", "wt", encoding="utf-8", compresslevel=6) as f:
                json.dump(part, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(sp + ".tmp", sp)
        if os.path.exists(p + ".gz"):
            os.remove(p + ".gz")


# ------------------------------------------------------------------ SEARCH

def cmd_search(pas):
    hits = load(f"hits_{pas}.json", {})
    epmc_up = sources.epmc_available()
    counts = {}
    if not epmc_up:
        print("Europe PMC unavailable now; searching PubMed only (re-run later to add Europe PMC).")
    epmc_cache = load("epmc_core.json", {})
    qs = query_set(pas) + broad_query_set(pas)
    if os.environ.get("AI4QI_SWEEP") and pas == "nonortho":
        from queries import sweep_query_set
        qs = sweep_query_set()          # sweep only; hits accumulate into the existing file
        if os.environ["AI4QI_SWEEP"] == "adherence":
            from queries import adherence_query_set
            qs = adherence_query_set()
        if os.environ["AI4QI_SWEEP"] == "thin":
            from queries import thin_query_set
            qs = thin_query_set()
        if os.environ["AI4QI_SWEEP"] in ("title", "title_early"):
            from queries import title_sweep_query_set
            qs = title_sweep_query_set()
    for lab, pm_q, ep_q in qs:
        pm = (sources.pubmed_search(pm_q) or []) if pm_q else []
        ep = (sources.epmc_search(ep_q) or []) if epmc_up else []
        for pmid in pm:
            h = hits.setdefault(f"PMID:{pmid}", {"pmid": pmid, "labels": [], "engines": []})
            _add(h, lab, "pubmed")
        for r in ep:
            key = f"PMID:{r['pmid']}" if r.get("pmid") else f"EPMC:{r.get('source')}:{r.get('id')}"
            h = hits.setdefault(key, {"pmid": r.get("pmid", ""), "labels": [], "engines": []})
            _add(h, lab, "europepmc")
            epmc_cache[key] = r
        counts[lab] = {"pubmed": len(pm), "europepmc": len(ep)}
        print(f"  {lab:28s} PubMed {len(pm):5d}   Europe PMC {len(ep):5d}")
    save(f"hits_{pas}.json", hits)
    save("epmc_core.json", epmc_cache)
    meta = load("run_meta.json", {})
    meta[f"search_{pas}"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "counts": counts,
                             "europepmc_available": epmc_up, "unique_hits": len(hits)}
    save("run_meta.json", meta)
    print(f"{pas}: {len(hits)} unique hits")


def _add(h, lab, eng):
    if lab not in h["labels"]:
        h["labels"].append(lab)
    if eng not in h["engines"]:
        h["engines"].append(eng)


# ------------------------------------------------------------------ FETCH

def _from_epmc(r):
    ji = r.get("journalInfo") or {}
    j = ji.get("journal") or {}
    authors, affs = [], []
    for a in (r.get("authorList") or {}).get("author", []):
        authors.append(a.get("fullName") or a.get("collectiveName") or "")
        for x in (a.get("authorAffiliationDetailsList") or {}).get("authorAffiliation", []):
            affs.append(x.get("affiliation", ""))
    if r.get("affiliation"):
        affs.append(r["affiliation"])
    abst = re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")
    return {
        "pmid": r.get("pmid", ""), "pmcid": r.get("pmcid", ""), "doi": (r.get("doi") or "").lower(),
        "title": re.sub(r"<[^>]+>", "", r.get("title") or "").strip(), "authors": authors,
        "journal": j.get("title", ""), "journal_abbrev": j.get("isoabbreviation", ""),
        "year": str(r.get("pubYear") or ji.get("yearOfPublication") or ""),
        "volume": ji.get("volume", ""), "issue": ji.get("issue", ""), "pages": r.get("pageInfo", ""),
        "abstract": re.sub(r"\s+", " ", abst).strip(),
        "pub_types": (r.get("pubTypeList") or {}).get("pubType", []),
        "affiliations": list(dict.fromkeys(x for x in affs if x)),
        "licence": (r.get("license") or "").lower(), "open_access": r.get("isOpenAccess") == "Y",
        "epmc_id": f"{r.get('source')}:{r.get('id')}",
    }


def cmd_fetch(pas):
    hits = load(f"hits_{pas}.json", {})
    recs = load("records.json", {})
    epmc_cache = load("epmc_core.json", {})
    epmc_up = sources.epmc_available()
    todo = [k for k in hits if k not in recs]
    # 1. Europe PMC core record (has licence + OA flag); fill gaps from PubMed.
    need = [hits[k]["pmid"] for k in todo if k not in epmc_cache and hits[k]["pmid"]]
    if need and epmc_up:
        got = sources.epmc_by_pmids(need)
        for pmid, r in got.items():
            epmc_cache[f"PMID:{pmid}"] = r
        save("epmc_core.json", epmc_cache)
    missing = [hits[k]["pmid"] for k in todo if k not in epmc_cache and hits[k]["pmid"]]
    pm = sources.pubmed_fetch(missing) if missing else {}
    for k in todo:
        if k in epmc_cache:
            rec = _from_epmc(epmc_cache[k])
            rec["metadata_source"] = "Europe PMC"
        elif hits[k]["pmid"] in pm:
            rec = pm[hits[k]["pmid"]]
            rec.update(licence="", open_access=False, metadata_source="PubMed")
        else:
            sources.log_failure("metadata", k, "no record from Europe PMC or PubMed")
            continue
        recs[k] = rec
    # 2. PubMed publication types for Europe PMC records lacking them is not needed;
    #    licence for PMC articles comes from the PMC open-access bucket when EPMC has none.
    for k in hits:
        r = recs.get(k)
        if not r or r.get("s3_checked"):
            continue
        if r.get("pmcid") and (not r.get("licence") or r["metadata_source"] == "PubMed"):
            m = sources.pmc_s3_meta(r["pmcid"])
            if m:
                r["licence"] = r.get("licence") or (m.get("license_code") or "").lower()
                r["open_access"] = r.get("open_access") or bool(m.get("is_pmc_openaccess"))
                r["s3_xml"] = m.get("xml_http")
        r["s3_checked"] = True
    save("records.json", recs)
    # 3. screen (open-access papers whose abstract lacks audit wording are re-screened on full text)
    scr = load("screen.json", {})
    for k in hits:
        if pas == "nonortho" and scr.get(k, {}).get("pass") == "ortho" and scr[k]["keep"]:
            continue            # already in the orthopaedic library
        if k in recs:
            if os.environ.get("AI4QI_NO_FT") and k in scr:
                continue              # quick re-run: keep earlier decisions (some came from full text)
            keep, why, kind = classify.screen(recs[k])
            if not keep and why.startswith("excluded: no audit wording") and recs[k].get("pmcid") \
                    and recs[k].get("open_access") and not os.environ.get("AI4QI_NO_FT"):
                ft = _fulltext(recs[k], epmc_up)
                if ft:
                    import extract
                    keep, why, kind = classify.screen(recs[k], extract.jats_text(ft))
            if keep and pas == "ortho" and not classify.is_ortho(recs[k]):
                keep, why = False, "excluded: no orthopaedic wording (ortho pass)"
            scr[k] = {"keep": keep, "reason": why, "audit_kind": kind, "pass": pas}
    save("screen.json", scr)
    # 4. full text for kept open-access papers
    n_ft = 0
    for k, s in scr.items():
        r = recs.get(k)
        if os.environ.get("AI4QI_NO_FT") or not (s["keep"] and s["pass"] == pas and r and r.get("pmcid") and r.get("open_access")):
            continue
        if _fulltext(r, epmc_up):
            n_ft += 1
    save("records.json", recs)
    kept = sum(1 for s in scr.values() if s["keep"] and s["pass"] == pas)
    print(f"{pas}: {len(todo)} new records fetched, {kept} kept after screening, {n_ft} with full text")


def _fulltext(r, epmc_up):
    """Path to cached JATS XML, fetching from Europe PMC (then the PMC bucket) if needed."""
    path = os.path.join(FT_DIR, f"{r['pmcid']}.xml")
    if os.path.exists(path):
        return path
    if r.get("fulltext_failed"):
        return None
    xml = sources.epmc_fulltext(r["pmcid"]) if epmc_up else None
    src = "Europe PMC"
    if not xml:
        if not r.get("s3_xml"):
            m = sources.pmc_s3_meta(r["pmcid"])
            r["s3_xml"] = m and m.get("xml_http")
        xml = sources.s3_fetch(r["s3_xml"]) if r.get("s3_xml") else None
        src = "PMC open-access bucket"
    if not xml:
        r["fulltext_failed"] = True
        sources.log_failure("fulltext", r["pmcid"], "open access but no XML from Europe PMC or PMC")
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    r["fulltext_source"] = src
    return path


def cmd_screen_crossref():
    """Screen Crossref conference-abstract hits; each kept paper goes to the orthopaedic or
    non-orthopaedic library by content (there is no query set to decide it)."""
    hits, recs, scr = load("hits_crossref.json", {}), load("records.json", {}), load("screen.json", {})
    kept = collections.Counter()
    for k in hits:
        r = recs.get(k)
        if not r or (k in scr and scr[k].get("source") != "crossref"):
            continue
        keep, why, kind = classify.screen(r)
        pas = "ortho" if classify.is_ortho(r) else "nonortho"
        scr[k] = {"keep": keep, "reason": why, "audit_kind": kind, "pass": pas, "source": "crossref"}
        kept[pas] += keep
    save("screen.json", scr)
    print(f"crossref screened {len(hits)}: kept {dict(kept)}")


def cmd_rescreen(pas):
    """Re-apply the screening rules offline (cached metadata and full text only)."""
    hits, recs, scr = load(f"hits_{pas}.json", {}), load("records.json", {}), load("screen.json", {})
    import extract
    for k in hits:
        r = recs.get(k)
        if not r:
            continue
        keep, why, kind = classify.screen(r)
        ft = os.path.join(FT_DIR, f"{r.get('pmcid')}.xml")
        if not keep and why.startswith("excluded: no audit wording") and r.get("pmcid") and os.path.exists(ft):
            keep, why, kind = classify.screen(r, extract.jats_text(ft))
        if keep and pas == "ortho" and not classify.is_ortho(r):
            keep, why = False, "excluded: no orthopaedic wording (ortho pass)"
        if pas == "nonortho" and scr.get(k, {}).get("pass") == "ortho" and scr[k]["keep"]:
            continue
        scr[k] = {"keep": keep, "reason": why, "audit_kind": kind, "pass": pas}
    save("screen.json", scr)
    print(f"{pas}: rescreened; kept {sum(1 for v in scr.values() if v['pass'] == pas and v['keep'])}")


# ------------------------------------------------------------------ FIGURES

CC_OK = {"cc by", "cc-by", "cc0", "cc by 4.0", "cc by 3.0", "cc by 2.0", "cc0 1.0"}
CHART = [("run chart", r"run chart|statistical process control|\bSPC\b|control chart"),
         ("bar chart", r"bar (?:chart|graph)|histogram|column chart"),
         ("line chart", r"line (?:chart|graph)|trend"),
         ("pie chart", r"pie chart"),
         ("chart", r"chart|graph|plot|comparison of|cycle 1|cycle 2|first cycle|second cycle|re-?audit|compliance"),
         ("flowchart or diagram", r"flow ?chart|flow diagram|driver diagram|fishbone|pathway|algorithm"),
         ("form or template", r"proforma|pro forma|template|checklist|sticker|form\b")]


def licence_ok(lic):
    lic = (lic or "").lower().strip()
    return lic in CC_OK or bool(re.fullmatch(r"cc[- ]by(?: \d\.\d)?|cc0(?: \d\.\d)?", lic))


def chart_type(caption):
    for name, rx in CHART:
        if re.search(rx, caption, re.I):
            return name
    return "other figure"


def credit_line(r, label):
    first = (r.get("authors") or ["Unknown"])[0]
    etal = " et al." if len(r.get("authors") or []) > 1 else ""
    lic = r.get("licence", "").upper().replace("CC BY", "CC BY")
    return (f"{label} from {first}{etal}, \"{r.get('title', '')}\", {r.get('journal', '')} "
            f"{r.get('year', '')}. doi:{r.get('doi', '') or 'n/a'}. Licensed under {lic}; "
            f"reproduced unchanged.")


def cmd_figures(pas):
    recs, scr = load("records.json", {}), load("screen.json", {})
    figs = load("figures.json", {})
    n_papers = n_imgs = 0
    for k, s in scr.items():
        r = recs.get(k)
        if not (s["keep"] and s["pass"] == pas and r and r.get("pmcid")):
            continue
        if k in figs:
            continue
        link = f"https://europepmc.org/article/PMC/{r['pmcid']}"
        if not licence_ok(r.get("licence")):
            figs[k] = {"downloaded": [], "figure_link": link,
                       "note": f"licence '{r.get('licence') or 'not reported'}': link only"}
            continue
        path = os.path.join(FT_DIR, f"{r['pmcid']}.xml")
        if not os.path.exists(path):
            figs[k] = {"downloaded": [], "figure_link": link, "note": "no full text XML"}
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            sources.log_failure("figure-xml", r["pmcid"], e)
            continue
        m = sources.pmc_s3_meta(r["pmcid"])
        media = {os.path.splitext(os.path.basename(u))[0]: u for u in (m or {}).get("media_http", [])}
        out = []
        for fig in root.iter("fig"):
            g = next((x for x in fig.iter() if x.tag == "graphic"), None)
            href = g.get("{http://www.w3.org/1999/xlink}href") if g is not None else None
            if not href:
                continue
            stem = os.path.splitext(href)[0]
            url = media.get(stem)
            cap = re.sub(r"\s+", " ", " ".join(fig.find("caption").itertext())).strip() \
                if fig.find("caption") is not None else ""
            lab = "".join(fig.find("label").itertext()).strip() if fig.find("label") is not None else "Figure"
            item = {"label": lab, "caption": cap, "chart_type": chart_type(f"{lab} {cap}"),
                    "licence": r.get("licence", "").upper(), "credit": credit_line(r, lab),
                    "source_url": link}
            if url:
                data = sources.s3_fetch(url, binary=True)
                if data:
                    d = os.path.join(FIG_DIR, r["pmcid"])
                    os.makedirs(d, exist_ok=True)
                    fn = os.path.basename(url)
                    with open(os.path.join(d, fn), "wb") as f:
                        f.write(data)
                    item["file"] = f"figures/{r['pmcid']}/{fn}"
                    n_imgs += 1
            else:
                sources.log_failure("figure", f"{r['pmcid']}/{href}", "image not in PMC open-access bucket")
            out.append(item)
        figs[k] = {"downloaded": out, "figure_link": link, "note": "CC BY/CC0: figures downloaded"}
        n_papers += 1
        if n_papers % 25 == 0:
            save("figures.json", figs)
    save("figures.json", figs)
    print(f"{pas}: figures downloaded from {n_papers} CC BY/CC0 papers, {n_imgs} image files")


# ------------------------------------------------------------------ MERGE

def norm_title(t):
    t = re.sub(r"<[^>]+>", "", t or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


SEED_ID_RX = [(re.compile(r"PMC(\d+)", re.I), "pmcid"), (re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)"), "pmid"),
              (re.compile(r"(10\.\d{4,9}/[^\s?#]+)"), "doi")]


def seed_ids(entry):
    ids = set()
    for field in ("source",):
        s = entry.get(field) or ""
        for rx, kind in SEED_ID_RX:
            for m in rx.findall(s):
                v = m if kind != "pmcid" else f"PMC{m}"
                ids.add(f"{kind}:{v.lower().rstrip('/')}")
    paper = entry.get("paper") or {}
    for kind in ("pmid", "pmcid", "doi"):
        if paper.get(kind):
            ids.add(f"{kind}:{str(paper[kind]).lower()}")
    return ids


def rec_ids(r):
    return {f"{k}:{str(r[k]).lower()}" for k in ("pmid", "pmcid", "doi") if r.get(k)}


def citation(r):
    au = r.get("authors") or []
    names = ", ".join(au[:6]) + (", et al." if len(au) > 6 else "")
    vol = r.get("volume", "")
    bits = f"{r.get('journal_abbrev') or r.get('journal', '')}. {r.get('year', '')}"
    if vol:
        bits += f";{vol}"
        if r.get("issue"):
            bits += f"({r['issue']})"
        if r.get("pages"):
            bits += f":{r['pages']}"
    doi = f" doi:{r['doi']}." if r.get("doi") else ""
    pmid = f" PMID: {r['pmid']}." if r.get("pmid") else ""
    return f"{names}. {r.get('title', '').rstrip('.')}. {bits}.{doi}{pmid}".strip()


def source_url(r):
    if r.get("pmcid"):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{r['pmcid']}/"
    if r.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/"
    if r.get("doi"):
        return f"https://doi.org/{r['doi']}"
    src, _, ident = (r.get("epmc_id") or "::").partition(":")
    return f"https://europepmc.org/article/{src}/{ident}"


def base_library():
    if os.path.exists(LIB):
        with open(LIB, encoding="utf-8") as f:
            return json.load(f)
    with open(SEED, encoding="utf-8") as f:
        return json.load(f)


def cmd_merge():
    """Rebuild the library from the seed every time, so numbering of new entries stays
    contiguous; pipeline-drafted topic cards are carried over from the current library."""
    with open(SEED, encoding="utf-8") as f:
        lib = json.load(f)
    old = {"audits": []}
    if os.path.exists(LIB):
        old = json.load(open(LIB, encoding="utf-8"))
        lib["topic_knowledge"] += [t for t in old.get("topic_knowledge", []) if t.get("generated_by") == "pipeline"]
    audits = lib["audits"]
    recs, scr = load("records.json", {}), load("screen.json", {})
    hits = {**load("hits_crossref.json", {}), **load("hits_nonortho.json", {}), **load("hits_ortho.json", {})}
    ext, figs = load("extractions.json", {}), load("figures.json", {})
    by_id = {}
    for e in audits:
        for i in seed_ids(e):
            by_id.setdefault(i, e)
    titles = {norm_title(e["title"]): e for e in audits if e.get("paper")}
    next_id = max(e["id"] for e in audits) + 1
    ekeys = {}                        # entry -> record keys folded into it (for permanent ids)
    added = matched_seed = dup = 0
    dedupe_log, excluded = [], []
    excl_ids, excl_titles = set(), set()
    for k, v in ext.items():          # papers read and judged not to be audits
        if (v.get("result") or {}).get("is_audit") == "no" and k in recs:
            excl_ids |= rec_ids(recs[k])
            excl_titles.add(norm_title(recs[k]["title"]))
    for k in sorted(scr, key=lambda k: (scr[k]["pass"] != "ortho", k)):
        s, r = scr[k], recs.get(k)
        if not (s["keep"] and r):
            continue
        x = (ext.get(k) or {}).get("result") or {}
        twin = not x and (rec_ids(r) & excl_ids or norm_title(r["title"]) in excl_titles)
        if x.get("is_audit") == "no" or twin:
            excluded.append({"key": k, "title": r["title"], "reason": "not an audit on reading"})
            continue
        if x and specialties.NOT_CLINICAL.search(x.get("specialty") or ""):
            excluded.append({"key": k, "title": r["title"], "reason": "not a human clinical audit"})
            continue
        if not x:                     # screened in but not read yet: wait for a reading pass
            continue
        if x.get("is_audit") == "unclear" and len(r.get("abstract") or "") < 200:
            excluded.append({"key": k, "title": r["title"], "reason": "title only; audit status unclear"})
            continue
        ids = rec_ids(r)
        hit = next((by_id[i] for i in ids if i in by_id), None)
        why = "identifier"
        if hit is None:
            nt = norm_title(r["title"])
            hit = titles.get(nt)
            why = "identical title"
            if hit is None and len(nt) > 30:
                close = difflib.get_close_matches(nt, list(titles), n=1, cutoff=0.93)
                if close:
                    hit, why = titles[close[0]], "near-identical title"
        entry = hit
        if entry is None:
            entry = {"id": next_id}
            next_id += 1
            audits.append(entry)
            added += 1
        elif entry["id"] <= 1145 and "paper" not in entry:
            matched_seed += 1
            dedupe_log.append({"key": k, "matched_id": entry["id"], "by": why})
        else:
            dup += 1
            dedupe_log.append({"key": k, "matched_id": entry["id"], "by": why})
        _fill(entry, k, r, s, hits.get(k, {}), ext.get(k), figs.get(k))
        ekeys.setdefault(id(entry), []).append(k)
        for i in ids:
            by_id[i] = entry
        titles[norm_title(r["title"])] = entry
    _assign_permanent_ids(audits, ekeys, old)
    mp = os.path.join(WORK, "topics_out", "mapping.json")
    if os.path.exists(mp):            # canonical topic names chosen in the topics step
        mapping = json.load(open(mp, encoding="utf-8"))
        for e in audits:
            if e["id"] > 1145 and e.get("topic") in mapping:
                e["topic_extracted"] = e["topic"]
                e["topic"] = mapping[e["topic"]]
    lib["audits"] = audits
    lib.setdefault("pipeline", {})["last_merge"] = time.strftime("%Y-%m-%d %H:%M")
    _write_lib(lib)
    save("dedupe_log.json", dedupe_log)
    save("excluded_on_reading.json", excluded)
    print(f"merge: {added} new entries, {matched_seed} seed entries enriched, {dup} duplicates folded; "
          f"{len(excluded)} excluded as not audits on reading; library now {len(audits)} entries")


def _assign_permanent_ids(audits, ekeys, old):
    """Ids above SEED_MAX never change once given: work/id_map.json maps record key -> id.
    New papers get the next unused number, so citations like [2124] stay valid for good."""
    idmap = load("id_map.json", None)
    if idmap is None:                 # first run: freeze the ids of the library as it stands
        idmap = {e["paper"]["key"]: e["id"] for e in old.get("audits", [])
                 if e["id"] > 1145 and (e.get("paper") or {}).get("key")}
    used = {e["id"] for e in audits if e["id"] <= 1145}
    nxt = max(list(idmap.values()) + [1145]) + 1
    for e in audits:
        if e["id"] <= 1145:
            continue
        ks = ekeys.get(id(e), []) or [(e.get("paper") or {}).get("key")]
        nid = next((idmap[k] for k in ks if k in idmap and idmap[k] not in used), None)
        if nid is None:
            nid, nxt = nxt, nxt + 1
        e["id"] = nid
        used.add(nid)
        for k in ks:
            if k:
                idmap.setdefault(k, nid)
    audits.sort(key=lambda e: e["id"])
    save("id_map.json", idmap)


def _fill(entry, key, r, s, hit, ext, fig):
    """New entries get the seed schema; seed entries only gain a 'paper' block (add-only)."""
    seed = entry["id"] <= 1145
    lib_name = "orthopaedic" if s["pass"] == "ortho" else "non-orthopaedic"
    if s["pass"] == "ortho":
        spec = "Orthopaedics – " + classify.ortho_subspecialty(r, hit.get("labels", [])).capitalize()
    else:
        spec = classify.nonortho_specialty(r, hit.get("labels", [])).capitalize()
        spec = {"Ent": "ENT"}.get(spec, spec)
    paper = {
        "key": key, "library": lib_name, "title": r["title"], "authors": r.get("authors", []),
        "journal": r.get("journal", ""), "year": r.get("year", ""), "doi": r.get("doi", ""),
        "pmid": r.get("pmid", ""), "pmcid": r.get("pmcid", ""), "abstract": r.get("abstract", ""),
        "licence": r.get("licence") or "not reported", "open_access": bool(r.get("open_access")),
        "full_text": bool(r.get("pmcid") and os.path.exists(os.path.join(FT_DIR, f"{r.get('pmcid')}.xml"))),
        "uk_ireland": classify.uk_ireland(r.get("affiliations")),
        "country_from_affiliation": classify.country_from_affiliations(r.get("affiliations")),
        "audit_kind_from_abstract": s["audit_kind"], "found_by": hit.get("engines", []),
        "search_labels": hit.get("labels", []), "metadata_source": r.get("metadata_source", ""),
        "citation": citation(r),
    }
    if seed:
        entry.setdefault("paper", paper)
        return
    entry.update({"specialty": spec, "title": r["title"], "standard": "", "finding": "", "change": "",
                  "source": source_url(r), "status": "published, pending extraction"})
    entry["paper"] = paper
    if fig:
        entry["figures"] = fig.get("downloaded", [])
        entry["figure_link"] = fig.get("figure_link")
    if ext and ext.get("result"):
        x = ext["result"]
        entry["status"] = "published, detailed" if x.get("is_audit") == "yes" else "published, audit status unclear"
        if x.get("specialty") and x["specialty"] != "not reported":
            entry["specialty"] = specialties.canonical(x["specialty"])
            # the reader's specialty decides the library: an obstetric audit found by an
            # orthopaedic query belongs in the non-orthopaedic library
            paper["library"] = "orthopaedic" if re.match(r"orthopaed", x["specialty"], re.I) else "non-orthopaedic"
        entry["topic"] = x.get("topic", "not reported")
        entry["standard"] = x.get("standard", "not reported")
        entry["finding"] = x.get("cycle1", {}).get("result", "not reported")
        entry["change"] = x.get("intervention", "not reported")
        entry["detail"] = {
            "setting": x.get("setting"), "country": norm_country(x.get("country")),
            "cycle1": {"period": x["cycle1"].get("dates"), "n": x["cycle1"].get("sample_size"),
                       "result": x["cycle1"].get("result")},
            "intervention": x.get("intervention"),
            "cycle2": {"period": x["cycle2"].get("dates"), "n": x["cycle2"].get("sample_size"),
                       "result": x["cycle2"].get("result")},
            "loop_closed": x.get("loop_closed"), "other_findings": x.get("other_findings"),
            "citation": paper["citation"], "fix_type": x.get("fix_type"),
            "fix_type_note": f"extracted by {ext.get('model')} from the {ext.get('input')}; check before use",
        }
        c = entry["detail"]["country"]
        if c and c != "not reported":        # the reader's country beats the affiliation regex
            paper["uk_ireland"] = c in ("UK", "Ireland")


COUNTRY_ALIASES = {"united kingdom": "UK", "england": "UK", "scotland": "UK", "wales": "UK",
                   "northern ireland": "UK", "great britain": "UK", "republic of ireland": "Ireland",
                   "united states": "USA", "united states of america": "USA"}


def norm_country(c):
    return COUNTRY_ALIASES.get((c or "").strip().lower(), c)


def _write_lib(lib):
    tmp = LIB + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LIB)


# ------------------------------------------------------------------ OUTPUT

CSV_COLS = ["id", "library", "specialty", "topic", "title", "standard", "finding", "change", "status",
            "source", "setting", "country", "uk_ireland", "cycle1_period", "cycle1_n", "cycle1_result",
            "intervention", "cycle2_period", "cycle2_n", "cycle2_result", "loop_closed", "fix_type",
            "other_findings", "next_audit", "authors", "journal", "year", "doi", "pmid", "pmcid",
            "licence", "open_access", "citation", "figures"]


def library_of(e):
    p = e.get("paper") or {}
    if p.get("library"):
        return p["library"]
    return "orthopaedic" if re.search(r"orthopaed", e.get("specialty", ""), re.I) else "non-orthopaedic"


def flat(e):
    d, p = e.get("detail") or {}, e.get("paper") or {}
    c1, c2 = d.get("cycle1") or {}, d.get("cycle2") or {}
    return {
        "id": e["id"], "library": library_of(e), "specialty": e.get("specialty", ""),
        "topic": e.get("topic", ""), "title": e.get("title", ""), "standard": e.get("standard", ""),
        "finding": e.get("finding", ""), "change": e.get("change", ""), "status": e.get("status", ""),
        "source": e.get("source") or "", "setting": d.get("setting", ""),
        "country": d.get("country") or p.get("country_from_affiliation", ""),
        "uk_ireland": p.get("uk_ireland", ""), "cycle1_period": c1.get("period", ""),
        "cycle1_n": c1.get("n", ""), "cycle1_result": c1.get("result", ""),
        "intervention": d.get("intervention", ""), "cycle2_period": c2.get("period", ""),
        "cycle2_n": c2.get("n", ""), "cycle2_result": c2.get("result", ""),
        "loop_closed": d.get("loop_closed", ""), "fix_type": d.get("fix_type", ""),
        "other_findings": d.get("other_findings", ""), "next_audit": e.get("next_audit", ""),
        "authors": "; ".join(p.get("authors", [])), "journal": p.get("journal", ""),
        "year": p.get("year", ""), "doi": p.get("doi", ""), "pmid": p.get("pmid", ""),
        "pmcid": p.get("pmcid", ""), "licence": p.get("licence", ""), "open_access": p.get("open_access", ""),
        "citation": d.get("citation") or p.get("citation", ""),
        "figures": " | ".join(f.get("file", "") for f in e.get("figures", []) if f.get("file")),
    }


def cmd_output():
    lib = base_library()
    with open(os.path.join(HERE, "ai4qi-library.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CSV_COLS)
        w.writeheader()
        for e in lib["audits"]:
            w.writerow(flat(e))
    import build_html
    build_html.build(lib, os.path.join(HERE, "index.html"))
    print(f"output: ai4qi-library.json ({len(lib['audits'])} entries), ai4qi-library.csv, index.html")


# ------------------------------------------------------------------ REPORT

def cmd_report():
    lib = base_library()
    scr, recs = load("screen.json", {}), load("records.json", {})
    meta = load("run_meta.json", {})
    for pas in ("ortho", "nonortho"):
        s = [v for v in scr.values() if v["pass"] == pas]
        if not s:
            continue
        kept = [k for k, v in scr.items() if v["pass"] == pas and v["keep"]]
        reasons = collections.Counter(v["reason"] for v in s)
        closed = sum(1 for k in kept if scr[k]["audit_kind"] == "closed-loop or re-audit")
        oa = sum(1 for k in kept if recs[k].get("open_access"))
        ccby = sum(1 for k in kept if licence_ok(recs[k].get("licence")))
        ukie = sum(1 for k in kept if classify.uk_ireland(recs[k].get("affiliations")))
        print(f"\n== {pas} pass ==")
        print(json.dumps(meta.get(f"search_{pas}", {}), indent=1))
        print(f"screened {len(s)}; kept {len(kept)} ({closed} closed-loop/re-audit wording, "
              f"{len(kept) - closed} audit/QI with loop unstated); UK/Ireland {ukie}; "
              f"open access {oa}; CC BY/CC0 {ccby}")
        for r, n in reasons.most_common():
            print(f"   {n:5d}  {r}")
    lib_counts = collections.Counter(library_of(e) for e in lib["audits"])
    print("\nlibrary entries by library:", dict(lib_counts))
    print("library entries by status:", dict(collections.Counter(e.get("status") for e in lib["audits"])))
    fails = []
    p = os.path.join(WORK, "failures.jsonl")
    if os.path.exists(p):
        fails = [json.loads(x) for x in open(p, encoding="utf-8")]
    print("\nfailures:", dict(collections.Counter((f["kind"], f["reason"]) for f in fails)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "fetch", "rescreen", "screen-crossref", "figures", "extract", "merge", "topics", "output", "report"])
    ap.add_argument("--pass", dest="pas", choices=["ortho", "nonortho"], default="ortho")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (half price)")
    a = ap.parse_args()
    if a.cmd == "search":
        cmd_search(a.pas)
    elif a.cmd == "fetch":
        cmd_fetch(a.pas)
    elif a.cmd == "screen-crossref":
        cmd_screen_crossref()
    elif a.cmd == "rescreen":
        cmd_rescreen(a.pas)
    elif a.cmd == "figures":
        cmd_figures(a.pas)
    elif a.cmd == "extract":
        import extract
        extract.run(a.pas, limit=a.limit, batch=a.batch)
    elif a.cmd == "merge":
        cmd_merge()
    elif a.cmd == "topics":
        import topics
        topics.run()
    elif a.cmd == "output":
        cmd_output()
    elif a.cmd == "report":
        cmd_report()


if __name__ == "__main__":
    sys.exit(main())
