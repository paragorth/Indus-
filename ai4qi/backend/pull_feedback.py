#!/usr/bin/env python3
"""Pull feedback on proposed audits from Supabase into ai4qi/new_audits/feedback.json.

Usage (from anywhere):
    export SUPABASE_URL=https://<project-ref>.supabase.co
    export SUPABASE_SERVICE_KEY=<service_role or secret key>   # never commit this key
    python3 ai4qi/backend/pull_feedback.py [--out PATH] [--dry-run]

Reads every row of public.feedback with the service key (which bypasses row level security),
converts each to the feedback.json format

    {"id": "<audit id>", "rating": "up"|"down", "reasons": [...], "comment": "...",
     "source": "app", "at": "<ISO timestamp>", "feedback_id": "<row uuid>"}

and merges them into the existing file. Entries already in the file (including hand-entered ones
without a feedback_id) are kept unchanged; rows are de-duplicated by their Supabase row id, so the
script can be re-run safely. Device ids and user ids are not copied.

Standard library only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "new_audits" / "feedback.json"
PAGE = 1000
COLUMNS = "id,audit_id,rating,reasons,comment,created_at"


def fetch_all(base, key):
    headers = {"apikey": key, "Accept": "application/json"}
    if key.startswith("eyJ"):  # legacy JWT service_role key; new sb_secret_ keys go in apikey only
        headers["Authorization"] = "Bearer " + key
    rows, start = [], 0
    while True:
        q = urllib.parse.urlencode({"select": COLUMNS, "order": "created_at.asc,id.asc"})
        req = urllib.request.Request(f"{base}/rest/v1/feedback?{q}", headers={
            **headers, "Range-Unit": "items", "Range": f"{start}-{start + PAGE - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                batch = json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            sys.exit(f"Supabase answered HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach Supabase: {e.reason}")
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        start += PAGE


def to_entry(row):
    return {
        "id": row["audit_id"],
        "rating": row["rating"],
        "reasons": list(row.get("reasons") or []),
        "comment": row.get("comment") or "",
        "source": "app",
        "at": row["created_at"],
        "feedback_id": row["id"],
    }


def merge(existing, rows):
    seen = {e.get("feedback_id") for e in existing if isinstance(e, dict) and e.get("feedback_id")}
    added = []
    for row in rows:
        if row.get("id") in seen:
            continue
        seen.add(row["id"])
        added.append(to_entry(row))
    return existing + added, added


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"feedback file (default {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true", help="report what would be added without writing")
    a = ap.parse_args()

    base = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not base or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment (the key is never read from a file).")
    if not (base.startswith("https://") or base.startswith(("http://localhost", "http://127.0.0.1"))):
        sys.exit("SUPABASE_URL must start with https:// (plain http is allowed only for a local test stack).")

    existing = []
    if a.out.is_file():
        existing = json.loads(a.out.read_text(encoding="utf-8") or "[]")
        if not isinstance(existing, list):
            sys.exit(f"{a.out} must hold a JSON list")

    rows = fetch_all(base, key)
    merged, added = merge(existing, rows)
    print(f"fetched {len(rows)} rows from Supabase; {len(added)} new; file will hold {len(merged)} entries")
    if a.dry_run or not added:
        return
    lines = ",\n".join(" " + json.dumps(e, ensure_ascii=False) for e in merged)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text("[\n" + lines + "\n]\n", encoding="utf-8")
    tmp.replace(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
