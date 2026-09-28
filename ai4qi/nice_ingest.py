"""Add recovered NICE Shared Learning case studies (retired NICE database, archived copies) as candidate
records: python3 nice_ingest.py <path to nice_shared_learning.json>. Screened and read like other hits."""
import json
import sys

from extra_harvest import Store, _record


def main(path):
    st = Store()
    data = json.load(open(path, encoding="utf-8"))
    for key, d in data.items():
        rec = _record(d.get("title"), d.get("abstract"), journal="NICE Shared Learning (local practice case study)",
                      year=d.get("year"), affs=[d.get("organisation") or ""], source="NICE Shared Learning")
        rec["pub_types"] = ["local practice case study (NICE Shared Learning)"]
        rec["source_url"] = d.get("source_url") or ""
        rec["archive_url"] = d.get("archive_url") or ""
        st.add(key, rec, "nice shared learning")
    st.save()
    print(f"nice shared learning: {st.new} new records")


if __name__ == "__main__":
    main(sys.argv[1])
