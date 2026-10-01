"""Print the site ideas ("Suggest a change") for a review: python3 read_ideas.py [days]

Uses the public site_ideas() function (migration 011): idea text, part and date only.
"""
import json
import sys
import urllib.request

cfg = json.load(open(__file__.rsplit("/", 1)[0] + "/app/config.json"))
url, key = cfg["supabase_url"].rstrip("/"), cfg["supabase_anon_key"]
days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
req = urllib.request.Request(url + "/rest/v1/rpc/site_ideas", data=json.dumps({"p_days": days}).encode(),
                             headers={"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"})
rows = json.load(urllib.request.urlopen(req, timeout=30))
print(f"{len(rows)} ideas in the last {days} days")
for r in rows:
    print(f"{r['id']} | {r['created']} | {r['part']} | {r['idea']}")
