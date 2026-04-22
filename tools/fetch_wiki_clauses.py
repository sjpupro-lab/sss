#!/usr/bin/env python3
"""Fetch Wikipedia intro paragraphs via REST API, split into clauses,
write a corpus file of at least N usable clauses.

Each line in output = 1 clause (>=10 UTF-8 bytes, no leading '<').
Uses urllib only (stdlib).
"""
import argparse, json, re, sys, time, urllib.request, urllib.parse

API = "https://en.wikipedia.org/w/api.php"
UA = "CANVAS-bench/0.1 (https://github.com/sjpupro-lab/CANVAS)"

def api_get(params):
    params = dict(params)
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def random_titles(n):
    """Return up to n random article titles (batches of 10)."""
    titles = []
    while len(titles) < n:
        data = api_get({"action":"query","list":"random","rnnamespace":0,"rnlimit":10})
        for p in data.get("query",{}).get("random",[]):
            titles.append(p["title"])
    return titles[:n]

def fetch_extracts(titles):
    """Fetch plaintext extracts in batches of 20 titles."""
    out = {}
    B = 20
    for i in range(0, len(titles), B):
        chunk = titles[i:i+B]
        data = api_get({
            "action":"query","prop":"extracts","exintro":0,"explaintext":1,
            "titles":"|".join(chunk),"redirects":1,
        })
        pages = data.get("query",{}).get("pages",{})
        for _, p in pages.items():
            t = p.get("title","")
            ex = p.get("extract","") or ""
            if ex:
                out[t] = ex
    return out

# split into clauses/sentences
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
def to_clauses(text):
    for s in SENT_SPLIT.split(text):
        s = s.strip()
        if len(s.encode("utf-8")) >= 10 and not s.startswith("<"):
            yield s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=5000)
    args = ap.parse_args()

    have = 0
    t0 = time.time()
    seen = set()
    with open(args.out, "w", encoding="utf-8") as f:
        while have < args.target:
            batch = random_titles(40)
            extracts = fetch_extracts(batch)
            for title, ex in extracts.items():
                for cl in to_clauses(ex):
                    if cl in seen: continue
                    seen.add(cl)
                    f.write(cl + "\n")
                    have += 1
                    if have >= args.target: break
                if have >= args.target: break
            print(f"  progress: {have}/{args.target}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"wrote {have} clauses to {args.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
