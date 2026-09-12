#!/usr/bin/env python3
"""Build BibTeX entries from arXiv metadata by ID (titles/authors verbatim from the API). Usage:
python3 analysis/ref_fetch.py key=arxivid[:venue note] ... > out.bib ; also writes a JSON record next to the bib."""
import sys, re, json, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
def fetch(ids):
    url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + f"&max_results={len(ids)}"
    xml = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ref-fetch/1.0"}), timeout=60).read().decode()
    ns = {"a": "http://www.w3.org/2005/Atom"}; out = {}
    for e in ET.fromstring(xml).findall("a:entry", ns):
        aid = e.findtext("a:id", "", ns).split("/abs/")[-1]; base = re.sub(r"v\d+$", "", aid)
        out[base] = {"title": " ".join(e.findtext("a:title", "", ns).split()), "authors": [x.findtext("a:name", "", ns) for x in e.findall("a:author", ns)],
                     "year": e.findtext("a:published", "", ns)[:4], "id": aid}
    return out
def bibtex(key, m, venue):
    authors = " and ".join(m["authors"]); t = m["title"].replace("{", "").replace("}", "")
    if venue:
        return f"@inproceedings{{{key},\n  title={{{t}}},\n  author={{{authors}}},\n  booktitle={{{venue}}},\n  year={{{m['year']}}},\n  note={{arXiv:{m['id']}}}\n}}\n"
    base = re.sub(r"v\d+$", "", m["id"])
    return f"@article{{{key},\n  title={{{t}}},\n  author={{{authors}}},\n  journal={{arXiv preprint arXiv:{base}}},\n  year={{{m['year']}}}\n}}\n"
def main():
    specs = []
    for a in sys.argv[1:]:
        key, rest = a.split("=", 1); aid, _, venue = rest.partition(":"); specs.append((key, aid, venue))
    meta = {}
    for i in range(0, len(specs), 25):
        meta.update(fetch([s[1] for s in specs[i:i+25]])); time.sleep(3.2)
    rec = {}; out = []
    for key, aid, venue in specs:
        m = meta.get(aid)
        if not m: print(f"% MISSING {key} {aid}", file=sys.stderr); continue
        out.append(bibtex(key, m, venue)); rec[key] = {"arxiv": m["id"], "title": m["title"], "first_author": m["authors"][0], "year": m["year"], "venue": venue}
        print(f"{key:26s} {m['id']:14s} {m['authors'][0][:22]:22s} {m['title'][:70]}", file=sys.stderr)
    sys.stdout.write("\n".join(out)); Path("/tmp/claude-1010/-home-anon-hq-projects-scaling-down-law/460f58d4-6561-477e-8d6d-434658e40d6f/scratchpad/ref_fetch_record.json").write_text(json.dumps(rec, indent=1))
if __name__ == "__main__": main()
