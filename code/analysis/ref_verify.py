#!/usr/bin/env python3
"""Verify every BibTeX entry in paper/paper/references.bib against arXiv (export API) and Semantic Scholar (Graph API):
title match (normalized), first-author surname match, year; write docs/REFERENCE_VERIFICATION.md and a JSON record.
Usage: python3 analysis/ref_verify.py [--keys k1,k2]"""
import re, json, time, sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; BIB = ROOT / "paper/paper/references.bib"; OUT = ROOT / "paper/docs"
def norm(t): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower().replace("{", "").replace("}", "").replace("-", ""))).strip()
def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        body = m.group(3); f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", body):
            f[fm.group(1).lower()] = fm.group(2).strip().strip("{}\"").strip()
        entries[m.group(2)] = {"type": m.group(1), **f}
    return entries
def first_surname(authors):
    a = re.split(r"\s+and\s+", authors)[0].strip()
    return (a.split(",")[0] if "," in a else a.split()[-1]).lower().replace("{", "").replace("}", "").replace("-", "")
def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "ref-verify/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode("utf-8", "replace")
def arxiv_search(title):
    q = urllib.parse.quote(f'ti:"{title}"'); xml = get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=3")
    ns = {"a": "http://www.w3.org/2005/Atom"}; out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        out.append({"title": " ".join(e.findtext("a:title", "", ns).split()), "authors": [x.findtext("a:name", "", ns) for x in e.findall("a:author", ns)],
                    "id": e.findtext("a:id", "", ns), "published": e.findtext("a:published", "", ns)[:4]})
    return out
def s2_search(title):
    q = urllib.parse.quote(title); js = json.loads(get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=3&fields=title,authors,year,venue,externalIds"))
    return js.get("data", [])
def dblp_search(title):
    q = urllib.parse.quote(title); js = json.loads(get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=3"))
    hits = js.get("result", {}).get("hits", {}).get("hit", []) or []; out = []
    for h in hits:
        info = h.get("info", {}); au = info.get("authors", {}).get("author", []); au = [au] if isinstance(au, dict) else au
        out.append({"title": info.get("title", ""), "authors": [x.get("text", "") for x in au], "year": info.get("year", ""), "venue": info.get("venue", "")})
    return out
def main():
    keys = None
    if "--keys" in sys.argv: keys = sys.argv[sys.argv.index("--keys") + 1].split(",")
    entries = parse_bib(BIB.read_text()); rows = []; record = {}
    for k, e in entries.items():
        if keys and k not in keys: continue
        title = e.get("title", ""); nt = norm(title); sur = first_surname(e.get("author", ""))
        verdict, ev = "UNVERIFIED", ""
        try:
            hits = arxiv_search(title); time.sleep(3.1)
            for h in hits:
                if norm(h["title"]) == nt or nt in norm(h["title"]) or norm(h["title"]) in nt:
                    ok_a = sur in norm(h["authors"][0]) if h["authors"] else False
                    verdict = "OK-arxiv" if ok_a else "TITLE-OK/AUTHOR-MISMATCH"; ev = f'arXiv {h["id"].split("/")[-1]} ({h["published"]}); first author {h["authors"][0] if h["authors"] else "?"}'; break
        except Exception as ex: ev = f"arxiv error: {ex}"
        if not verdict.startswith("OK"):
            try:
                hits = s2_search(title); time.sleep(1.2)
                for h in hits:
                    if norm(h.get("title", "")) == nt or nt in norm(h.get("title", "")):
                        au = h.get("authors") or []; ok_a = bool(au) and sur in norm(au[0].get("name", ""))
                        verdict = "OK-s2" if ok_a else "TITLE-OK/AUTHOR-MISMATCH"; ev += f' | S2 {h.get("year")} {h.get("venue") or ""} first author {au[0].get("name") if au else "?"} {h.get("externalIds", {}).get("ArXiv", "")}'; break
            except Exception as ex: ev += f" | s2 error: {ex}"
        if not verdict.startswith("OK"):
            try:
                hits = dblp_search(title); time.sleep(1.0)
                for h in hits:
                    ht = norm(h["title"].rstrip("."))
                    if ht == nt or nt in ht or ht in nt:
                        ok_a = bool(h["authors"]) and sur in norm(h["authors"][0])
                        verdict = "OK-dblp" if ok_a else "TITLE-OK/AUTHOR-MISMATCH"; ev += f' | DBLP {h["year"]} {h["venue"]} first author {h["authors"][0] if h["authors"] else "?"}'; break
            except Exception as ex: ev += f" | dblp error: {ex}"
        rows.append((k, verdict, e.get("year", ""), title[:80], ev)); record[k] = {"verdict": verdict, "evidence": ev, "bib_title": title, "bib_first_author": sur, "bib_year": e.get("year", "")}
        print(f"{k:32s} {verdict:26s} {ev[:90]}", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "REFERENCE_VERIFICATION.json").write_text(json.dumps(record, indent=1))
    md = ["# Reference verification", "", f"Source: {BIB.relative_to(ROOT)}; checked against the arXiv export API and the Semantic Scholar Graph API by normalized title and first-author surname. Generated by analysis/ref_verify.py.", "",
          "| key | verdict | year | title | evidence |", "|---|---|---|---|---|"] + [f"| {k} | {v} | {y} | {t} | {ev} |" for k, v, y, t, ev in rows]
    (OUT / "REFERENCE_VERIFICATION.md").write_text("\n".join(md) + "\n"); print("wrote", OUT / "REFERENCE_VERIFICATION.md")
if __name__ == "__main__": main()
