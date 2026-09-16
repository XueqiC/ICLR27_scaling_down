#!/usr/bin/env python3
"""Compare staged tables with the paper tables: report which changed and whether anything outside captions/comments differs."""
import re, glob, os, sys
S = sys.argv[1]; root = sys.argv[2] if len(sys.argv) > 2 else "paper/paper/tables"
def strip(s):
    s = re.sub(r'(?m)^%[^\n]*\n', '', s)
    out = []; i = 0
    while True:
        m = re.search(r'\\caption\*?\{', s[i:])
        if not m:
            out.append(s[i:]); break
        out.append(s[i:i + m.start()]); j = i + m.end(); d = 1
        while d:
            c = s[j]
            if c == '{' and s[j - 1] != '\\': d += 1
            elif c == '}' and s[j - 1] != '\\': d -= 1
            j += 1
        i = j
    return ''.join(out)
changed = []; bad = []
for f in sorted(glob.glob(f"{S}/*.tex")):
    n = os.path.basename(f); new = open(f).read(); old = open(f"{root}/{n}").read()
    if new != old:
        changed.append(n)
        if strip(new) != strip(old): bad.append(n)
print("changed tables:", len(changed)); print("non-caption differences:", bad); print(" ".join(changed))
sys.exit(1 if bad else 0)
