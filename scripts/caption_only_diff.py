#!/usr/bin/env python3
"""Compare captions, or assert identical ordered numerical table content.

--numeric includes captions, notes, formulas, model sizes, signs and trailing
zeroes. TeX layout dimensions, references and identifier digits (A1, V70,
coefficient subscripts) are not numeric values. No rounding or sorting is used.
"""
import argparse
import json
import re
from pathlib import Path


def group(s, start):
    depth, end = 1, start + 1
    while depth:
        if s[end] == '\\':
            end += 2
            continue
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[start + 1:end - 1], end


def visible_text(s):
    s = re.sub(r'(?m)^%[^\n]*\n?', '', s)
    patterns = [
        (r'\\begin\{(?:tabular\*|tabularx)\}', 2),
        (r'\\begin\{tabular\}', 1),
        (r'\\(?:setlength|renewcommand)', 2),
        (r'\\(?:label|ref|eqref|cite|texttt|hspace|vspace)', 1),
        (r'\\cmidrule(?:\([^)]*\))?', 1),
    ]
    for pattern, count in patterns:
        while m := re.search(pattern, s):
            end = m.end()
            if s[end:end+1] != '{':
                break
            for _ in range(count):
                _, end = group(s, end)
            s = s[:m.start()] + ' ' + s[end:]
    s = re.sub(r'\\addlinespace(?:\[[^]]*\])?', '', s)
    while m := re.search(r'\\multicolumn', s):
        _, pos = group(s, m.end())
        _, pos = group(s, pos)
        value, end = group(s, pos)
        s = s[:m.start()] + value + s[end:]
    s = re.sub(r'_(?:\{[^{}]*\}|[0-9])', '', s)
    s = re.sub(r'\b(?:[Vv]\d+|[AFKGQO]\d+|L0|logN)\b', '', s)
    s = s.replace(r'\allowbreak', '').replace(r'\newline', ' ')
    s = re.sub(r'\\[A-Za-z]+\*?', ' ', s)
    return s


def numeric_tokens(s):
    s = visible_text(s)
    pattern = r'(?<![\w.])(?:[+-](?=\d))?(?:\d+(?:\.\d+)?|\.\d+)'
    tokens = []
    for m in re.finditer(pattern, s):
        token = m.group()
        if token.startswith('-') and m.start() and (s[m.start()-1].isalnum() or s[m.start()-1] == '-'):
            token = token[1:]
        tokens.append(token)
    return tokens


def strip(s):
    s = re.sub(r'(?m)^%[^\n]*\n', '', s)
    while m := re.search(r'\\caption(?:\*|\[\])?\{', s):
        _, end = group(s, m.end()-1)
        s = s[:m.start()] + s[end:]
    return s


def compare(staged, root, *, numeric=False, exclude=()):
    rows = []
    for path in sorted(Path(staged).glob('*.tex')):
        if path.name in exclude:
            continue
        new, old = path.read_text(), (Path(root)/path.name).read_text()
        left, right = (numeric_tokens(old), numeric_tokens(new)) if numeric else (strip(old), strip(new))
        item = {'table': path.name, 'changed': old != new, 'identical': left == right}
        if numeric:
            item.update(old_tokens=len(left), new_tokens=len(right))
        if left != right:
            i = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
            item.update(first_difference=i, old=left[max(0,i-3):i+7], new=right[max(0,i-3):i+7])
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staged', type=Path)
    parser.add_argument('root', nargs='?', type=Path, default=Path('paper/paper/tables'))
    parser.add_argument('--numeric', action='store_true')
    parser.add_argument('--exclude', action='append', default=[])
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rows = compare(args.staged, args.root, numeric=args.numeric, exclude=args.exclude)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2)+'\n')
    bad = [r for r in rows if not r['identical']]
    print(f"Compared {len(rows)} tables; changed {sum(r['changed'] for r in rows)}; {'numeric-token' if args.numeric else 'non-caption'} differences {len(bad)}")
    for item in bad:
        print(json.dumps(item))
    return bool(bad)


if __name__ == '__main__':
    raise SystemExit(main())
