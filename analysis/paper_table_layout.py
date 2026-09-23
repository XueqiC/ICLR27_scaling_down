"""Publication table widths, with unchanged type sizes and numeric content.

Small standalone tables use elastic intercolumn space. Dense tables use explicit
paragraph columns, so labels and lists wrap at the paper's existing font size.
Profiles are fractions of the space left after the intercolumn padding.
"""
from __future__ import annotations

import re

TABLE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.S)
TABULAR = re.compile(r"\\begin\{(tabular\*?|tabularx)\}.*?\\end\{\1\}", re.S)

# One profile per tabular, in source order; None preserves natural columns.
PROFILES = {
    'tab:main-prediction-v2': [[24, 76]],
    'tab:models': [[12, 25, 10, 18, 9, 15, 11]],
    'tab:v56_forms': [[19, 9, 12, 12, 12, 12, 12, 12]],
    'tab:v56_condition': [[19, 17, 17, 11, 14, 9, 13]],
    'tab:distill_forms_audit': [[19, 25, 10, 8, 12, 9, 17], [22, 15, 21, 21, 21]],
    'tab:shared_structure': [[15, 15, 16, 19, 16, 19]],
    'tab:main_final': [[12.7, 23, 9.7, 16.3, 7.3, 13, 18]],
    'tab:main_context': [[12.7, 23, 9.7, 16.3, 7.3, 13, 18]],
    'tab:final': [[12, 23, 14, 17, 20, 14]],
    'tab:pred_full': [[31, 12, 12, 15, 15, 15]],
    'tab:pred_source': [[34, 26, 13.333333, 13.333333, 13.333334]],
    'tab:pred_config_prune': [[34, 26, 13.333333, 13.333333, 13.333334]],
    'tab:pred_config_qd': [[34, 26, 13.333333, 13.333333, 13.333334]] * 2,
    'tab:p1v2': [[14, 7, 14, 7, 7, 6, 9, 8.5, 9, 7, 13]],
    'tab:round3_quant': [[23, 8] + [7.6666666667] * 9],
    'tab:p2v2_test': [[25, 12, 13, 10, 11, 23, 6]],
    'tab:distill_paired': [[15, 10, 8, 8, 14, 14, 8, 14, 14]],
    'tab:distill_confirm': [[13, 12, 17, 17, 10, 10, 21]],
    'tab:musique_scope': [[18, 24, 12, 12, 17, 17]],
    'tab:panel_prune': [[21, 8, 8, 8, 11, 13, 15, 16]],
    'tab:panel_quant': [[19, 12, 8, 8, 8, 8, 8, 8, 15]],
    'tab:rule-confirm': [[16, 28, 11, 12, 15, 18]],
    'tab:rule-confirm-by-state': [[23, 16, 14, 14, 14, 13, 6]],
    'tab:rule-confirm-candidate-sizes': [[16, 26, 6, 6, 6, 6, 17, 17]],
    'tab:candidate-coverage': [[23, 14, 16, 16, 15, 16]],
    'tab:cohorts': [[2.3, 4, 3.3, 3.4]],
    'tab:disp_boundary': [[25, 75]],
    'tab:budgets': [[2.5, 5.7, 4.9]],
    'tab:s3-ablation': [[22, 14, 13, 9, 9, 11, 8, 14]],
    'tab:a14-baselines': [[28, 20, 8.67, 8.67, 8.67, 8.67, 8.66, 8.66]],
    'tab:p1-behaviour': [[13, 28, 7, 8, 8, 8, 8, 20]],
    'tab:storage-accounting': [[26, 26, 48]],
    'tab:accuracy-comparison': [[34, 26, 10, 10, 10, 10]],
    'tab:wanda-panel': [[16, 12, 12, 12, 12, 12, 12, 12]],
    'tab:second-family': [[26, 14, 10, 10, 10, 10, 10, 10]],
    'tab:efficiency-body': [[36, 16, 16, 16, 16]],
    'tab:eos-control': [[29, 13, 18, 40]],
    'tab:s3-ablation-body': [[19, 13, 11, 11, 11, 11, 11, 13]],
}


def group(text, start):
    """Return a balanced TeX group and the index after it."""
    assert text[start] == '{', text[start:start + 30]
    depth = 1
    i = start + 1
    while depth:
        if text[i] == '\\':
            i += 2
            continue
        depth += (text[i] == '{') - (text[i] == '}')
        i += 1
    return text[start + 1:i - 1], i


def unscale(text):
    """Remove legacy scaling commands; retain their contents verbatim."""
    pattern = re.compile(r'\\(resizebox|scalebox)\*?')
    while match := pattern.search(text):
        pos = match.end()
        _, pos = group(text, pos)
        if match[1] == 'resizebox':
            _, pos = group(text, pos)
        elif text[pos:pos + 1] == '[':
            pos = text.index(']', pos) + 1
        body, end = group(text, pos)
        text = text[:match.start()] + body + text[end:]
    return text


def _paragraphs(block):
    # A shortstack is unbreakable even inside a paragraph column. Preserve its
    # text and punctuation while allowing ordinary TeX wrapping instead.
    pattern = re.compile(r'\\shortstack(?:\[[^]]*\])?')
    while match := pattern.search(block):
        body, end = group(block, match.end())
        body = body.replace(r'\\', ' ')
        block = block[:match.start()] + '{' + body + '}' + block[end:]
    # Comma-separated math lists (not numbers themselves) may wrap as well.
    block = re.sub(r'\$(?:\\.|[^$])*\$',
                   lambda m: m.group().replace(',', r',\allowbreak '), block)
    return block


def _tabular(block, weights, *, wrap=True):
    opening = re.match(r'\\begin\{(tabular\*?|tabularx)\}', block)
    env = opening[1]
    pos = opening.end()
    if env != 'tabular':
        _, pos = group(block, pos)
    spec, pos = group(block, pos)
    body = block[pos:block.rfind(r'\end{')]
    if weights:
        if wrap:
            body = _paragraphs(body)
            body = re.sub(r'(?<=[0-9}])/+(?=[0-9\\])', lambda m: m.group() + r'\allowbreak ', body)
        n = len(weights)
        total = sum(weights)
        specs = []
        for weight in weights:
            f = weight / total
            specs.append(r'>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr '
                         + f'{f:.9f}' + r'\textwidth-'
                         + f'{2 * (n - 1) * f:.9f}' + r'\tabcolsep\relax}')
        spec = ''.join(specs)
        # Spanning cells must respect the same width as the paragraph columns.
        # All-column section headings otherwise override TeX's width constraint.
        edits = []
        col = 0
        i = 0
        while i < len(body):
            if body.startswith(r'\multicolumn', i):
                count_text, pos = group(body, i + len(r'\multicolumn'))
                count = int(count_text)
                old, end = group(body, pos)
                fraction = sum(weights[col:col + count]) / total
                width = (r'\dimexpr ' + f'{fraction:.9f}' + r'\textwidth-'
                         + f'{2 * (n - 1) * fraction - 2 * (count - 1):.9f}'
                         + r'\tabcolsep\relax')
                align = r'\raggedright' if count == n else r'\centering'
                spec_span = (('@{}' if col == 0 else '') + '>{' + align
                             + r'\arraybackslash\hspace{0pt}}p{' + width + '}'
                             + ('@{}' if col + count == n else ''))
                edits.append((pos, end, '{' + spec_span + '}'))
                _, i = group(body, end)
                col += count - 1
                continue
            if body.startswith(r'\\', i):
                col = 0
                i += 2
                continue
            if body[i] == '{':
                _, i = group(body, i)
                continue
            if body[i] == '&':
                col += 1
            if body[i] == '\\':
                i += 1
            i += 1
        for start, end, value in reversed(edits):
            body = body[:start] + value + body[end:]

    else:
        spec = re.sub(r'^@\{\}|@\{\}$', '', spec.strip())
    return (r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'
            + spec + r'@{}}' + body + r'\end{tabular*}')


# Capability names are declared as Math, Code and QA in Section 2, so table
# bodies use the short forms; captions keep the full words.
CAPABILITY_SHORT = ((r"Question answering", "QA"), (r"question answering", "QA"),
                    (r"Mathematics", "Math"), (r"mathematics", "Math"))


def _abbreviate(body):
    for long, short in CAPABILITY_SHORT:
        body = re.sub(r"(?<![\\\w-])" + long + r"(?![\w-])", short, body)
    return body


# Tables whose cells hold sentences rather than numbers read better with a rule
# between rows; without one the wrapped lines of adjacent rows run together.
ROW_RULES = {
    'tab:cohorts', 'tab:models', 'tab:final', 'tab:shared_structure',
    'tab:cap_conditioning', 'tab:budgets', 'tab:s3-identity', 'tab:locked_rule',
    'tab:disp_boundary', 'tab:panel_prune', 'tab:panel_quant',
}


def _rule_between_rows(body):
    """Insert a rule between body rows, leaving the head and the last row alone."""
    marker = chr(92) * 2          # the LaTeX row separator
    rule = chr(92) + "midrule"
    if body.count(rule) > 1:      # rules are already in place
        return body
    head, sep, rest = body.partition(rule)
    if not sep:
        return body
    tail_index = rest.rfind(chr(92) + "bottomrule")
    if tail_index < 0:
        return body
    rows, tail = rest[:tail_index], rest[tail_index:]
    parts = [p.strip() for p in rows.split(marker) if p.strip()]
    if len(parts) < 2:
        return body
    joined = (marker + "\n" + rule + "\n").join(parts)
    return head + sep + "\n" + joined + marker + "\n" + tail


def _abbreviate_body(block):
    """Shorten capability names inside the tabular; captions keep the full words."""
    return TABULAR.sub(lambda m: _abbreviate(m.group()), block)


# Cell text is compressed once, in one place, so a phrase reads the same in
# every table. Captions keep the long form, which is where a term is defined.
CELL_COMPRESSIONS = (
    ("Grouped round-to-nearest quantization", "Grouped quantization"),
    ("Per-channel round-to-nearest quantization", "Per-channel quantization"),
    ("Channel quantization development", "Channel development"),
    ("Grouped quantization development", "Grouped development"),
    ("Pruning development median curve", "Pruning median curve"),
    ("Fixed-recipe student development", "Fixed-recipe students"),
    ("crossed inside the tested pools", "Crossed in range"),
    ("upper bound at the smallest pool", "Upper bound"),
    ("lower bound at the largest pool", "Lower bound"),
    ("Seen source state; new density", "Seen state, new density"),
    ("Maximum across capabilities", "Largest increase"),
    ("Source-conditioned predictor", "Source-conditioned"),
    ("A model state with no earlier recorded use, compressed here for the first time",
     "No earlier use; first compression"),
    ("A base with no earlier recorded use, trained here for the first time",
     "No earlier use; first training"),
    ("A new set of candidates on familiar weights", "New candidates, familiar weights"),
    ("A student trained again from familiar weights", "Retrained from familiar weights"),
    ("Yes, in earlier development", "Yes"),
    ("Not reached by density 0.60", "Not reached by 0.60"),
    ("Piecewise source interpolation", "Piecewise interpolation"),
    ("Response surface with initial loss", "Response surface, initial loss"),
    ("Response surface with student size", "Response surface, student size"),
    ("Joint budget and pool response", "Joint budget-pool response"),
)


def _compress_cells(block):
    """Shorten recurring phrases inside a tabular; captions are untouched."""
    def one(match):
        body = match.group()
        for long, short in CELL_COMPRESSIONS:
            body = body.replace(long, short)
        return body
    return TABULAR.sub(one, block)


TABLE_ENV = re.compile(r'\\begin\{table\}(\[[^\]]*\])?')


def _reorder_table(block):
    """Inside one table body: preamble, caption, label, tabular, notes, anything else, in that order."""
    m = re.search(r'\\caption\{', block)
    tm = re.search(r'\\begin\{tabular\*?\}', block)
    if not m or not tm:
        return block
    _, cap_end = group(block, m.end() - 1)
    segs = [(m.start(), cap_end, 'caption')]
    te = re.search(r'\\end\{tabular\*?\}', block[tm.end():])
    segs.append((tm.start(), tm.end() + te.end(), 'tabular'))
    for lm in re.finditer(r'\\label\{[^}]*\}', block):
        if not (m.start() <= lm.start() < cap_end):
            segs.append((lm.start(), lm.end(), 'label'))
            break
    for nm in re.finditer(r'(?:\\par\\smallskip\s*)?\\begin\{minipage\}', block):
        ne = block.find('\\end{minipage}', nm.end())
        if ne >= 0:
            segs.append((nm.start(), ne + len('\\end{minipage}'), 'note'))
    segs.sort()
    pieces, last = [], 0
    for start, end, kind in segs:
        if start < last:
            continue
        pieces.append((block[last:start], 'other'))
        pieces.append((block[start:end], kind))
        last = end
    pieces.append((block[last:], 'other'))
    found = {k: [] for k in ('caption', 'label', 'tabular', 'note', 'other')}
    before_tabular, seen_tabular = [], False
    for body, kind in pieces:
        if kind == 'tabular':
            seen_tabular = True
        if kind == 'other':
            if body.strip():
                (before_tabular if not seen_tabular else found['other']).append(body.strip('\n'))
        else:
            found[kind].append(body.strip('\n'))
    ordered = before_tabular + found['caption'] + found['label'] + found['tabular'] + found['note'] + found['other']
    return '\n' + '\n'.join(ordered) + '\n'


def caption_above(text):
    """Every table caption above its tabular, every note below it (author's rule, 2026-09-22)."""
    out, pos = [], 0
    while (m := TABLE_ENV.search(text, pos)):
        end = text.find('\\end{table}', m.end())
        if end < 0:
            break
        out += [text[pos:m.end()], _reorder_table(text[m.end():end]), '\\end{table}']
        pos = end + len('\\end{table}')
    out.append(text[pos:])
    return ''.join(out)


def house_style(text):
    """Capability short forms and compressed phrases in the body, rules where declared."""
    if __package__:
        from .paper_table_text import plain_process_words
    else:
        from paper_table_text import plain_process_words
    text = plain_process_words(_compress_cells(_abbreviate_body(text)))
    label = next((m for m in re.findall(r"\\label\{([^}]+)\}", text) if m in ROW_RULES), None)
    if label:
        text = TABULAR.sub(lambda m: _rule_between_rows(m.group()), text)
    return caption_above(text)


def table_layout(text):
    """Set full-width floating tables without changing fonts or cell contents."""
    all_labels = re.findall(r"\\label\{([^}]+)\}", text)
    fallback = next((s for s in all_labels if s in PROFILES), None)
    def layout(match):
        block = unscale(match.group())
        # Top or bottom only: a float page or a float in the middle of a column
        # leaves the blank bands this appendix was rewritten to remove.
        block = re.sub(r'(\\begin\{table\*?\})(?:\[[^]]*\])?',
                       r'\1[tb]', block, count=1)
        block = re.sub(r'(\\begin\{table\*?\}(?:\[[^]]*\])?)', r'\1\\normalfont', block, count=1)
        labels = re.findall(r'\\label\{([^}]+)\}', block)
        label = next((s for s in labels if s in PROFILES), fallback)
        profiles = iter(PROFILES.get(label, []))
        if label == 'tab:v55_loso':
            block = block.replace(r'\centering', r'\centering\setlength{\tabcolsep}{8pt}', 1)
        block = TABULAR.sub(lambda m: _tabular(m.group(), next(profiles, None), wrap=label != "tab:main-prediction-v2"), block)
        block = _compress_cells(_abbreviate_body(block))
        if label in ROW_RULES:
            block = TABULAR.sub(lambda m: _rule_between_rows(m.group()), block)
        return _paginate(block, label)
    return TABLE.sub(layout, text)

# Page breaks are part of layout: these bodies cannot fit at their existing font
# size on a single page. Continue their caption number and repeat column heads.
PAGE_ROWS = {
    'tab:pred_full': 9,
    'tab:pred_source': 10,
    'tab:pred_config_prune': 10,
    'tab:pred_config_qd': 12,
    'tab:p1v2': 8,
    'tab:locked_rule': 12,
}


def _continued_open(opening, numbered):
    return opening + (r'\ContinuedFloat' if numbered else '') + '\n'


def _paginate(block, label):
    tabs = list(TABULAR.finditer(block))
    if label == 'tab:distill_forms_audit' and len(tabs) == 2:
        first, second = tabs
        heading = block.index(r'\textbf{B.', first.end())
        notes = block[first.end():heading]
        opening = re.match(r'\\begin\{table\*?\}(?:\[[^]]*\])?', block).group()
        prefix = block[len(opening):block.index(r'\caption')]
        return (block[:first.end()] + '\n' + r'\end{table}' + '\n'
                + r'\begingroup\normalfont\scriptsize' + notes + r'\endgroup' + '\n'
                + _continued_open(opening, True) + prefix
                + r'\caption[]{Frozen forms and coefficients (continued).}' + '\n'
                + block[heading:])
    if label not in PAGE_ROWS or len(tabs) != 1:
        return block
    tab = tabs[0]
    text = tab.group()
    header_end = text.index(r'\midrule') + len(r'\midrule')
    opening = re.match(r'\\begin\{table\*?\}(?:\[[^]]*\])?', block).group()
    closing = re.search(r'\\end\{table\*?\}$', block).group()
    prefix = block[len(opening):tab.start()]
    suffix = block[tab.end():block.rfind(closing)]
    numbered = bool(re.search(r'\\caption(?!\*)\{', suffix))
    lines = text[header_end:text.rfind(r'\bottomrule')].strip().splitlines()
    head = text[:header_end]
    chunks = []
    current = []
    count = 0
    for line in lines:
        # The pruning and quantization panels in P1 have different heads.
        if label == 'tab:p1v2' and line.startswith('Source & Protocol &'):
            if current:
                chunks.append((head, current))
                current = []
                count = 0
            head = head[:head.index(r'\toprule')] + '\n' + r'\toprule' + '\n' + line + '\n' + r'\midrule'
            continue
        is_row = ' & ' in line and line.rstrip().endswith(r'\\')
        if is_row and count >= PAGE_ROWS[label]:
            chunks.append((head, current))
            current = []
            count = 0
        current.append(line)
        count += is_row
    if current:
        chunks.append((head, current))
    if len(chunks) < 2:
        return block
    result = []
    for index, (head, rows) in enumerate(chunks):
        # A rule immediately adjacent to the repeated head/footer is redundant.
        while rows and rows[0].strip() in (r'\midrule', r'\addlinespace'):
            rows.pop(0)
        while rows and rows[-1].strip() in (r'\midrule', r'\addlinespace'):
            rows.pop()
        caption = suffix if index == 0 else ('\n' +
            (r'\caption[]{' if numbered else r'\caption*{') + 'Continued from the preceding table part.}\n')
        begin = _continued_open(opening, numbered and index > 0)
        result.append(begin + prefix + head + '\n' + '\n'.join(rows) + '\n'
                      + r'\bottomrule\end{tabular*}' + caption + closing)
    return '\n'.join(result)


# Expanded words need paragraph columns even in the formerly compact tables.
# We retain type sizes and the existing data order.
READER_PROFILES = {
    'tab:round3_coef': [[18, 14, 14, 18, 18, 18]],
    'tab:quant2d_coef': [[15, 25, 14, 15, 15, 16]],
    'tab:v53_loso': [[20, 27, 13, 12, 15, 13]],
    'tab:v55_loso': [[46, 18, 18, 18]],
    'tab:round3_prune': [[23, 10]+[11.1666667]*6],
    'tab:model_arch': [[23, 21, 14, 14, 14, 14]],
    'tab:cap_conditioning': [[17, 16, 12, 12, 12, 12, 19]],
    'tab:cond_audit': [[40, 20, 20, 20], [20, 17, 17, 23, 23]],
    'tab:quant_ident': [[44, 11, 15, 15, 15], [24, 34, 14, 14, 14]],
    'tab:quant_threeway': [[22, 33, 15, 15, 15]],
    'tab:prune_repeat': [[34, 17, 15, 19, 15]],
    'tab:p3_check': [[31, 23, 23, 23]],
    'tab:p2v2_test': [[24, 12, 13, 10, 11, 22, 8]],
    'tab:qa_scope': [[34]+[11]*6],
    'tab:locked_rule': [[19, 13, 16, 22, 17, 13]],
    'tab:rule_decomp': [[22, 30, 16, 16, 16]],
    'tab:selection-feasible': [[22, 19, 10, 12, 12, 12, 13]],
    'tab:pred_full': [[27, 11, 17, 15, 15, 15]],
    'tab:main_final': [[13, 20, 12, 18, 11, 13, 13]],
    'tab:main_context': [[13, 20, 12, 18, 11, 13, 13]],
    'tab:p1v2': [[14, 12, 14, 7, 8, 7, 8, 7, 7, 7, 9]],
    'tab:shared_structure': [[17, 13, 14, 18, 18, 20]],
    'tab:rule-confirm-by-state': [[23, 15, 13, 13, 13, 13, 10]],
}


def reader_layout(block, label):
    profiles = iter(READER_PROFILES.get(label, PROFILES.get(label, [])))
    # The coefficient panel has already been split into a continued float.
    if label == 'tab:distill_forms_audit' and r'\ContinuedFloat' in block:
        profiles = iter([[35, 17, 28, 20]])
    block = TABULAR.sub(lambda m: _tabular(m.group(), next(profiles, None)), block)
    if label == 'tab:distill_forms_audit' and 'Coefficient or budget &' in block:
        return _split_coefficient_rows(block)
    if label == 'tab:locked_rule':
        block = _paginate(block, label)
        block = block.replace(r'\caption[]{Continued from the preceding table part.}',
                              r'\caption[]{The table continues the frozen selection rules. Absolute losses are in nats per native token.}')
    return block


def _split_coefficient_rows(block):
    """Split named scalar coefficients while printing numeric notes only once."""
    tab = TABULAR.search(block)
    body = tab.group()
    start = body.index(r'\midrule') + len(r'\midrule')
    end = body.rfind(r'\bottomrule')
    rows = body[start:end].strip().splitlines()
    chunks = [rows[i:i+24] for i in range(0, len(rows), 24)]
    opening = re.match(r'\\begin\{table\*?\}(?:\[[^]]*\])?', block).group()
    continued = (opening+r'\ContinuedFloat\normalfont\centering\scriptsize'
                 +r'\setlength{\tabcolsep}{3pt}'+'\n'
                 +r'\caption[]{The table continues the named coefficient list. Coefficients produce loss changes in nats per native token; saturation budgets are in supervised tokens.}'+'\n')
    result = []
    for i, chunk in enumerate(chunks):
        prefix = block[:tab.start()] if i == 0 else continued
        suffix = block[tab.end():] if i == len(chunks)-1 else '\n'+r'\end{table}'
        result.append(prefix + body[:start]+'\n'+'\n'.join(chunk)+'\n'+body[end:]+suffix)
    return '\n'.join(result)
