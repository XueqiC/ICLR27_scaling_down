"""Reader-facing tables preserve numerical evidence and reject opaque labels."""
from pathlib import Path
import importlib.util
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'results/published-tables'
sys.path.insert(0, str(ROOT))
from analysis.paper_table_text import plain_language, PLAIN_CAPTIONS

spec = importlib.util.spec_from_file_location('numeric_table_audit', ROOT/'scripts/caption_only_diff.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
BEFORE = ROOT/'results/table-language-before'
NAMES = sorted(p.name for p in BEFORE.glob('*.tex'))


@pytest.mark.parametrize('name', NAMES)
def test_all_printed_numeric_tokens_keep_order_signs_and_precision(name):
    assert audit.numeric_tokens((BEFORE/name).read_text()) == audit.numeric_tokens((TABLES/name).read_text())


@pytest.mark.parametrize('name', NAMES)
def test_headers_and_cells_have_no_codes_or_abbreviations(name):
    text = (TABLES/name).read_text()
    cells = '\n'.join(re.findall(r'\\begin\{tabular\*\}.*?\\end\{tabular\*\}', text, re.S))
    cells = re.sub(r'(?m)^%[^\n]*', '', cells)
    cells = re.sub(r'\\(?:label|ref|eqref)\{[^}]*\}', '', cells)
    assert not re.search(r'\b(?:MAEs?|CI|LOSO|LOCO|RFRA|PFFA|nD|pbm|pba|med|so|A1|A2|F1|F2|K0|K1|[Vv]\d{2}|df)\b', cells)
    assert not re.search(r'\b[PFR][·/][F–R][·/][FR][·/]A\b', cells)
    assert not re.search(r'\w@(?:step)?\d', cells)
    assert r'\shortstack' not in cells
    assert r'\resizebox' not in cells
    assert r'\begin{tabular*}{\textwidth}' in cells


def test_numeric_comparison_detects_reorder_sign_precision_and_formula_changes():
    original = r'\begin{tabular*}{\textwidth}{p{0.25\textwidth}r} A1 & +0.010 / -2.00 \\ $1/128$ \end{tabular*}'
    assert audit.numeric_tokens(original) == ['+0.010', '-2.00', '1', '128']
    for old, new in [('+0.010', '0.010'), ('-2.00', '-2.0'), ('1/128', '1/256'), ('+0.010 / -2.00', '-2.00 / +0.010')]:
        assert audit.numeric_tokens(original) != audit.numeric_tokens(original.replace(old,new))
    assert audit.numeric_tokens(original) == audit.numeric_tokens(original.replace('A1', 'linear power form').replace('0.25\\textwidth','0.40\\textwidth'))


def test_excluded_main_table_is_untouched_by_language_pass():
    text = r'\begin{table}\caption{Existing caption.}\label{tab:main-prediction-v2} A1 & 0.123 \end{table}'
    assert plain_language(text) == text


def test_all_registered_captions_are_complete_short_sentences_with_units():
    for label, text in PLAIN_CAPTIONS.items():
        assert text.startswith('The table '), label
        assert text.endswith('.'), label
        assert len(text.split()) <= 60, (label,len(text.split()))
        assert any(unit in text for unit in ['nats', 'parameters', 'count', 'percentage']), label


def test_varied_baselines_are_named_in_pair_headings():
    text = (TABLES/'pred_full.tex').read_text()
    assert r'\textit{Relation / median}\newline' in text
    assert r'\textit{Relation / per-bit median}\newline' in text
    assert r'\textit{Relation / no pretraining-token input}\newline' in text
    assert 'What is held out' in text
    assert 'Fold rule: relation / input baseline / simple baseline' in text


def test_distillation_descriptor_is_not_a_measurement_count():
    text = (TABLES/'distill_forms_audit.tex').read_text()
    assert ' & Logarithmic student size &' in text
    assert ' & Standardized initial loss &' in text
    assert ' & Measured cells &' not in text


def test_coefficient_table_has_one_named_parameter_per_row():
    text = (TABLES/'distill_forms_audit.tex').read_text()
    assert 'Form & Capability & Coefficient or budget & Value' in text
    assert 'Coefficient order &' not in text
    rows = [line for line in text.splitlines() if ' & Saturation budget in tokens & ' in line]
    assert len(rows) == 6
    assert all(len(line.split(' & ')) == 4 for line in rows)


@pytest.mark.parametrize('name', NAMES)
def test_actual_captions_including_continuations_name_units(name):
    text = (TABLES/name).read_text()
    for match in re.finditer(r"\\caption(?:\*|\[\])?\{", text):
        caption, _ = audit.group(text, match.end()-1)
        assert caption.startswith('The table '), (name, caption)
        assert caption.endswith('.'), (name, caption)
        assert len(caption.split()) <= 60, (name, caption)
        assert any(word in caption for word in ['nats', 'parameters', 'count', 'percentage']), (name, caption)
