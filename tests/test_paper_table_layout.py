"""Layout preserves table content while replacing scaling with column wrapping."""
from pathlib import Path
import re

from analysis.paper_table_layout import table_layout
from analysis.final_deliverables_table import generate


def table(body, label, spec='lrr', font=r'\footnotesize'):
    return (r'\begin{table}[H]' + font + '\n'
            + r'\begin{tabular}{' + spec + '}\n'
            + r'\toprule ' + ' & '.join(['Name'] + [chr(65+i) for i in range(len(spec)-1)]) + r' \\ \midrule' + '\n'
            + body + '\n' + r'\bottomrule\end{tabular}'
            + r'\caption{Frozen results.}\label{' + label + r'}\end{table}')


def test_natural_columns_expand_without_changing_cells_or_type_size():
    body = r'Alpha & -0.010 & 3.14159 \\'
    result = table_layout(table(body, 'tab:round3_coef'))
    assert r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrr@{}}' in result
    assert body in result
    assert r'\footnotesize' in result
    assert not re.search(r'\\(?:resizebox|scalebox|scriptsize|tiny)', result)
    assert r'\caption{Frozen results.}\label{tab:round3_coef}' in result


def test_dense_columns_allow_wraps_without_changing_values_or_formulas():
    body = r'\shortstack[l]{Two words\\on lines} & $1,2$ & -0.12345678 & 4 & 5 \\'
    result = table_layout(table(body, 'tab:pred_source', spec='lllll'))
    assert r'\shortstack' not in result
    assert '{Two words on lines}' in result
    assert r'$1,\allowbreak 2$' in result
    assert '-0.12345678' in result
    assert r'\hspace{0pt}' in result  # permit first-word hyphenation
    assert result.count(r'\label{tab:pred_source}') == 1


def test_continuations_keep_one_label_and_all_rows_in_order():
    rows = [f'Row {i} & {i}.125 & -{i}.875 & 4 & 5 '+r'\\' for i in range(23)]
    result = table_layout(table('\n'.join(rows), 'tab:pred_source', spec='lllll'))
    assert result.count(r'\begin{table}[tb]') == 3  # top or bottom only; never a float page
    assert result.count(r'\ContinuedFloat') == 2
    assert result.count(r'\label{tab:pred_source}') == 1
    assert result.count(r'\caption{Frozen results.}') == 1
    assert result.count(r'\caption[]{') == 2
    assert [m.group() for m in re.finditer(r'Row \d+ & \d+\.125 & -\d+\.875 & 4 & 5 \\\\',result)] == rows


def test_legacy_scaling_is_removed_without_losing_its_contents():
    source = table(r'Alpha & 1 & 2 \\', 'tab:round3_coef')
    source = source.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
    source = source.replace(r'\end{tabular}', r'\end{tabular}}')
    result = table_layout(source)
    assert r'\resizebox' not in result
    assert r'Alpha & 1 & 2 \\' in result


def test_frozen_text_generator_supports_private_and_public_layouts(tmp_path):
    for parent in (tmp_path/'private'/'paper', tmp_path/'public'):
        (parent/'data_mirror/tables').mkdir(parents=True)
        (parent/'paper/tables').mkdir(parents=True)
        source=parent/'data_mirror/tables/final_deliverables.tex'
        source.write_text(table(r'Alpha & -0.12 & 3.45 & 4 & 5 & 6 \\', 'tab:final', spec='llllll'))
        original=source.read_bytes()
        root=parent.parent if parent.name=='paper' else parent
        output=generate(root)
        assert output==parent/'paper/tables/final_deliverables.tex'
        assert source.read_bytes()==original
        assert r'Alpha & -0.12 & 3.45 & 4 & 5 & 6 \\' in output.read_text()
        assert r'\begin{tabular*}{\textwidth}' in output.read_text()
