"""Regenerate the editorial deliverables table from its frozen text source."""
from pathlib import Path

try:
    from .paper_table_text import table_text
except ImportError:
    from paper_table_text import table_text


def generate(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    # The public mirror is rooted one level closer to the paper and data mirror.
    base = root / 'paper' if (root / 'paper/data_mirror').is_dir() else root
    source = base / 'data_mirror/tables/final_deliverables.tex'
    output = base / 'paper/tables/final_deliverables.tex'
    output.write_text(table_text(source.read_text()))
    return output


if __name__ == '__main__':
    print(generate())
