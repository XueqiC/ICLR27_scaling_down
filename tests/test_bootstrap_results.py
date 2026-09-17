"""Publication layout must preserve measurement identity and avoid stale code."""
from pathlib import Path

import bootstrap_results as bootstrap


def test_v12_uses_one_directory_matching_its_portable_student_tag():
    relative = Path("v12-distill/pythia-160m--step16000/run/eval.json")
    assert bootstrap.targets(relative) == [bootstrap.RESULTS / relative]


def test_frozen_state_inputs_keep_both_directory_and_filename_spellings():
    relative = Path("v53-prune-dev/pythia-160m--step16000/dense--step16000.json")
    assert set(bootstrap.targets(relative)) == {
        bootstrap.RESULTS / "v53-prune-dev" / state / filename
        for state in ("pythia-160m--step16000", "pythia-160m@step16000")
        for filename in ("dense--step16000.json", "dense@step16000.json")
    }
