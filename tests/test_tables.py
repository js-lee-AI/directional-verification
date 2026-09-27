"""Every table script rebuilds its paper values from the bundled data and exits 0 on a full match."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p.name for p in (ROOT / "experiments").glob("table*.py"))


def test_every_table_has_a_script():
    assert len(SCRIPTS) == 13


@pytest.mark.parametrize("script", SCRIPTS)
def test_table_script_matches_the_paper(script, tmp_path):
    out = tmp_path / "rows.json"
    result = subprocess.run([sys.executable, str(ROOT / "experiments" / script), "--out", str(out)],
                            capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    total = result.stdout.strip().splitlines()[-1].split()
    assert total[0] == total[2] and result.stdout.rstrip().endswith("values match the paper.")
    assert out.exists()


def test_cli_validate_reconstructs_every_label():
    result = subprocess.run([sys.executable, "-m", "ddv", "validate", "--root", str(ROOT)],
                            capture_output=True, text=True, timeout=600, cwd=ROOT, check=True)
    assert result.stdout.count("labels match") == 3


def test_cli_evaluate_gives_the_run_a_row(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ddv", "evaluate", "--data", "data/screened.json",
                             "--predictions", "data/predictions/screened_known_mean_seed0.json.gz",
                             "--out", str(tmp_path / "seed0.json")],
                            capture_output=True, text=True, timeout=600, cwd=ROOT, check=True)
    for line in ("open_accuracy               80.29", "whole_accuracy              80.00",
                 "inventory_accuracy          89.28"):
        assert line in result.stdout
