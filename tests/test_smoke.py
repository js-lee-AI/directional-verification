import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ddv

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ("known direction: Paulina Rubio\n"
              "reverse:         Diego Luna\n"
              "reverse with DC: Odiseo Bichir\n")


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120, cwd=ROOT).stdout


def test_import_does_not_pull_torch():
    # A fresh interpreter, since pytest plugins may have imported torch already.
    out = run("-c", "import sys, ddv; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_mean_rule_on_a_hand_checked_case():
    scores = [[-1.0, -2.0, -0.5], [-3.0, -1.0, -0.5]]
    # Means are -2.0, -1.5 and -0.5, but the last candidate is the queried parent.
    assert ddv.select_label(scores, ["A", "B", "P"], parent="P") == "B"
    assert ddv.select_label(scores, ["A", "B", "C"], parent="P") == "C"


def test_domain_context_is_eq3():
    reverse, context = np.array([-10.0, -12.0]), np.array([-20.0, -15.0])
    assert ddv.domain_context(reverse, context).tolist() == [10.0, 3.0]
    assert ddv.domain_context(reverse, context, lam=0.5).tolist() == [0.0, -4.5]


def test_quickstart_prints_what_the_readme_shows():
    assert run(str(ROOT / "examples" / "quickstart.py")) == QUICKSTART


def test_cli_demo_matches_quickstart():
    assert run("-m", "ddv", "demo") == QUICKSTART


def test_cli_help():
    out = run("-m", "ddv", "--help")
    for command in ("demo", "select", "validate", "evaluate", "summarize"):
        assert command in out


@pytest.mark.gpu
def test_score_candidates_on_cuda(tiny_model):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model, tokenizer = ddv.load_teacher({"model": str(tiny_model)}, "cuda")
    scores = ddv.score_candidates(model, tokenizer, "Parent Name", ["First Child", "Other Child"])
    assert scores.shape == (2,) and np.isfinite(scores).all()
