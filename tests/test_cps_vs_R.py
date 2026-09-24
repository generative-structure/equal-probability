"""Validate the CPS implementation against R `sampling` 2.11 (project-local library) and by
empirical inclusion frequencies."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import cps  # noqa: E402

RLIB = ROOT / ".Rlib"


def run_R(code):
    env = dict(os.environ, R_LIBS_USER=str(RLIB))
    out = subprocess.run(["Rscript", "-e", "suppressMessages(library(sampling)); " + code],
                         capture_output=True, text=True, check=True, env=env).stdout
    return np.array([float(x) for x in out.split()])


@pytest.fixture(scope="module")
def target():
    rng = np.random.default_rng(5)
    N, n = 300, 30
    s = rng.lognormal(0, 0.8, N)
    pi = n * s / s.sum()
    assert pi.max() < 1
    return pi, n


def test_first_order_matches_target(target):
    pi, n = target
    w, err = cps.working_odds(pi, n)
    assert err < 1e-9


def test_matches_R_sampling(target):
    """(a) R's recursion applied to OUR working odds returns the target pi;
    (b) R's own working probabilities (UPMEpiktildefrompik) give odds proportional to ours
        (CPS is invariant to rescaling the odds)."""
    pi, n = target
    w, _ = cps.working_odds(pi, n)
    f = ROOT / "logs" / "_cps_w.txt"; np.savetxt(f, w)
    pk_R = run_R(f'w <- scan("{f}", quiet=TRUE); q <- UPMEqfromw(w, {n}); '
                 f'cat(sprintf("%.15f", UPMEpikfromq(q)), sep="\\n")')
    assert np.max(np.abs(pk_R - pi)) < 1e-8
    g = ROOT / "logs" / "_cps_pi.txt"; np.savetxt(g, pi)
    pt = run_R(f'p <- scan("{g}", quiet=TRUE); cat(sprintf("%.15f", UPMEpiktildefrompik(p)), sep="\\n")')
    ratio = (pt / (1 - pt)) / w
    assert ratio.std() / ratio.mean() < 1e-6


def test_empirical_frequencies(target):
    pi, n = target
    d = cps.CPS(pi)
    I = d.sample(np.random.default_rng(9), 20000)
    assert np.all(I.sum(1) == n)
    freq = I.mean(0)
    z = (freq - pi) / np.sqrt(pi * (1 - pi) / 20000)
    assert np.abs(z).max() < 4.5 and abs(z.mean()) < 0.2
