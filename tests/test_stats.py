"""Statistics module; the paper's printed values serve as regression anchors."""

import numpy as np

from eval import stats as S


def test_gap_recovery_matches_paper():
    assert abs(S.gap_recovery(38.6, 38.4, 57.0) - 1.0752688) < 1e-6      # standard OPD, anisotropy probe
    assert round(S.gap_recovery(44.3, 38.4, 57.0), 2) == 31.72              # CHOP full


def test_rounded_pair_table_spearman():
    alpha = [0.06, 0.13, 0.05, 0.27, 0.18, 0.17, 0.25, 0.33, 0.45, 0.38, 0.49]
    auroc = [0.74, 0.75, 0.71, 0.78, 0.74, 0.77, 0.76, 0.73, 0.74, 0.76, 0.79]
    resolve = [8.2, 13.2, 7.8, 13.6, 10.4, 11.8, 12.9, 12.2, 13.8, 13.4, 14.9]
    assert abs(S.spearman(alpha, resolve) - 0.8454545) < 1e-6
    assert abs(S.spearman(auroc, resolve) - 0.6345247) < 1e-6


def test_category_reconciliation():
    n = [211, 86, 124, 79]
    rates = {"off": [15.4, 8.2, 10.7, 7.7], "opd": [9.9, 2.7, 4.7, 4.4], "deploy": [16.2, 9.3, 12.6, 10.3],
             "oracle": [16.4, 12.5, 14.3, 9.9]}
    got = {k: round(S.weighted_mean(v, n), 1) for k, v in rates.items()}
    assert got == {"off": 11.8, "opd": 6.5, "deploy": 13.2, "oracle": 14.2}


def test_bootstrap_ci_and_paired_tests():
    rng = np.random.default_rng(0)
    a = (rng.random((3, 500)) < 0.14).astype(int)
    b = (rng.random((3, 500)) < 0.07).astype(int)
    est = S.resolve_rate_ci(a, n_resamples=2000)
    assert est.lo < est.mean < est.hi and 1.0 < est.hi - est.lo < 8.0
    diff, p = S.paired_bootstrap(a, b, n_resamples=2000)
    assert diff > 0 and p < 0.01
    _, p_same = S.paired_bootstrap(a, a, n_resamples=500)
    assert p_same == 1.0
    m1, m2 = (rng.random((3, 30, 16)) < 0.45), (rng.random((3, 30, 16)) < 0.43)
    d, p = S.paired_bootstrap_problems_seeds(m1, m2, n_resamples=2000)
    assert abs(d - 100 * (m1.mean() - m2.mean())) < 1e-9 and 0 <= p <= 1


def test_holm_bonferroni():
    assert S.holm_bonferroni([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
