# =============================================================================
# angular.py — Angular-concentration features (feature group 4 of 4)
# =============================================================================
# Separates Edge-Ring from Edge-Loc — the hardest pair in this dataset.
#
# Physical motivation
# -------------------
# Both patterns concentrate failures at the wafer EDGE (high radius), so
# radius alone cannot tell them apart. The distinguishing axis is angular
# spread of the edge failures:
#   - Edge-Ring : failures wrap the entire rim   -> angles fill 0..360 deg
#                 uniformly -> LOW angular concentration.
#   - Edge-Loc  : failures sit in one arc/sector -> angles cluster in a narrow
#                 band -> HIGH angular concentration.
#
# Honest caveat (confirmed in the Stage-5 confusion analysis): this is a WEAK
# separator. A partial Edge-Ring or a wide Edge-Loc arc lands in the middle,
# and the two classes overlap more than any other pair in the feature space.
# The feature is retained because it is the *best available* angular signal,
# not because it is decisive — see test_angular.py for the modest-margin test
# that documents this.
#
# Two complementary scalars are returned:
#   - resultant_length  : |mean unit vector| of edge-failure angles (circular
#                         concentration, R). 0 = perfectly uniform ring,
#                         1 = all failures at one angle.
#   - peak_bin_fraction : share of edge failures in the densest angular bin.
#
# WM-811K cell encoding:  0 = no die (off-wafer)   1 = pass   2 = fail
# =============================================================================

import numpy as np

FAIL_VALUE = 2          # a die that failed electrical test
MIN_EDGE_DEFECTS = 4    # below this the angle statistic is meaningless


def angular_concentration(
    wafer,
    edge_frac: float = 0.55,
    n_bins: int = 16,
) -> np.ndarray:
    """Return [resultant_length, peak_bin_fraction] for edge failures.

    Logic:
      1. Extract failed dies and the centroid of all dies (stable centre).
      2. Keep only EDGE failures: radius >= edge_frac * max die radius. Both
         target patterns live at the edge, so restricting here removes centre
         noise and focuses the comparison.
      3. resultant_length R = | (1/N) * sum exp(i*theta) | over edge-failure
         angles theta. R -> 0 for a full uniform ring (Edge-Ring),
         R -> 1 for a tight arc (Edge-Loc).
      4. peak_bin_fraction = densest of `n_bins` angular bins / N.

    Parameters
    ----------
    wafer : array-like of shape (H, W)
    edge_frac : float
        Radius threshold (as a fraction of max die radius) defining "edge".
    n_bins : int
        Number of angular bins for the peak-bin statistic.

    Returns
    -------
    np.ndarray of shape (2,), dtype float64
        [resultant_length, peak_bin_fraction], each in [0, 1]. Returns
        [0.0, 0.0] when there are too few edge failures to be meaningful.
    """
    wafer = np.asarray(wafer)

    die_mask = (wafer == 1) | (wafer == FAIL_VALUE)
    fail_mask = wafer == FAIL_VALUE

    dy, dx = np.nonzero(die_mask)
    if dy.size == 0:
        return np.zeros(2, dtype=np.float64)

    cy, cx = dy.mean(), dx.mean()                        # centroid of all dies
    r_max = np.sqrt((dy - cy) ** 2 + (dx - cx) ** 2).max()
    if r_max == 0:
        return np.zeros(2, dtype=np.float64)

    fy, fx = np.nonzero(fail_mask)                        # failed-die coords
    if fy.size == 0:
        return np.zeros(2, dtype=np.float64)

    r_fail = np.sqrt((fy - cy) ** 2 + (fx - cx) ** 2)
    edge = r_fail >= edge_frac * r_max                    # keep edge failures
    if edge.sum() < MIN_EDGE_DEFECTS:
        return np.zeros(2, dtype=np.float64)

    theta = np.arctan2(fy[edge] - cy, fx[edge] - cx)      # angle per edge fail

    # --- circular resultant length R -----------------------------------------
    resultant_length = np.abs(np.exp(1j * theta).mean())

    # --- densest angular bin -------------------------------------------------
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(theta, bins=bins)
    peak_bin_fraction = counts.max() / counts.sum()

    return np.array([resultant_length, peak_bin_fraction], dtype=np.float64)
