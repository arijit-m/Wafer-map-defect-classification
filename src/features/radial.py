# =============================================================================
# radial.py — Radial defect-density features (feature group 1 of 4)
# =============================================================================
# Computes a concentric-ring density profile of a wafer's failure mask.
#
# Physical motivation
# -------------------
# Radially organised defect patterns leave a signature in how defect
# density varies with distance from the wafer centre:
#   - Center      -> density concentrated in the INNER rings
#   - Edge-Ring   -> density concentrated in the OUTER rings
#   - Ring/Donut  -> density peaks in the MIDDLE rings
#   - Random/none -> density roughly flat across all rings
# A pure data-science model sees 10 numbers; a process engineer reads them
# as "where on the wafer is this failing", which is the first question in
# any tool investigation.
#
# WM-811K cell encoding:  0 = no die (off-wafer)   1 = pass   2 = fail
# =============================================================================

import numpy as np

# Cell-value semantics, named so the intent is testable and self-documenting.
DIE_VALUES = (1, 2)   # any physical die sits in the wafer grid (pass or fail)
FAIL_VALUE = 2        # a die that failed electrical test


def radial_density_profile(wafer, n_rings: int = 10) -> np.ndarray:
    """Return the per-ring failure density of a single wafer map.

    Logic:
      1. Identify die cells (value 1 or 2) and failed cells (value 2).
      2. Take the centroid of the die cells as the wafer centre — robust to
         off-centre or partially-populated wafers, unlike a fixed geometric
         centre.
      3. Compute each die's radius from that centroid, normalise to [0, 1]
         by the largest radius, and bin into `n_rings` equal-width rings.
      4. For each ring, density = (failed dies) / (total dies). Empty rings
         return 0.0 rather than NaN so the feature vector is always finite.

    Parameters
    ----------
    wafer : array-like of shape (H, W)
        A single wafer map in WM-811K encoding.
    n_rings : int
        Number of concentric rings (default 10).

    Returns
    -------
    np.ndarray of shape (n_rings,), dtype float64
        Failure density per ring, ordered inner -> outer. Each value is in
        [0, 1]. An empty or die-less wafer yields an all-zero vector.
    """
    wafer = np.asarray(wafer)

    die_mask = (wafer == DIE_VALUES[0]) | (wafer == DIE_VALUES[1])  # bool grid
    fail_mask = wafer == FAIL_VALUE                                 # bool grid

    ys, xs = np.nonzero(die_mask)                # coords of every die cell
    if ys.size == 0:                             # no dies -> nothing to score
        return np.zeros(n_rings, dtype=np.float64)

    cy, cx = ys.mean(), xs.mean()                          # centroid of dies
    radius = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)      # radius per die
    r_max = radius.max()

    if r_max == 0:                               # all dies coincide (degenerate)
        ring_idx = np.zeros_like(radius, dtype=int)
    else:
        # Scale radius to [0, n_rings); clip the r == r_max point (which would
        # otherwise land in ring index n_rings) back into the outermost ring.
        ring_idx = np.minimum(
            (radius / r_max * n_rings).astype(int), n_rings - 1
        )

    fail_at_die = fail_mask[ys, xs]              # bool per die, aligned to rings

    dies_per_ring = np.bincount(ring_idx, minlength=n_rings)
    fails_per_ring = np.bincount(ring_idx, weights=fail_at_die, minlength=n_rings)

    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(dies_per_ring > 0, fails_per_ring / dies_per_ring, 0.0)

    return density.astype(np.float64)
