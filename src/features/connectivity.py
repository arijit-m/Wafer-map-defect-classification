# =============================================================================
# connectivity.py — Connected-component features (feature group 3 of 4)
# =============================================================================
# Separates Near-full from Random by asking a single structural question:
# is the failure mask ONE big blob, or MANY little specks?
#
# Physical motivation
# -------------------
#   - Near-full : nearly every die failed -> one component covers almost the
#                 whole wafer. Largest-component share -> 1, fragmentation -> 0.
#   - Random    : failures are scattered singletons -> many components, none
#                 dominant. Largest-component share small, fragmentation -> 1.
#
# Note on scope: this feature reports HOW the failures are grouped, not WHERE
# they sit. A tight Center cluster and a Near-full wafer both read as "one
# dominant component" here — the radial profile is what tells those two
# apart. Keeping each feature group to one physical question is deliberate;
# it makes each one testable in isolation.
#
# Connectivity is 8-way (diagonally touching dies belong to one component),
# the convention for wafer-map defect blobs.
#
# WM-811K cell encoding:  0 = no die (off-wafer)   1 = pass   2 = fail
# =============================================================================

import numpy as np
from skimage.measure import label

FAIL_VALUE = 2          # a die that failed electrical test


def connectivity_profile(wafer) -> np.ndarray:
    """Return [largest_fraction, fragmentation] for a wafer's failure mask.

    Logic:
      1. Extract the failure mask (cells == FAIL_VALUE).
      2. Label 8-connected components of failed dies.
      3. largest_fraction = size of the biggest component / total failed dies.
         -> near 1.0 for one dominant blob (Near-full, Center, Scratch),
            small when failures fragment into many pieces (Random).
      4. fragmentation   = number of components / total failed dies.
         -> near 1.0 when every failure is its own speck (Random),
            near 0.0 when all failures fuse into one blob (Near-full).

    Parameters
    ----------
    wafer : array-like of shape (H, W)
        A single wafer map in WM-811K encoding.

    Returns
    -------
    np.ndarray of shape (2,), dtype float64
        [largest_fraction, fragmentation], each in [0, 1]. An empty mask
        yields [0.0, 0.0].
    """
    wafer = np.asarray(wafer)

    fail_mask = wafer == FAIL_VALUE                     # bool grid
    n_fail = int(fail_mask.sum())
    if n_fail == 0:                                     # nothing failed
        return np.zeros(2, dtype=np.float64)

    labelled = label(fail_mask, connectivity=2)         # 8-connectivity
    component_sizes = np.bincount(labelled.ravel())[1:]  # drop background (0)

    n_components = component_sizes.size
    largest_fraction = component_sizes.max() / n_fail   # dominant blob share
    fragmentation = n_components / n_fail                # specks per failed die

    return np.array([largest_fraction, fragmentation], dtype=np.float64)
