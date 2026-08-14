# =============================================================================
# radon.py — Radon-based linearity feature (feature group 2 of 4)
# =============================================================================
# Scores how "line-like" a wafer's failure mask is. This is the Scratch
# detector: a scratch is a thin, collimated track of failed dies, whereas
# Random / Center / Edge-Ring defects are areal blobs.
#
# Physical motivation
# -------------------
# The Radon transform integrates the mask along parallel rays at many
# angles. When the projection direction is PARALLEL to a scratch, every
# die on the line piles into (almost) one offset bin, producing a tall,
# sharp peak. A diffuse blob spreads its mass across many offset bins at
# every angle, so no sharp peak ever appears. We therefore measure, at the
# best angle, what fraction of that angle's projected mass sits in its
# single densest bin:
#
#     linearity = max_theta [ max_offset S(offset, theta)
#                             / sum_offset S(offset, theta) ]
#
# The metric is scale-invariant (normalised by projected mass), bounded in
# (0, 1], and orientation-agnostic (we take the max over all angles), so a
# scratch scores high regardless of how it is rotated on the wafer.
#
# Performance note: the raw mask is downsampled to a small working grid and
# projected at coarse angular steps (3 deg by default). Radon cost grows
# with pixels x angles, and neither the downsample nor the coarse step
# changes which pattern is most linear — they only cut compute.
#
# WM-811K cell encoding:  0 = no die (off-wafer)   1 = pass   2 = fail
# =============================================================================

import numpy as np
from skimage.transform import radon, resize

FAIL_VALUE = 2          # a die that failed electrical test

# Below this many failed dies there is no meaningful "line" to detect; a
# one- or two-die spike would otherwise score a spurious linearity of ~1.0.
MIN_DEFECTS = 4


def radon_linearity(
    wafer,
    work_size: int = 48,
    angle_step: float = 3.0,
) -> float:
    """Return a scalar in [0, 1] scoring how line-like the failure mask is.

    Logic:
      1. Extract the failure mask (cells == FAIL_VALUE) as a float image.
      2. If fewer than MIN_DEFECTS dies failed, return 0.0 — too sparse to
         be a scratch.
      3. Downsample the mask to `work_size` x `work_size` (nearest-neighbour,
         no anti-aliasing) purely to bound Radon compute, then re-binarise.
      4. Radon-project at `angle_step`-degree increments over 0..180 deg.
      5. For each angle, concentration = densest offset bin / total projected
         mass. Return the maximum concentration over all angles.

    Parameters
    ----------
    wafer : array-like of shape (H, W)
        A single wafer map in WM-811K encoding.
    work_size : int
        Side length of the square grid the mask is resized to (default 48).
    angle_step : float
        Angular sampling of the Radon projection in degrees (default 3.0).

    Returns
    -------
    float
        Linearity score. ~1.0 for an ideal collimated scratch, small for an
        areal blob, exactly 0.0 for an empty or near-empty mask.
    """
    wafer = np.asarray(wafer)

    mask = (wafer == FAIL_VALUE).astype(np.float64)     # binary failure image
    if mask.sum() < MIN_DEFECTS:                        # too few dies for a line
        return 0.0

    # --- downsample to bound Radon cost (order=0 keeps the mask binary) ------
    if max(mask.shape) > work_size:
        mask = resize(
            mask,
            (work_size, work_size),
            order=0,                                    # nearest-neighbour
            anti_aliasing=False,
            preserve_range=True,
        )
        mask = (mask > 0.5).astype(np.float64)          # re-binarise
        if mask.sum() < MIN_DEFECTS:                    # line lost in downsample
            return 0.0

    # --- project and score peak concentration per angle ----------------------
    theta = np.arange(0.0, 180.0, angle_step)           # e.g. 60 angles at 3 deg
    sinogram = radon(mask, theta=theta, circle=False)   # (offsets, n_angles)

    col_sums = sinogram.sum(axis=0)                     # ~projected mass / angle
    col_peaks = sinogram.max(axis=0)                    # densest bin / angle

    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = np.where(col_sums > 0, col_peaks / col_sums, 0.0)

    return float(concentration.max())                   # best (most linear) angle
