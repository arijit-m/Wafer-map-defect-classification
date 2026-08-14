# =============================================================================
# test_radon.py — Unit tests for the Radon linearity (scratch) feature
# =============================================================================
# The assertions encode the physical claim the feature makes:
#   - a straight track of failed dies is LINEAR (score near 1), at any angle
#   - an areal blob, a random scatter, or a centre cluster is NOT linear
#   - the score is finite, bounded in [0, 1], and 0 on empty / sparse masks
# Thresholds below were calibrated on the synthetic shapes in this file:
# thin scratches measured 0.90-1.00, blobs/scatter/centre measured 0.09-0.17.
# Run with:  pytest -q
# =============================================================================

import numpy as np
import pytest

from src.features.radon import radon_linearity


# -----------------------------------------------------------------------------
# Builders — synthetic wafers with a known ground truth
# -----------------------------------------------------------------------------
def make_disk_wafer(size: int = 64, radius: float | None = None) -> np.ndarray:
    """All-pass circular wafer (1 inside the disk, 0 outside)."""
    if radius is None:
        radius = size / 2 - 1
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[disk] = 1
    return wafer


def add_line(wafer, y0, x0, y1, x1):
    """Mark a 1-die-wide straight track of failures between two die cells."""
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.linspace(y0, y1, n).round().astype(int)
    xs = np.linspace(x0, x1, n).round().astype(int)
    for y, x in zip(ys, xs):
        if wafer[y, x] != 0:                             # stay on the wafer
            wafer[y, x] = 2
    return wafer


# -----------------------------------------------------------------------------
# Contract tests — the feature must always be well-formed
# -----------------------------------------------------------------------------
def test_returns_finite_float_in_unit_interval():
    wafer = add_line(make_disk_wafer(), 32, 12, 32, 51)
    out = radon_linearity(wafer)
    assert isinstance(out, float)
    assert np.isfinite(out)
    assert 0.0 <= out <= 1.0


def test_empty_wafer_scores_zero():
    wafer = np.zeros((64, 64), dtype=np.uint8)
    assert radon_linearity(wafer) == 0.0


def test_all_pass_wafer_scores_zero():
    assert radon_linearity(make_disk_wafer()) == 0.0


def test_below_min_defects_scores_zero():
    """A two-die spike is not a scratch and must not score a spurious 1.0."""
    wafer = make_disk_wafer()
    wafer[32, 30] = 2
    wafer[32, 31] = 2                                    # only 2 failed dies
    assert radon_linearity(wafer) == 0.0


# -----------------------------------------------------------------------------
# Physical-signature tests — the reason the feature exists
# -----------------------------------------------------------------------------
def test_thin_scratch_scores_high():
    wafer = add_line(make_disk_wafer(), 32, 12, 32, 51)  # horizontal track
    assert radon_linearity(wafer) > 0.80


def test_scratch_beats_blob_by_a_wide_margin():
    scratch = add_line(make_disk_wafer(), 32, 12, 32, 51)

    blob = make_disk_wafer()
    blob[26:38, 26:38][blob[26:38, 26:38] == 1] = 2      # compact square

    s = radon_linearity(scratch)
    b = radon_linearity(blob)
    assert b < 0.35
    assert s - b > 0.40                                  # unambiguous separation


def test_random_scatter_is_not_linear():
    rng = np.random.default_rng(0)
    wafer = make_disk_wafer()
    die = np.argwhere(wafer == 1)
    for y, x in die[rng.choice(len(die), 40, replace=False)]:
        wafer[y, x] = 2
    assert radon_linearity(wafer) < 0.35


def test_center_cluster_is_not_linear():
    size = 64
    wafer = make_disk_wafer(size)
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    wafer[((yy - c) ** 2 + (xx - c) ** 2 <= (size * 0.15) ** 2) & (wafer == 1)] = 2
    assert radon_linearity(wafer) < 0.35


def test_linearity_is_orientation_invariant():
    """Horizontal, vertical, and diagonal scratches must all score high and
    close together — the feature detects lineness, not a preferred angle."""
    horiz = radon_linearity(add_line(make_disk_wafer(), 32, 12, 32, 51))
    vert = radon_linearity(add_line(make_disk_wafer(), 12, 32, 51, 32))
    diag = radon_linearity(add_line(make_disk_wafer(), 14, 14, 49, 49))

    for score in (horiz, vert, diag):
        assert score > 0.80
    assert max(horiz, vert, diag) - min(horiz, vert, diag) < 0.20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
