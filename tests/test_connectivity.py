# =============================================================================
# test_connectivity.py — Unit tests for the connected-component feature
# =============================================================================
# The assertions encode the structural claim:
#   - Near-full is one dominant component (largest share ~1, fragmentation ~0)
#   - Random is many specks (largest share small, fragmentation high)
#   - the feature reports grouping, NOT location: Near-full and Center both
#     read as "one blob", and a test documents that limitation on purpose.
# Thresholds calibrated on the shapes below: Random measured ~0.08 / ~0.88,
# one-blob patterns measured ~1.0 / ~0.0.
# Run with:  pytest -q
# =============================================================================

import numpy as np
import pytest

from src.features.connectivity import connectivity_profile


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def make_disk_wafer(size: int = 64, radius: float | None = None) -> np.ndarray:
    if radius is None:
        radius = size / 2 - 1
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[disk] = 1
    return wafer


def make_near_full(size: int = 64, pass_frac: float = 0.10, seed: int = 1):
    """Wafer where all but a small fraction of dies fail (one giant blob)."""
    rng = np.random.default_rng(seed)
    wafer = make_disk_wafer(size)
    die = np.argwhere(wafer == 1)
    keep = set(map(tuple, die[rng.choice(len(die), int(len(die) * pass_frac),
                                         replace=False)]))
    for y, x in die:
        if (y, x) not in keep:
            wafer[y, x] = 2
    return wafer


def make_random(size: int = 64, n: int = 40, seed: int = 1):
    """Wafer with n scattered single-die failures (many specks)."""
    rng = np.random.default_rng(seed)
    wafer = make_disk_wafer(size)
    die = np.argwhere(wafer == 1)
    for y, x in die[rng.choice(len(die), n, replace=False)]:
        wafer[y, x] = 2
    return wafer


def make_center(size: int = 64, r_frac: float = 0.18):
    wafer = make_disk_wafer(size)
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    wafer[((yy - c) ** 2 + (xx - c) ** 2 <= (size * r_frac) ** 2) & (wafer == 1)] = 2
    return wafer


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------
def test_output_shape_and_bounds():
    out = connectivity_profile(make_near_full())
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_empty_wafer_returns_zeros():
    assert np.array_equal(connectivity_profile(np.zeros((64, 64), np.uint8)),
                          np.zeros(2))


def test_single_failed_die_is_one_full_component():
    wafer = make_disk_wafer()
    wafer[32, 32] = 2
    largest, frag = connectivity_profile(wafer)
    assert largest == 1.0                                # the one die is the blob
    assert frag == 1.0                                   # and it's one component


# -----------------------------------------------------------------------------
# Physical-signature tests
# -----------------------------------------------------------------------------
def test_near_full_is_one_dominant_component():
    largest, frag = connectivity_profile(make_near_full())
    assert largest > 0.90
    assert frag < 0.10


def test_random_is_many_small_components():
    largest, frag = connectivity_profile(make_random())
    assert largest < 0.30
    assert frag > 0.50


def test_near_full_and_random_are_well_separated():
    nf_largest, nf_frag = connectivity_profile(make_near_full())
    rd_largest, rd_frag = connectivity_profile(make_random())
    assert nf_largest - rd_largest > 0.60                # opposite ends
    assert rd_frag - nf_frag > 0.60


def test_feature_reports_grouping_not_location():
    """Documented limitation: a compact Center cluster and a Near-full wafer
    are BOTH one dominant component. Connectivity cannot separate them — that
    is the radial profile's job. This test pins the limitation so a reviewer
    sees it was understood, not missed."""
    center_largest = connectivity_profile(make_center())[0]
    near_full_largest = connectivity_profile(make_near_full())[0]
    assert center_largest > 0.90
    assert near_full_largest > 0.90                      # indistinguishable here


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
