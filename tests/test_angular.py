# =============================================================================
# test_angular.py — Unit tests for the angular-concentration feature
# =============================================================================
# This is the hard pair. The tests do two jobs:
#   (1) confirm the feature works on CLEAN cases — a full rim reads as uniform,
#       a tight arc reads as concentrated;
#   (2) DOCUMENT the weakness — a partial ring lands in the ambiguous middle,
#       and the peak-bin statistic separates the classes by a much smaller
#       margin than the other feature groups achieve. This overlap is the
#       Stage-5 Edge-Ring vs Edge-Loc confusion, made explicit rather than
#       hidden.
# Thresholds calibrated on the arcs below: full ring R~0.00 / peak~0.06,
# tight 45deg arc R~0.98 / peak~0.51, half ring R~0.64 (the trap).
# Run with:  pytest -q
# =============================================================================

import numpy as np
import pytest

from src.features.angular import angular_concentration


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


def edge_arc(size, a0_deg, a1_deg, r_in=0.80, r_out=1.0):
    """Fail edge dies whose angle falls in [a0, a1] degrees (wraps at 360)."""
    wafer = make_disk_wafer(size)
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    r_max = size / 2 - 1
    ang = (np.degrees(np.arctan2(yy - c, xx - c)) + 360) % 360
    a0, a1 = a0_deg % 360, a1_deg % 360
    in_arc = (ang >= a0) & (ang <= a1) if a0 <= a1 else (ang >= a0) | (ang <= a1)
    band = (wafer == 1) & (r >= r_in * r_max) & (r <= r_out * r_max) & in_arc
    wafer[band] = 2
    return wafer


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------
def test_output_shape_and_bounds():
    out = angular_concentration(edge_arc(64, 40, 85))
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_empty_wafer_returns_zeros():
    assert np.array_equal(angular_concentration(np.zeros((64, 64), np.uint8)),
                          np.zeros(2))


def test_center_pattern_has_no_edge_failures():
    """A centre cluster has no edge failures, so the angular statistic is
    undefined and must return zeros rather than a spurious value."""
    size = 64
    wafer = make_disk_wafer(size)
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    wafer[((yy - c) ** 2 + (xx - c) ** 2 <= (size * 0.15) ** 2) & (wafer == 1)] = 2
    assert np.array_equal(angular_concentration(wafer), np.zeros(2))


# -----------------------------------------------------------------------------
# Clean-case signature tests — the feature works at the extremes
# -----------------------------------------------------------------------------
def test_full_edge_ring_is_uniform():
    r, peak = angular_concentration(edge_arc(64, 0, 359))
    assert r < 0.20                                      # near-zero resultant
    assert peak < 0.15                                   # no dominant bin


def test_tight_edge_loc_is_concentrated():
    r, _ = angular_concentration(edge_arc(64, 40, 85))   # 45-degree arc
    assert r > 0.90


def test_ring_and_tight_loc_are_clearly_separated():
    ring_r = angular_concentration(edge_arc(64, 0, 359))[0]
    loc_r = angular_concentration(edge_arc(64, 20, 110))[0]
    assert loc_r - ring_r > 0.50                         # clean cases separate


def test_resultant_length_is_rotation_invariant():
    """The same-width arc at two positions must score the same concentration."""
    a = angular_concentration(edge_arc(64, 40, 85))[0]
    b = angular_concentration(edge_arc(64, 200, 245))[0]
    assert abs(a - b) < 0.10


# -----------------------------------------------------------------------------
# Honest weakness tests — the documented hard pair
# -----------------------------------------------------------------------------
def test_partial_ring_lands_in_the_ambiguous_middle():
    """A half-ring is neither a clean rim nor a clean arc. It scores between
    the two, so no single threshold cleanly splits the Edge family — this is
    the geometric root of the Stage-5 Edge-Ring/Edge-Loc confusion."""
    ring_r = angular_concentration(edge_arc(64, 0, 359))[0]      # ~0.00
    half_r = angular_concentration(edge_arc(64, 0, 180))[0]      # ~0.64
    tight_r = angular_concentration(edge_arc(64, 40, 85))[0]     # ~0.98
    assert ring_r + 0.20 < half_r < tight_r - 0.10               # genuinely mid

def test_peak_bin_separates_the_pair_only_weakly():
    """The peak-bin statistic — the original angular feature — distinguishes a
    moderate Edge-Loc arc from a full Edge-Ring by a small margin, far less
    than connectivity (>0.6) or radon (>0.4) achieve on their target pairs.
    Retained as the best available angular signal, not a decisive one."""
    ring_peak = angular_concentration(edge_arc(64, 0, 359))[1]
    loc_peak = angular_concentration(edge_arc(64, 20, 110))[1]   # 90-degree arc
    assert loc_peak > ring_peak                                  # direction holds
    assert loc_peak - ring_peak < 0.30                           # but margin is small


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
