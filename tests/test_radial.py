# =============================================================================
# test_radial.py — Unit tests for the radial density feature
# =============================================================================
# Each test encodes a physical expectation about the feature, so a reviewer
# can read the assertions as a specification of what "radial density" means:
#   - a centre defect must load the inner rings
#   - an edge defect must load the outer rings
#   - the output must always be finite and bounded, even on degenerate input
# Run with:  pytest -q
# =============================================================================

import numpy as np
import pytest

from src.features.radial import radial_density_profile


# -----------------------------------------------------------------------------
# Fixtures / builders — synthetic wafers with a known ground truth
# -----------------------------------------------------------------------------
def make_disk_wafer(size: int = 64, radius: float | None = None) -> np.ndarray:
    """Return an all-pass circular wafer (1 inside the disk, 0 outside)."""
    if radius is None:
        radius = size / 2 - 1
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[disk] = 1
    return wafer


def annulus_mask(size: int, r_inner: float, r_outer: float) -> np.ndarray:
    """Boolean ring between r_inner and r_outer, centred on the grid."""
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    d2 = (yy - c) ** 2 + (xx - c) ** 2
    return (d2 >= r_inner ** 2) & (d2 <= r_outer ** 2)


# -----------------------------------------------------------------------------
# Shape / contract tests — the feature must always be well-formed
# -----------------------------------------------------------------------------
def test_output_shape_matches_n_rings():
    wafer = make_disk_wafer()
    for n in (5, 10, 20):
        assert radial_density_profile(wafer, n_rings=n).shape == (n,)


def test_empty_wafer_returns_zeros():
    wafer = np.zeros((64, 64), dtype=np.uint8)          # no dies at all
    out = radial_density_profile(wafer, n_rings=10)
    assert np.array_equal(out, np.zeros(10))


def test_all_pass_wafer_has_zero_density():
    wafer = make_disk_wafer()                            # dies present, no fails
    out = radial_density_profile(wafer)
    assert np.allclose(out, 0.0)


def test_density_is_finite_and_bounded():
    """No NaN/inf, and density is a fraction in [0, 1] by construction."""
    rng = np.random.default_rng(0)
    wafer = make_disk_wafer()
    die = wafer == 1
    wafer[die & (rng.random(wafer.shape) < 0.3)] = 2     # 30% random fails
    out = radial_density_profile(wafer)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_fully_failed_wafer_is_density_one_where_dies_exist():
    wafer = make_disk_wafer()
    wafer[wafer == 1] = 2                                 # every die fails
    out = radial_density_profile(wafer)
    populated = out > 0
    assert np.allclose(out[populated], 1.0)              # each such ring == 1.0


# -----------------------------------------------------------------------------
# Physical-signature tests — the whole reason the feature exists
# -----------------------------------------------------------------------------
def test_center_pattern_loads_inner_rings():
    """A central defect cluster must give inner-ring density > outer-ring."""
    size = 64
    wafer = make_disk_wafer(size)
    center = annulus_mask(size, 0, size * 0.15)          # small central blob
    wafer[(wafer == 1) & center] = 2
    out = radial_density_profile(wafer, n_rings=10)
    assert out[0] > out[-1]
    assert out[:3].mean() > out[-3:].mean()


def test_edge_ring_pattern_loads_outer_rings():
    """An edge-ring defect must give outer-ring density > inner-ring."""
    size = 64
    wafer = make_disk_wafer(size)
    edge = annulus_mask(size, size * 0.42, size * 0.5)   # thin boundary ring
    wafer[(wafer == 1) & edge] = 2
    out = radial_density_profile(wafer, n_rings=10)
    assert out[-1] > out[0]
    assert out[-3:].mean() > out[:3].mean()


def test_center_and_edge_profiles_are_distinguishable():
    """The two patterns must produce clearly different vectors (the property
    a downstream classifier relies on to tell Center from Edge-Ring)."""
    size = 64
    center_w = make_disk_wafer(size)
    center_w[(center_w == 1) & annulus_mask(size, 0, size * 0.15)] = 2

    edge_w = make_disk_wafer(size)
    edge_w[(edge_w == 1) & annulus_mask(size, size * 0.42, size * 0.5)] = 2

    c = radial_density_profile(center_w)
    e = radial_density_profile(edge_w)
    # inner-heavy vs outer-heavy: argmax should sit on opposite ends
    assert np.argmax(c) < np.argmax(e)


if __name__ == "__main__":                               # allows plain `python`
    raise SystemExit(pytest.main([__file__, "-q"]))
