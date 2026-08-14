# =============================================================================
# test_build_features.py — Unit tests for the feature-table assembler
# =============================================================================
# These test the SCHEMA and the WIRING, not the individual features (each
# group has its own test file):
#   - names and values stay the same length and in the same order
#   - the table is finite and correctly shaped
#   - row i of the table belongs to wafer i (alignment with y.npy)
#   - a NaN anywhere is raised, never silently written
# Run with:  pytest -q
# =============================================================================

import numpy as np
import pytest

from src.build_features import (
    extract_features,
    feature_names,
    build_feature_table,
)


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


def make_center(size: int = 64):
    wafer = make_disk_wafer(size)
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    wafer[((yy - c) ** 2 + (xx - c) ** 2 <= (size * 0.15) ** 2) & (wafer == 1)] = 2
    return wafer


def make_scratch(size: int = 64):
    wafer = make_disk_wafer(size)
    wafer[32, 12:52] = np.where(wafer[32, 12:52] > 0, 2, 0)
    return wafer


# -----------------------------------------------------------------------------
# Schema tests — names and values must never desync
# -----------------------------------------------------------------------------
def test_vector_length_matches_names():
    for n_rings in (5, 10, 20):
        vec = extract_features(make_center(), n_rings=n_rings)
        names = feature_names(n_rings)
        assert vec.shape == (len(names),)


def test_names_are_unique_and_ordered_by_group():
    names = feature_names(10)
    assert len(names) == len(set(names))                 # no duplicate columns
    assert names[:10] == [f"radial_ring_{i:02d}" for i in range(10)]
    assert names[10] == "radon_linearity"
    assert names[-2:] == ["ang_resultant_len", "ang_peak_bin"]


def test_default_schema_width():
    """Pins the current column count so an accidental schema change trips a
    test. NOTE: this is 15 for the reconstructed extractors; set to your real
    table width (16) once you reconcile the modules with your notebook."""
    assert len(feature_names()) == 15


# -----------------------------------------------------------------------------
# Table-building tests
# -----------------------------------------------------------------------------
def test_table_shape_and_finiteness():
    images = np.stack([make_center(), make_scratch(), make_disk_wafer()])
    X, names = build_feature_table(images)
    assert X.shape == (3, len(names))
    assert np.all(np.isfinite(X))


def test_rows_are_aligned_to_input_order():
    """Row i must describe images[i]. Wafer 0 is a scratch (high linearity);
    wafer 1 is a centre cluster (low linearity)."""
    images = np.stack([make_scratch(), make_center()])
    X, names = build_feature_table(images)
    radon_col = names.index("radon_linearity")
    assert X[0, radon_col] > 0.80                        # scratch row
    assert X[1, radon_col] < 0.35                        # centre row


def test_center_wafer_loads_inner_radial_columns():
    """End-to-end sanity: assembly preserves the radial group's orientation."""
    X, names = build_feature_table(np.stack([make_center()]))
    inner = names.index("radial_ring_00")
    outer = names.index("radial_ring_09")
    assert X[0, inner] > X[0, outer]


def test_nonfinite_value_is_raised_not_written():
    """A wafer that produced a NaN must abort the build loudly."""
    class Sneaky:
        """Length-1 iterable whose single item forces a NaN into the table."""
        def __len__(self): return 1
        def __getitem__(self, i):
            w = make_disk_wafer()
            w[32, 12:52] = np.where(w[32, 12:52] > 0, 2, 0)
            return w

    # Monkeypatch one extractor to emit NaN, then confirm the guard fires.
    import src.build_features as bf
    original = bf.radon_linearity
    bf.radon_linearity = lambda wafer: np.nan
    try:
        with pytest.raises(ValueError, match="Non-finite"):
            build_feature_table(np.stack([make_scratch()]))
    finally:
        bf.radon_linearity = original                    # always restore


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
