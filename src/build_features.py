# =============================================================================
# build_features.py — Assemble the classical feature table (X_feat)
# =============================================================================
# Ties the four feature-group extractors into one row-per-wafer table:
#
#     radial (10)  +  radon (1)  +  connectivity (2)  +  angular (2)
#
# This module owns the TABLE SCHEMA — the column order and names — so the
# feature modules stay pure (wafer in, array out) and this file is the single
# place that decides how they stack. Keeping the schema here means a change to
# column order can never silently desync the names from the values.
#
# CLI:  python -m src.build_features --images X_img.npy --labels y.npy \
#                                    --out X_feat.pkl
#
# WM-811K cell encoding:  0 = no die (off-wafer)   1 = pass   2 = fail
# =============================================================================

import argparse

import numpy as np

from .features.radial import radial_density_profile
from .features.radon import radon_linearity
from .features.connectivity import connectivity_profile
from .features.angular import angular_concentration

DEFAULT_N_RINGS = 10


def feature_names(n_rings: int = DEFAULT_N_RINGS) -> list[str]:
    """Ordered column names, matching the concatenation in extract_features.

    The order here is the single source of truth for the table schema; the
    values in extract_features MUST be concatenated in this same order.
    """
    names = [f"radial_ring_{i:02d}" for i in range(n_rings)]  # group 1: 10 cols
    names += ["radon_linearity"]                              # group 2: 1 col
    names += ["conn_largest_frac", "conn_fragmentation"]      # group 3: 2 cols
    names += ["ang_resultant_len", "ang_peak_bin"]            # group 4: 2 cols
    return names


def extract_features(wafer, n_rings: int = DEFAULT_N_RINGS) -> np.ndarray:
    """Return the full feature vector for one wafer map.

    Concatenation order is fixed and must mirror feature_names():
      1. radial density profile      -> n_rings values
      2. radon linearity             -> 1 value
      3. connectivity profile        -> 2 values
      4. angular concentration       -> 2 values
    """
    radial = radial_density_profile(wafer, n_rings=n_rings)   # (n_rings,)
    radon = np.array([radon_linearity(wafer)])                # (1,)
    conn = connectivity_profile(wafer)                        # (2,)
    angular = angular_concentration(wafer)                    # (2,)

    return np.concatenate([radial, radon, conn, angular]).astype(np.float64)


def build_feature_table(
    images,
    n_rings: int = DEFAULT_N_RINGS,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the (N, F) feature matrix from an iterable of wafer maps.

    Logic:
      1. Allocate an (N, F) float64 matrix from the known schema width.
      2. Extract each wafer's features into its row — row i of the output is
         row-aligned with images[i], so it stays aligned with y.npy.
      3. Assert the whole matrix is finite. A NaN/inf here would poison
         training silently, so fail loudly instead (verify-before-you-trust).

    Returns
    -------
    (X, names) : (np.ndarray of shape (N, F), list[str] of length F)
    """
    names = feature_names(n_rings)
    n = len(images)
    X = np.empty((n, len(names)), dtype=np.float64)

    for i in range(n):
        X[i] = extract_features(images[i], n_rings=n_rings)
        if verbose and (i + 1) % 10_000 == 0:
            print(f"  ...{i + 1:>7,} / {n:,} wafers")

    if not np.all(np.isfinite(X)):                            # loud, not silent
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(
            f"Non-finite value(s) in feature table at (row, col): "
            f"{bad[:5].tolist()}{' ...' if len(bad) > 5 else ''}"
        )

    return X, names


# =============================================================================
# CLI — reproduce the feature table end-to-end from saved arrays
# =============================================================================
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the classical feature table from a wafer image tensor."
    )
    parser.add_argument("--images", default="X_img.npy",
                        help="Path to the (N, H, W) uint8 wafer tensor.")
    parser.add_argument("--labels", default=None,
                        help="Optional (N,) label array, only used to check "
                             "row-count alignment.")
    parser.add_argument("--out", default="X_feat.pkl",
                        help="Output path for the pickled feature DataFrame.")
    parser.add_argument("--n-rings", type=int, default=DEFAULT_N_RINGS)
    args = parser.parse_args()

    import pandas as pd

    images = np.load(args.images)
    print(f"Loaded images: {images.shape}")

    if args.labels is not None:                              # alignment guard
        y = np.load(args.labels, allow_pickle=True)
        if len(y) != len(images):
            raise ValueError(
                f"Row mismatch: {len(images)} images vs {len(y)} labels."
            )
        print(f"Loaded labels: {len(y):,} (row-aligned)")

    print("Building feature table...")
    X, names = build_feature_table(images, n_rings=args.n_rings, verbose=True)

    df = pd.DataFrame(X, columns=names)
    df.to_pickle(args.out)
    print(f"Wrote {df.shape[0]:,} x {df.shape[1]} feature table -> {args.out}")


if __name__ == "__main__":
    _main()
