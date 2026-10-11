"""
preprocess.py — Smart Home Appliance Energy Predictor
=====================================================
Reproducible preprocessing module for the UCI Appliances Energy Prediction
dataset (19 735 rows × 28 features).

Responsibilities
----------------
1. Download / load the dataset from the UCI archive.
2. Inspect data types, missing values, and duplicates.
3. Drop random-noise columns 'rv1' and 'rv2'.
4. Separate features (X) from target (y = 'Appliances').
5. Train-Test split — two strategies:
   • Random split  (shuffle=True, stratify=None)
   • Chronological / time-based split (preserves temporal order)
6. Fit StandardScaler on the training set only, then transform both sets.
7. Persist processed artefacts to  data/processed/.

Author : Phase-1 scaffolding
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/374/"
    "appliances+energy+prediction.zip"
)
TARGET_COL = "Appliances"
DROP_COLS = ["rv1", "rv2"]
DATE_COL = "date"

# Resolve project root relative to *this* file (src/preprocess.py → ..)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# 1.  Download / Load
# ---------------------------------------------------------------------------
def download_dataset(dest_dir: Path | str = RAW_DATA_DIR) -> Path:
    """Download the UCI zip archive and extract the CSV.

    If the CSV already exists locally it is **not** re-downloaded.

    Returns
    -------
    Path
        Absolute path to the extracted CSV file.
    """
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "energydata_complete.csv"
    if csv_path.exists():
        print(f"[INFO] Dataset already present at {csv_path}")
        return csv_path

    zip_path = dest_dir / "appliances_energy_prediction.zip"
    print(f"[INFO] Downloading dataset from UCI archive …")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print(f"[INFO] Saved zip to {zip_path}")

    # Extract — find the CSV inside the archive
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV found inside the zip archive.")
        # Extract the first CSV found
        target_name = csv_names[0]
        data = zf.read(target_name)
        csv_path.write_bytes(data)
        print(f"[INFO] Extracted '{target_name}' → {csv_path}")

    # Clean up the zip
    zip_path.unlink(missing_ok=True)
    return csv_path


def load_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the Appliances Energy Prediction CSV into a DataFrame.

    Parameters
    ----------
    csv_path : Path or str, optional
        Path to the CSV.  If *None*, ``download_dataset()`` is called first.

    Returns
    -------
    pd.DataFrame
    """
    if csv_path is None:
        csv_path = download_dataset()
    else:
        csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Parse the date column if it exists
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2.  Inspect
# ---------------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame) -> dict:
    """Print and return a summary of data types, missing values, and duplicates.

    Returns
    -------
    dict
        Keys: 'dtypes', 'missing', 'duplicates', 'shape', 'describe'.
    """
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nShape : {df.shape}")
    print(f"\nData types:\n{df.dtypes}")

    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values per column:\n{missing[missing > 0] if total_missing > 0 else 'None'}")
    print(f"Total missing values: {total_missing}")

    n_dups = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dups}")

    print(f"\nDescriptive statistics:\n{df.describe().T.to_string()}")
    print("=" * 60 + "\n")

    return {
        "dtypes": df.dtypes,
        "missing": missing,
        "duplicates": n_dups,
        "shape": df.shape,
        "describe": df.describe(),
    }


# ---------------------------------------------------------------------------
# 3.  Clean — drop rv1, rv2
# ---------------------------------------------------------------------------
def drop_noise_columns(
    df: pd.DataFrame,
    columns: list[str] = DROP_COLS,
) -> pd.DataFrame:
    """Drop the random-variable noise columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Defaults to ``['rv1', 'rv2']``.

    Returns
    -------
    pd.DataFrame
        DataFrame without the specified columns.
    """
    cols_present = [c for c in columns if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        print(f"[INFO] Dropped columns: {cols_present}")
    else:
        print(f"[WARN] None of {columns} found in DataFrame — skipping drop.")
    return df


# ---------------------------------------------------------------------------
# 4.  Feature / Target separation
# ---------------------------------------------------------------------------
def split_features_target(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    exclude: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y.

    The ``date`` column is excluded from X by default (not numeric).

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    exclude : list[str] or None
        Additional columns to drop from X (e.g. ``['date']``).

    Returns
    -------
    (X, y)
    """
    if exclude is None:
        exclude = []

    # Always exclude the date column from features if present
    if DATE_COL in df.columns and DATE_COL not in exclude:
        exclude.append(DATE_COL)

    y = df[target].copy()
    X = df.drop(columns=[target] + [c for c in exclude if c in df.columns])

    print(f"[INFO] Features X: {X.shape}  |  Target y: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# 5.  Train / Test splits
# ---------------------------------------------------------------------------
def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standard random train-test split.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"[INFO] Random split → Train: {X_train.shape[0]}  |  "
        f"Test: {X_test.shape[0]}  (test_size={test_size})"
    )
    return X_train, X_test, y_train, y_test


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-based (chronological) split — preserves temporal order.

    The first ``1 - test_size`` fraction of rows is used for training and
    the remaining rows for testing.  **No shuffling** takes place, which
    prevents future data from leaking into the training set.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(
        f"[INFO] Chronological split → Train: {X_train.shape[0]}  |  "
        f"Test: {X_test.shape[0]}  (test_size={test_size})"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6.  Feature Scaling (fit on train only)
# ---------------------------------------------------------------------------
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_scaler: bool = True,
    scaler_path: Path | str = PROCESSED_DIR / "scaler.joblib",
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on X_train and transform both sets.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    save_scaler : bool
        Persist the fitted scaler via joblib.
    scaler_path : Path or str
        Where to save the scaler artefact.

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
        Scaled DataFrames retain original column names and indices.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    if save_scaler:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved → {scaler_path}")

    print(
        f"[INFO] Scaling complete — "
        f"Train mean ≈ {X_train_scaled.mean().mean():.6f}, "
        f"Train std ≈ {X_train_scaled.std().mean():.6f}"
    )
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 7.  Save processed splits
# ---------------------------------------------------------------------------
def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str = PROCESSED_DIR,
    prefix: str = "",
) -> None:
    """Persist the processed train/test splits as CSV files.

    Files
    -----
    {prefix}X_train.csv, {prefix}X_test.csv,
    {prefix}y_train.csv, {prefix}y_test.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / f"{prefix}X_train.csv", index=False)
    X_test.to_csv(output_dir / f"{prefix}X_test.csv", index=False)
    y_train.to_csv(output_dir / f"{prefix}y_train.csv", index=False)
    y_test.to_csv(output_dir / f"{prefix}y_test.csv", index=False)

    print(f"[INFO] Saved processed splits → {output_dir}/  (prefix='{prefix}')")


# ---------------------------------------------------------------------------
# 8.  End-to-end pipeline convenience function
# ---------------------------------------------------------------------------
def run_pipeline(
    csv_path: str | Path | None = None,
    split_strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    csv_path : str, Path, or None
        Path to the CSV.  If None the dataset is auto-downloaded.
    split_strategy : {'random', 'chronological'}
        Which train-test split strategy to use.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility (random split only).
    scale : bool
        Whether to apply StandardScaler.

    Returns
    -------
    dict
        Keys: 'X_train', 'X_test', 'y_train', 'y_test', 'scaler',
              'inspection', 'df_clean'.
    """
    # Step 1 — Load
    df = load_dataset(csv_path)

    # Step 2 — Inspect
    inspection = inspect_dataset(df)

    # Step 3 — Drop noise columns
    df_clean = drop_noise_columns(df)

    # Step 4 — Feature / target split
    X, y = split_features_target(df_clean)

    # Step 5 — Train / test split
    if split_strategy == "chronological":
        X_train, X_test, y_train, y_test = chronological_split(
            X, y, test_size=test_size
        )
    else:
        X_train, X_test, y_train, y_test = random_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Step 6 — Scale
    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 7 — Save
    prefix = f"{split_strategy}_"
    save_processed(X_train, X_test, y_train, y_test, prefix=prefix)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "inspection": inspection,
        "df_clean": df_clean,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess the UCI Appliances Energy Prediction dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the raw CSV. If omitted the dataset is downloaded.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "chronological", "both"],
        default="both",
        help="Train-test split strategy (default: both).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Skip feature scaling.",
    )

    args = parser.parse_args()

    strategies = (
        ["random", "chronological"] if args.split == "both" else [args.split]
    )

    for strategy in strategies:
        print(f"\n{'━' * 60}")
        print(f"  PIPELINE RUN — strategy: {strategy.upper()}")
        print(f"{'━' * 60}")
        results = run_pipeline(
            csv_path=args.csv,
            split_strategy=strategy,
            test_size=args.test_size,
            random_state=args.seed,
            scale=not args.no_scale,
        )

        # ── Summary statistics ──
        print(f"\n── Summary ({strategy}) ──")
        print(f"  Train samples : {results['X_train'].shape[0]}")
        print(f"  Test samples  : {results['X_test'].shape[0]}")
        print(f"  Features      : {results['X_train'].shape[1]}")
        print(f"  Target (train) — mean: {results['y_train'].mean():.2f}, "
              f"std: {results['y_train'].std():.2f}, "
              f"min: {results['y_train'].min()}, "
              f"max: {results['y_train'].max()}")
        print(f"  Target (test)  — mean: {results['y_test'].mean():.2f}, "
              f"std: {results['y_test'].std():.2f}, "
              f"min: {results['y_test'].min()}, "
              f"max: {results['y_test'].max()}")

    print("\n✅ Preprocessing complete. Artefacts saved to data/processed/")
