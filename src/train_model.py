"""
train_model.py — Smart Home Appliance Energy Predictor (Phase 3)
================================================================
Train, evaluate, and compare four candidate regression models on the
UCI Appliances Energy Prediction dataset.

Models
------
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. HistGradientBoosting Regressor

Evaluation Metrics
------------------
• MAE  (Mean Absolute Error, Wh)
• RMSE (Root Mean Squared Error, Wh)
• R²   (Coefficient of Determination)

Outputs
-------
figures/fig5_mae_comparison.png          – Bar chart: MAE across models
figures/fig6_rmse_comparison.png         – Bar chart: RMSE across models
figures/fig7_actual_vs_predicted.png     – Scatter: Actual vs Predicted (best model)
figures/model_comparison_results.md      – Markdown table of all metrics
models/best_appliance_model.joblib       – Serialised scaler + best model pipeline

Usage
-----
    python src/train_model.py                         # random split (default)
    python src/train_model.py --split chronological   # time-based split
    python src/train_model.py --csv data/energydata_complete.csv
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Import Phase-1 preprocessing pipeline
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocess import (
    load_dataset,
    drop_noise_columns,
    split_features_target,
    random_split,
    chronological_split,
    scale_features,
    TARGET_COL,
)

# ── Plot styling ──────────────────────────────────────────────────────────
DPI = 300
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.15,
    rc={
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.titleweight": "bold",
        "font.family": "sans-serif",
    },
)

PALETTE = {
    "Linear Regression": "#4361EE",
    "Decision Tree": "#F72585",
    "Random Forest": "#7209B7",
    "HistGradientBoosting": "#4CC9F0",
}
MODEL_NAMES = list(PALETTE.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  1. Build candidate models
# ═══════════════════════════════════════════════════════════════════════════
def build_models() -> Dict[str, object]:
    """Return an ordered dict of model name → unfitted estimator."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  2. Evaluate a single model
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float | np.ndarray]:
    """Fit a model on training data, predict on test, return metrics + predictions."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3. Train & evaluate all models
# ═══════════════════════════════════════════════════════════════════════════
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, object]]:
    """Train all candidate models and collect evaluation metrics.

    Returns
    -------
    results_df : pd.DataFrame
        One row per model with MAE, RMSE, R² columns.
    predictions : dict
        model_name → {'MAE', 'RMSE', 'R2', 'y_pred'}.
    fitted_models : dict
        model_name → fitted estimator.
    """
    models = build_models()
    predictions: Dict[str, dict] = {}
    fitted_models: Dict[str, object] = {}

    print("\n" + "═" * 65)
    print("  MODEL TRAINING & EVALUATION")
    print("═" * 65)

    for name, model in models.items():
        print(f"\n▸ Training {name} …", end=" ")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        predictions[name] = metrics
        fitted_models[name] = model
        print(
            f"Done  │  MAE={metrics['MAE']:.2f}  "
            f"RMSE={metrics['RMSE']:.2f}  "
            f"R²={metrics['R2']:.4f}"
        )

    # Build results DataFrame
    rows = []
    for name in MODEL_NAMES:
        m = predictions[name]
        rows.append({"Model": name, "MAE (Wh)": m["MAE"], "RMSE (Wh)": m["RMSE"], "R²": m["R2"]})
    results_df = pd.DataFrame(rows)

    print("\n" + "─" * 65)
    print(results_df.to_string(index=False))
    print("─" * 65)

    return results_df, predictions, fitted_models


# ═══════════════════════════════════════════════════════════════════════════
#  4. Identify best model (lowest RMSE)
# ═══════════════════════════════════════════════════════════════════════════
def identify_best_model(
    results_df: pd.DataFrame,
    fitted_models: Dict[str, object],
) -> Tuple[str, object]:
    """Return (name, fitted_estimator) of the best model by RMSE."""
    best_idx = results_df["RMSE (Wh)"].idxmin()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = fitted_models[best_name]
    print(f"\n🏆 Best model: {best_name}  (RMSE = {results_df.loc[best_idx, 'RMSE (Wh)']:.2f} Wh)")
    return best_name, best_model


# ═══════════════════════════════════════════════════════════════════════════
#  5. Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════
def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_mae_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing MAE across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[m] for m in results_df["Model"]]

    bars = ax.bar(
        results_df["Model"], results_df["MAE (Wh)"],
        color=colors, edgecolor="white", linewidth=1.2, width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, results_df["MAE (Wh)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Mean Absolute Error (MAE) Comparison", fontsize=14, pad=12)
    ax.set_ylabel("MAE (Wh)")
    ax.set_xlabel("")
    ax.set_ylim(0, results_df["MAE (Wh)"].max() * 1.18)
    ax.tick_params(axis="x", rotation=15)

    # Highlight best
    best_idx = results_df["MAE (Wh)"].idxmin()
    bars[best_idx].set_edgecolor("#2D6A4F")
    bars[best_idx].set_linewidth(3)

    sns.despine(left=True)
    _save(fig, "fig5_mae_comparison.png")


def plot_rmse_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing RMSE across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[m] for m in results_df["Model"]]

    bars = ax.bar(
        results_df["Model"], results_df["RMSE (Wh)"],
        color=colors, edgecolor="white", linewidth=1.2, width=0.6,
    )

    for bar, val in zip(bars, results_df["RMSE (Wh)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Root Mean Squared Error (RMSE) Comparison", fontsize=14, pad=12)
    ax.set_ylabel("RMSE (Wh)")
    ax.set_xlabel("")
    ax.set_ylim(0, results_df["RMSE (Wh)"].max() * 1.18)
    ax.tick_params(axis="x", rotation=15)

    best_idx = results_df["RMSE (Wh)"].idxmin()
    bars[best_idx].set_edgecolor("#2D6A4F")
    bars[best_idx].set_linewidth(3)

    sns.despine(left=True)
    _save(fig, "fig6_rmse_comparison.png")


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    r2: float,
    rmse: float,
) -> None:
    """Scatter plot of actual vs predicted values for the best model."""
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(
        y_test, y_pred,
        alpha=0.25, s=12,
        color=PALETTE.get(model_name, "#4361EE"),
        edgecolors="none",
        label="Predictions",
    )

    # y = x reference line
    lims = [
        min(y_test.min(), y_pred.min()) - 10,
        max(y_test.max(), y_pred.max()) + 10,
    ]
    ax.plot(lims, lims, "--", color="#E63946", linewidth=2, alpha=0.8, label="y = x (perfect)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual Energy (Wh)", fontsize=12)
    ax.set_ylabel("Predicted Energy (Wh)", fontsize=12)
    ax.set_title(
        f"Actual vs Predicted — {model_name}",
        fontsize=14, pad=12,
    )

    # Metrics annotation
    ax.text(
        0.05, 0.93,
        f"R² = {r2:.4f}\nRMSE = {rmse:.2f} Wh",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    _save(fig, "fig7_actual_vs_predicted.png")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Save results as Markdown table
# ═══════════════════════════════════════════════════════════════════════════
def save_results_markdown(
    results_df: pd.DataFrame,
    best_name: str,
    split_strategy: str,
) -> Path:
    """Write a formatted Markdown comparison table to figures/."""
    md_path = FIGURES_DIR / "model_comparison_results.md"

    header = textwrap.dedent(f"""\
    # Model Comparison Results — Smart Home Appliance Energy Predictor

    **Split strategy:** {split_strategy}
    **Evaluation set:** Test split ({results_df.shape[0]} models evaluated)

    | # | Model | MAE (Wh) | RMSE (Wh) | R² |
    |---|-------|----------|-----------|------|
    """)

    rows = []
    for i, row in results_df.iterrows():
        marker = " 🏆" if row["Model"] == best_name else ""
        rows.append(
            f"| {i+1} | {row['Model']}{marker} | "
            f"{row['MAE (Wh)']:.2f} | {row['RMSE (Wh)']:.2f} | "
            f"{row['R²']:.4f} |"
        )

    footer = textwrap.dedent(f"""

    > **Best model:** {best_name} (lowest RMSE)
    >
    > Serialised pipeline saved to `models/best_appliance_model.joblib`
    """)

    md_path.write_text(header + "\n".join(rows) + footer)
    print(f"[SAVED] {md_path}")
    return md_path


# ═══════════════════════════════════════════════════════════════════════════
#  7. Serialise best pipeline (scaler + model)
# ═══════════════════════════════════════════════════════════════════════════
def save_best_pipeline(
    best_name: str,
    best_model: object,
    scaler: object,
) -> Path:
    """Save the complete prediction pipeline: scaler + trained model."""
    pipeline = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler,
    }
    path = MODELS_DIR / "best_appliance_model.joblib"
    joblib.dump(pipeline, path)
    print(f"[SAVED] Pipeline → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  8. End-to-end training pipeline
# ═══════════════════════════════════════════════════════════════════════════
def run_training(
    csv_path: str | Path | None = None,
    split_strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Full training pipeline: load → preprocess → train → evaluate → save.

    Parameters
    ----------
    csv_path : path to CSV (auto-downloads if None)
    split_strategy : 'random' or 'chronological'
    test_size : fraction for test set
    random_state : seed for reproducibility

    Returns
    -------
    dict with keys: 'results_df', 'best_name', 'best_model', 'predictions'
    """
    # ── 1. Load and clean ──
    df = load_dataset(csv_path)
    df = drop_noise_columns(df)
    X, y = split_features_target(df)

    # ── 2. Train / test split ──
    if split_strategy == "chronological":
        X_train, X_test, y_train, y_test = chronological_split(
            X, y, test_size=test_size
        )
    else:
        X_train, X_test, y_train, y_test = random_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # ── 3. Scale features (fit on train only) ──
    X_train_s, X_test_s, scaler = scale_features(
        X_train, X_test, save_scaler=False
    )

    # ── 4. Train & evaluate all models ──
    results_df, predictions, fitted_models = train_and_evaluate(
        X_train_s, X_test_s, y_train, y_test
    )

    # ── 5. Identify best model ──
    best_name, best_model = identify_best_model(results_df, fitted_models)

    # ── 6. Generate comparison plots ──
    print(f"\n📊 Generating comparison figures → {FIGURES_DIR}/\n")
    plot_mae_comparison(results_df)
    plot_rmse_comparison(results_df)

    best_metrics = predictions[best_name]
    plot_actual_vs_predicted(
        y_test,
        best_metrics["y_pred"],
        best_name,
        r2=best_metrics["R2"],
        rmse=best_metrics["RMSE"],
    )

    # ── 7. Save Markdown results ──
    save_results_markdown(results_df, best_name, split_strategy)

    # ── 8. Serialise best pipeline ──
    save_best_pipeline(best_name, best_model, scaler)

    return {
        "results_df": results_df,
        "best_name": best_name,
        "best_model": best_model,
        "predictions": predictions,
        "scaler": scaler,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train & compare regression models for appliance energy prediction."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to the raw CSV. If omitted, auto-downloads.",
    )
    parser.add_argument(
        "--split", choices=["random", "chronological"], default="random",
        help="Train-test split strategy (default: random).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    print("━" * 65)
    print(f"  PHASE 3 — MODEL TRAINING  │  split={args.split.upper()}")
    print("━" * 65)

    results = run_training(
        csv_path=args.csv,
        split_strategy=args.split,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("\n✅ Phase 3 complete.")
    print(f"   Best model : {results['best_name']}")
    print(f"   Pipeline   : models/best_appliance_model.joblib")
    print(f"   Figures    : figures/fig5–fig7 + model_comparison_results.md")
