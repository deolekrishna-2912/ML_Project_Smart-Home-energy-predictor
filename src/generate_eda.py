"""
generate_eda.py — Smart Home Appliance Energy Predictor (Phase 2)
=================================================================
Exploratory Data Analysis script that generates publication-quality
visualisations and prints comprehensive summary statistics.

Figures saved to  figures/:
    fig1_target_distribution.png   – Histogram + KDE + boxplot of Appliances
    fig2_correlation_heatmap.png   – Annotated correlation matrix
    fig3_time_series_trend.png     – Hourly energy consumption patterns
    fig4_feature_relationships.png – Scatter / regplots of top-correlated features

Usage:
    python src/generate_eda.py                       # uses raw CSV from data/
    python src/generate_eda.py --csv path/to/file.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Non-interactive backend — MUST be set before any other matplotlib import
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Add src/ to path so we can reuse Phase-1 helpers
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocess import load_dataset, drop_noise_columns, DATE_COL, TARGET_COL

# ── Plotting defaults ─────────────────────────────────────────────────────
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

# Colour palette – a curated palette that looks good on light & dark
PALETTE = {
    "primary": "#4361EE",
    "secondary": "#7209B7",
    "accent": "#F72585",
    "highlight": "#4CC9F0",
    "warm": "#FF6B35",
    "bg": "#F8F9FA",
}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: save figure
# ═══════════════════════════════════════════════════════════════════════════
def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 1 — Target Distribution (Histogram + KDE + Boxplot)
# ═══════════════════════════════════════════════════════════════════════════
def fig1_target_distribution(df: pd.DataFrame) -> None:
    """Histogram with KDE overlay and an inset boxplot of the Appliances column."""
    target = df[TARGET_COL]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)

    # ── Top: Histogram + KDE ──
    ax_hist = fig.add_subplot(gs[0])
    sns.histplot(
        target,
        bins=60,
        kde=True,
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.75,
        ax=ax_hist,
        stat="density",
    )
    ax_hist.axvline(
        target.mean(), color=PALETTE["accent"], ls="--", lw=1.8,
        label=f"Mean = {target.mean():.1f} Wh",
    )
    ax_hist.axvline(
        target.median(), color=PALETTE["warm"], ls="-.", lw=1.8,
        label=f"Median = {target.median():.1f} Wh",
    )
    ax_hist.set_title(
        "Distribution of Appliance Energy Consumption",
        fontsize=14, pad=10,
    )
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=10, framealpha=0.9)
    ax_hist.set_xlim(left=0)

    # Add skewness / kurtosis annotation
    sk = target.skew()
    ku = target.kurtosis()
    ax_hist.text(
        0.97, 0.92,
        f"Skewness = {sk:.2f}\nKurtosis  = {ku:.2f}",
        transform=ax_hist.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
    )

    # ── Bottom: Boxplot ──
    ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
    sns.boxplot(
        x=target, ax=ax_box,
        color=PALETTE["highlight"],
        flierprops=dict(marker="o", markerfacecolor=PALETTE["accent"],
                        markersize=3, alpha=0.5),
        width=0.5,
    )
    ax_box.set_xlabel("Appliance Energy Use (Wh)")
    ax_box.set_ylabel("")
    ax_box.set_yticks([])

    plt.setp(ax_hist.get_xticklabels(), visible=False)

    _save(fig, "fig1_target_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 2 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig2_correlation_heatmap(df: pd.DataFrame) -> None:
    """Annotated correlation matrix for key variables + Appliances."""
    # Select key columns: indoor temps/humidities, outdoor weather, target
    indoor_temp_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    indoor_rh_cols = [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]
    outdoor_cols = ["T_out", "RH_out", "Press_mm_hg", "Windspeed",
                    "Visibility", "Tdewpoint"]
    other_cols = ["lights"]

    selected = (
        [TARGET_COL]
        + other_cols
        + indoor_temp_cols
        + indoor_rh_cols
        + [c for c in outdoor_cols if c in df.columns]
    )

    corr = df[selected].corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(16, 13))
    cmap = sns.diverging_palette(250, 15, s=90, l=40, n=20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(
        "Correlation Matrix — Indoor Climate, Outdoor Weather & Energy Consumption",
        fontsize=14, pad=14,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    _save(fig, "fig2_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 3 — Time-Series Trend (Hourly patterns)
# ═══════════════════════════════════════════════════════════════════════════
def fig3_time_series_trend(df: pd.DataFrame) -> None:
    """Line plot of mean energy consumption by hour of day, with a
    daily rolling-mean overlay for the full time series."""
    if DATE_COL not in df.columns:
        print("[WARN] 'date' column missing — skipping fig3.")
        return

    date_series = pd.to_datetime(df[DATE_COL])

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 2]})

    # ── Top: Full time-series with daily rolling average ──
    ax1 = axes[0]
    ts = df[[TARGET_COL]].copy()
    ts.index = date_series
    ts_sorted = ts.sort_index()

    # Raw 10-min readings (subsampled for visual clarity)
    ax1.plot(
        ts_sorted.index, ts_sorted[TARGET_COL],
        color=PALETTE["primary"], alpha=0.15, linewidth=0.4,
        label="10-min readings",
    )
    # Daily rolling mean
    rolling = ts_sorted[TARGET_COL].rolling(window=144, min_periods=72).mean()
    ax1.plot(
        rolling.index, rolling,
        color=PALETTE["accent"], linewidth=2,
        label="24h rolling mean",
    )
    ax1.set_title("Appliance Energy Consumption Over Time", fontsize=14, pad=10)
    ax1.set_ylabel("Energy (Wh)")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.set_xlim(ts_sorted.index.min(), ts_sorted.index.max())

    # ── Bottom: Mean energy by hour-of-day ──
    ax2 = axes[1]
    hourly = df.copy()
    hourly["hour"] = date_series.dt.hour
    hourly_stats = hourly.groupby("hour")[TARGET_COL].agg(["mean", "std"])

    ax2.fill_between(
        hourly_stats.index,
        hourly_stats["mean"] - hourly_stats["std"],
        hourly_stats["mean"] + hourly_stats["std"],
        alpha=0.2, color=PALETTE["secondary"],
        label="±1 std",
    )
    ax2.plot(
        hourly_stats.index, hourly_stats["mean"],
        marker="o", markersize=6, linewidth=2.5,
        color=PALETTE["secondary"],
        label="Mean",
    )

    # Highlight peak hours
    peak_hour = hourly_stats["mean"].idxmax()
    ax2.annotate(
        f"Peak: {peak_hour}:00\n({hourly_stats['mean'].max():.0f} Wh)",
        xy=(peak_hour, hourly_stats["mean"].max()),
        xytext=(peak_hour + 2, hourly_stats["mean"].max() + 20),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["accent"]),
    )

    ax2.set_title("Average Energy Consumption by Hour of Day", fontsize=13, pad=10)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Energy (Wh)")
    ax2.set_xticks(range(0, 24))
    ax2.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout(h_pad=3)
    _save(fig, "fig3_time_series_trend.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 4 — Feature Relationships (Top-Correlated Scatter/Regplots)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_feature_relationships(df: pd.DataFrame) -> None:
    """Scatter + regression plots for the top-6 features most correlated
    (absolute value) with the target."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
    )
    top_features = corr_with_target.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = [
        PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
        PALETTE["highlight"], PALETTE["warm"], "#2EC4B6",
    ]

    for i, (feat, color) in enumerate(zip(top_features, colors)):
        ax = axes[i]
        r_val = numeric_df[[feat, TARGET_COL]].corr().iloc[0, 1]

        sns.regplot(
            x=feat, y=TARGET_COL, data=df,
            ax=ax,
            scatter_kws=dict(alpha=0.15, s=8, color=color),
            line_kws=dict(color=PALETTE["accent"], linewidth=2),
            ci=95,
        )
        ax.set_title(f"{feat}  (r = {r_val:+.3f})", fontsize=12)
        ax.set_xlabel(feat)
        ax.set_ylabel("Appliances (Wh)" if i % 3 == 0 else "")

    fig.suptitle(
        "Top-6 Features Correlated with Appliance Energy Consumption",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig4_feature_relationships.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════
def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print missing-value report, distribution metrics, and correlation
    rankings to the console."""
    print("\n" + "═" * 70)
    print("  EXPLORATORY DATA ANALYSIS — SUMMARY STATISTICS")
    print("═" * 70)

    # ── Missing values ──
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n{'─' * 40}")
    print("MISSING VALUE REPORT")
    print(f"{'─' * 40}")
    if total_missing == 0:
        print("  ✓ No missing values in any column.")
    else:
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            print(f"  {col:>20s}: {missing[col]:>6d} ({pct:.2f}%)")
    print(f"  Total missing: {total_missing}")

    # ── Distribution metrics ──
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"\n{'─' * 40}")
    print("DISTRIBUTION METRICS (Numeric Columns)")
    print(f"{'─' * 40}")

    dist_df = pd.DataFrame({
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "skewness": numeric_df.skew(),
        "kurtosis": numeric_df.kurtosis(),
        "min": numeric_df.min(),
        "max": numeric_df.max(),
    })
    print(dist_df.to_string(float_format="{:.3f}".format))

    # ── Target-specific stats ──
    target = df[TARGET_COL]
    print(f"\n{'─' * 40}")
    print(f"TARGET VARIABLE: '{TARGET_COL}'")
    print(f"{'─' * 40}")
    print(f"  Count    : {len(target)}")
    print(f"  Mean     : {target.mean():.2f} Wh")
    print(f"  Median   : {target.median():.2f} Wh")
    print(f"  Std Dev  : {target.std():.2f} Wh")
    print(f"  Skewness : {target.skew():.3f}  {'(right-skewed)' if target.skew() > 0 else '(left-skewed)'}")
    print(f"  Kurtosis : {target.kurtosis():.3f}  {'(leptokurtic / heavy-tailed)' if target.kurtosis() > 0 else '(platykurtic)'}")
    print(f"  Min      : {target.min()} Wh")
    print(f"  Max      : {target.max()} Wh")
    print(f"  IQR      : {target.quantile(0.75) - target.quantile(0.25):.1f} Wh")

    # ── Feature correlation rankings ──
    corr_with_target = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )
    print(f"\n{'─' * 40}")
    print(f"FEATURE CORRELATION RANKINGS (vs {TARGET_COL})")
    print(f"{'─' * 40}")
    for rank, (feat, r) in enumerate(corr_with_target.items(), 1):
        bar = "█" * int(abs(r) * 40)
        sign = "+" if r > 0 else "−"
        print(f"  {rank:>2d}. {feat:>15s}  {sign}{abs(r):.4f}  {bar}")

    print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ═══════════════════════════════════════════════════════════════════════════
def run_eda(csv_path: str | Path | None = None) -> None:
    """Load dataset, generate all EDA figures, and print statistics."""
    # Load and clean
    df = load_dataset(csv_path)
    df = drop_noise_columns(df)

    print(f"\n📊 Generating EDA figures → {FIGURES_DIR}/\n")

    # Generate figures
    fig1_target_distribution(df)
    fig2_correlation_heatmap(df)
    fig3_time_series_trend(df)
    fig4_feature_relationships(df)

    # Print statistics
    print_summary_statistics(df)

    print(f"✅ EDA complete — 4 figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate EDA visualisations for the Appliances Energy dataset."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to the raw CSV. If omitted, auto-downloads.",
    )
    args = parser.parse_args()

    run_eda(csv_path=args.csv)
