"""
app.py — Smart Home Appliance Energy Predictor (Phase 4)
========================================================
Interactive Streamlit web interface that loads the serialised
HistGradientBoosting pipeline and serves real-time energy predictions.

Run:
    cd smart-home-energy-predictor
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
PIPELINE_PATH = MODELS_DIR / "best_appliance_model.joblib"

# ── Feature order (must match training exactly) ───────────────────────────
FEATURE_COLUMNS = [
    "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
    "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7",
    "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
    "RH_out", "Windspeed", "Visibility", "Tdewpoint",
]

# ── Feature metadata: labels, ranges, defaults ───────────────────────────
FEATURE_META = {
    "lights":      {"label": "Lights Energy (Wh)",          "min": 0.0,   "max": 70.0,   "default": 0.0,    "step": 1.0,   "help": "Energy use of lights in the house (Wh)"},
    "T1":          {"label": "Temperature (°C)",            "min": 14.0,  "max": 30.0,   "default": 21.5,   "step": 0.1,   "help": "Temperature in kitchen area"},
    "RH_1":        {"label": "Humidity (%)",                 "min": 20.0,  "max": 70.0,   "default": 40.0,   "step": 0.5,   "help": "Relative humidity in kitchen area"},
    "T2":          {"label": "Temperature (°C)",            "min": 14.0,  "max": 32.0,   "default": 20.0,   "step": 0.1,   "help": "Temperature in living room"},
    "RH_2":        {"label": "Humidity (%)",                 "min": 18.0,  "max": 60.0,   "default": 40.0,   "step": 0.5,   "help": "Relative humidity in living room"},
    "T3":          {"label": "Temperature (°C)",            "min": 14.0,  "max": 32.0,   "default": 22.0,   "step": 0.1,   "help": "Temperature in laundry room"},
    "RH_3":        {"label": "Humidity (%)",                 "min": 25.0,  "max": 55.0,   "default": 39.0,   "step": 0.5,   "help": "Relative humidity in laundry room"},
    "T4":          {"label": "Temperature (°C)",            "min": 13.0,  "max": 28.0,   "default": 20.5,   "step": 0.1,   "help": "Temperature in office room"},
    "RH_4":        {"label": "Humidity (%)",                 "min": 25.0,  "max": 55.0,   "default": 39.0,   "step": 0.5,   "help": "Relative humidity in office room"},
    "T5":          {"label": "Temperature (°C)",            "min": 13.0,  "max": 28.0,   "default": 19.5,   "step": 0.1,   "help": "Temperature in bathroom"},
    "RH_5":        {"label": "Humidity (%)",                 "min": 28.0,  "max": 100.0,  "default": 51.0,   "step": 0.5,   "help": "Relative humidity in bathroom"},
    "T6":          {"label": "Temperature (°C)",            "min": -10.0, "max": 30.0,   "default": 8.0,    "step": 0.1,   "help": "Temperature outside (north-facing sensor)"},
    "RH_6":        {"label": "Humidity (%)",                 "min": 1.0,   "max": 100.0,  "default": 55.0,   "step": 0.5,   "help": "Relative humidity outside (north sensor)"},
    "T7":          {"label": "Temperature (°C)",            "min": 13.0,  "max": 28.0,   "default": 20.0,   "step": 0.1,   "help": "Temperature in ironing room"},
    "RH_7":        {"label": "Humidity (%)",                 "min": 20.0,  "max": 55.0,   "default": 35.0,   "step": 0.5,   "help": "Relative humidity in ironing room"},
    "T8":          {"label": "Temperature (°C)",            "min": 14.0,  "max": 30.0,   "default": 22.0,   "step": 0.1,   "help": "Temperature in teenager room"},
    "RH_8":        {"label": "Humidity (%)",                 "min": 27.0,  "max": 62.0,   "default": 43.0,   "step": 0.5,   "help": "Relative humidity in teenager room"},
    "T9":          {"label": "Temperature (°C)",            "min": 13.0,  "max": 26.0,   "default": 19.5,   "step": 0.1,   "help": "Temperature in parents room"},
    "RH_9":        {"label": "Humidity (%)",                 "min": 27.0,  "max": 56.0,   "default": 41.5,   "step": 0.5,   "help": "Relative humidity in parents room"},
    "T_out":       {"label": "Outdoor Temperature (°C)",    "min": -8.0,  "max": 28.0,   "default": 7.5,    "step": 0.1,   "help": "Weather station outdoor temperature"},
    "Press_mm_hg": {"label": "Barometric Pressure (mmHg)",  "min": 725.0, "max": 775.0,  "default": 756.0,  "step": 0.5,   "help": "Atmospheric pressure from weather station"},
    "RH_out":      {"label": "Outdoor Humidity (%)",         "min": 20.0,  "max": 100.0,  "default": 80.0,   "step": 0.5,   "help": "Outdoor relative humidity"},
    "Windspeed":   {"label": "Wind Speed (m/s)",             "min": 0.0,   "max": 15.0,   "default": 4.0,    "step": 0.5,   "help": "Wind speed from weather station"},
    "Visibility":  {"label": "Visibility (km)",             "min": 1.0,   "max": 70.0,   "default": 38.0,   "step": 1.0,   "help": "Visibility distance from weather station"},
    "Tdewpoint":   {"label": "Dew Point (°C)",             "min": -10.0, "max": 18.0,   "default": 3.8,    "step": 0.1,   "help": "Dew point temperature from weather station"},
}

# Phase 3 results table
MODEL_RESULTS = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "HistGradientBoosting (Best)"],
    "MAE (Wh)": [52.55, 42.37, 35.63, 35.42],
    "RMSE (Wh)": [91.17, 84.35, 73.27, 70.47],
    "R²": [0.1694, 0.2890, 0.4636, 0.5037],
})

ROOM_NAMES = {
    "1": "Kitchen", "2": "Living Room", "3": "Laundry",
    "4": "Office", "5": "Bathroom", "6": "Outside (North)",
    "7": "Ironing Room", "8": "Teenager Room", "9": "Parents Room",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════
def make_slider(key: str) -> float:
    """Create a Streamlit slider from FEATURE_META and return the value."""
    m = FEATURE_META[key]
    return st.slider(
        label=m["label"],
        min_value=m["min"],
        max_value=m["max"],
        value=m["default"],
        step=m["step"],
        help=m["help"],
        key=f"slider_{key}",
    )


@st.cache_resource
def load_pipeline():
    """Load the serialised scaler + model pipeline."""
    import joblib
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  Page config & styling
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Home Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Clean, minimal student project styling */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .main-header {
        color: #212529;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #6c757d;
        font-size: 0.95rem;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        color: #212529;
        margin: 16px 0 20px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .prediction-label {
        color: #495057;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .prediction-value {
        color: #0d6efd;
        font-size: 3rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .prediction-unit {
        color: #6c757d;
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }
    section[data-testid="stSidebar"] h2 {
        color: #212529;
        font-size: 1.25rem;
    }
    .disclaimer {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 4px solid #6c757d;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.85rem;
        color: #495057;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 16px;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-header">Smart Home Energy Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Predict appliance energy consumption in Watt-hours using indoor climate '
    'sensors, outdoor weather data, and a trained HistGradientBoosting model.'
    '</p>',
    unsafe_allow_html=True,
)

# ── Load model ─────────────────────────────────────────────────────────────
pipeline = load_pipeline()

if pipeline is None:
    st.error(
        "⚠️ **Model not found!**\n\n"
        f"Expected pipeline at:\n`{PIPELINE_PATH}`\n\n"
        "Please run Phase 3 first:\n"
        "```bash\npython src/train_model.py\n```"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar — Input Controls
# ═══════════════════════════════════════════════════════════════════════════
input_values = {}

with st.sidebar:
    st.header("Input Parameters")
    st.caption("Adjust sensor readings and weather conditions below.")

    st.markdown("---")

    # ── Lighting ──
    st.subheader("Lighting")
    input_values["lights"] = make_slider("lights")

    st.markdown("---")

    # ── Indoor Climate (grouped by room) ──
    st.subheader("Indoor Climate")
    for room_num, room_name in ROOM_NAMES.items():
        with st.expander(room_name, expanded=(room_num == "1")):
            temp_key = f"T{room_num}"
            rh_key = f"RH_{room_num}"
            if temp_key in FEATURE_META:
                input_values[temp_key] = make_slider(temp_key)
            if rh_key in FEATURE_META:
                input_values[rh_key] = make_slider(rh_key)

    st.markdown("---")

    # ── Outdoor Weather ──
    st.subheader("Outdoor Weather")
    for key in ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]:
        input_values[key] = make_slider(key)


# ═══════════════════════════════════════════════════════════════════════════
#  Build input DataFrame (exact column order)
# ═══════════════════════════════════════════════════════════════════════════
input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]


# ═══════════════════════════════════════════════════════════════════════════
#  Run prediction
# ═══════════════════════════════════════════════════════════════════════════
scaler = pipeline["scaler"]
model = pipeline["model"]

input_scaled = pd.DataFrame(
    scaler.transform(input_df),
    columns=FEATURE_COLUMNS,
)
prediction = max(model.predict(input_scaled)[0], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Content — Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_predict, tab_insights = st.tabs(["Prediction", "Model Insights"])

# ────────────────────────────────────────────────────────────────────────
#  Tab 1: Prediction
# ────────────────────────────────────────────────────────────────────────
with tab_predict:

    # ── Prediction metric card ──
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-label">Predicted Appliance Energy Consumption</div>
        <div class="prediction-value">{prediction:.1f}</div>
        <div class="prediction-unit">Watt-hours (Wh)</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Context metrics ──
    indoor_temps = [input_values.get(f"T{i}", 20) for i in range(1, 10)]
    indoor_rhs = [input_values.get(f"RH_{i}", 40) for i in range(1, 10)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Indoor Temp", f"{np.mean(indoor_temps):.1f} °C")
    with col2:
        st.metric("Avg Indoor Humidity", f"{np.mean(indoor_rhs):.1f} %")
    with col3:
        st.metric("Outdoor Temp", f"{input_values.get('T_out', 7.5):.1f} °C")

    # ── Energy interpretation ──
    st.markdown("---")
    if prediction < 50:
        st.success("Low energy usage — Appliances are in standby or minimal use.")
    elif prediction < 100:
        st.info("Normal energy usage — Typical for moderate appliance activity.")
    elif prediction < 200:
        st.warning("Elevated energy usage — Multiple appliances may be active.")
    else:
        st.error("High energy usage — Peak consumption detected.")

    # ── Input summary ──
    with st.expander("View Input Feature Values"):
        display_df = input_df.T.copy()
        display_df.columns = ["Value"]
        st.dataframe(
            display_df.style.format("{:.2f}"),
            use_container_width=True,
        )

    # ── Disclaimer ──
    st.markdown(
        '<div class="disclaimer">'
        '<strong>Disclaimer:</strong> Predictions are estimates based on patterns '
        'learned from a single low-energy house in Stambruges, Belgium (4.5-month dataset, '
        'Jan–May 2016). Actual consumption may vary significantly for different homes, '
        'climates, and appliance configurations.'
        '</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────
#  Tab 2: Model Insights
# ────────────────────────────────────────────────────────────────────────
with tab_insights:

    st.subheader("Model Comparison Results")
    st.caption("Phase 3 evaluation on the 20% test split (3,947 samples)")

    # ── Results table ──
    st.dataframe(
        MODEL_RESULTS.style
        .highlight_min(subset=["MAE (Wh)", "RMSE (Wh)"], color="#e9ecef")
        .highlight_max(subset=["R²"], color="#dbeafe")
        .format({"MAE (Wh)": "{:.2f}", "RMSE (Wh)": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Comparison figures ──
    col_a, col_b = st.columns(2)

    fig5_path = FIGURES_DIR / "fig5_mae_comparison.png"
    fig6_path = FIGURES_DIR / "fig6_rmse_comparison.png"
    fig7_path = FIGURES_DIR / "fig7_actual_vs_predicted.png"

    with col_a:
        if fig5_path.exists():
            st.image(str(fig5_path), caption="MAE Comparison", use_container_width=True)
        else:
            st.info("fig5_mae_comparison.png not found — run Phase 3 first.")

    with col_b:
        if fig6_path.exists():
            st.image(str(fig6_path), caption="RMSE Comparison", use_container_width=True)
        else:
            st.info("fig6_rmse_comparison.png not found — run Phase 3 first.")

    if fig7_path.exists():
        st.image(str(fig7_path), caption="Actual vs Predicted — HistGradientBoosting", use_container_width=True)
    else:
        st.info("fig7_actual_vs_predicted.png not found — run Phase 3 first.")

    # ── Best model details ──
    st.markdown("---")
    st.subheader("Best Model Details (HistGradientBoosting)")

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Algorithm** | HistGradientBoosting |
        | **MAE** | 35.42 Wh |
        | **RMSE** | 70.47 Wh |
        | **R²** | 0.5037 |
        | **Features** | {len(FEATURE_COLUMNS)} |
        """)
    with detail_col2:
        st.markdown("""
        | Hyperparameter | Value |
        |----------------|-------|
        | **Max Iterations** | 300 |
        | **Max Depth** | 8 |
        | **Learning Rate** | 0.1 |
        | **Min Samples Leaf** | 20 |
        | **Random State** | 42 |
        """)


# ── Footer ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Smart Home Appliance Energy Predictor · '
    'UCI Appliances Energy Prediction Dataset · '
    'Built with Streamlit & scikit-learn'
    '</div>',
    unsafe_allow_html=True,
)
