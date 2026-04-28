import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Vehicle Predictive Maintenance",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Predictive Maintenance Risk Scorer")
st.markdown("Real-time machine failure prediction powered by XGBoost + SHAP explainability")
st.markdown("---")


# ── SHAP helper ───────────────────────────────────────────────────────────────
def fix_shap(sv, ev):
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    if isinstance(ev, (list, np.ndarray)):
        arr = list(np.array(ev).flat)
        ev = float(arr[1]) if len(arr) > 1 else float(arr[0])
    else:
        ev = float(ev)
    return sv, ev


# ── Load & prepare ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    # Clean feature names — no brackets allowed by XGBoost
    features = ['Type_encoded', 'Air_temp_K', 'Process_temp_K',
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']

    # Display labels for UI
    labels = {
        'Type_encoded':        'Machine Type',
        'Air_temp_K':          'Air Temperature (K)',
        'Process_temp_K':      'Process Temperature (K)',
        'Rotational_speed_rpm':'Rotational Speed (rpm)',
        'Torque_Nm':           'Torque (Nm)',
        'Tool_wear_min':       'Tool Wear (min)'
    }

    if os.path.exists('data/ai4i2020.csv'):
        df = pd.read_csv('data/ai4i2020.csv')
        le = LabelEncoder()
        df['Type_encoded'] = le.fit_transform(df['Type'])
        df = df.rename(columns={
            'Air temperature [K]':     'Air_temp_K',
            'Process temperature [K]': 'Process_temp_K',
            'Rotational speed [rpm]':  'Rotational_speed_rpm',
            'Torque [Nm]':             'Torque_Nm',
            'Tool wear [min]':         'Tool_wear_min'
        })
    else:
        # Synthetic data for Streamlit Cloud
        rng = np.random.default_rng(42)
        n = 5000
        n_fail = int(n * 0.034)
        n_ok   = n - n_fail

        def make(size, fail):
            return pd.DataFrame({
                'Type_encoded':         rng.choice([0, 1, 2], size),
                'Air_temp_K':           rng.normal(300 if fail else 298, 2, size),
                'Process_temp_K':       rng.normal(312 if fail else 310, 2, size),
                'Rotational_speed_rpm': rng.normal(1400 if fail else 1500, 100, size),
                'Torque_Nm':            rng.normal(50 if fail else 40, 8, size),
                'Tool_wear_min':        rng.normal(180 if fail else 100, 40, size),
                'Machine failure':      int(fail)
            })

        df = pd.concat(
            [make(n_ok, False), make(n_fail, True)],
            ignore_index=True
        ).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[features]
    y = df['Machine failure']
    return X, y, features, labels


# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    X, y, features, labels = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_r, y_train_r = sm.fit_resample(X_train_s, y_train)

    model = XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric='logloss', verbosity=0,
        feature_names=None  # prevent XGBoost from storing bracket names
    )
    model.fit(X_train_r, y_train_r)

    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    f1    = f1_score(y_test, preds, zero_division=0)
    cm    = confusion_matrix(y_test, preds)

    return model, scaler, features, labels, auc, f1, cm, X_test_s, y_test


with st.spinner("Loading model — please wait ~30 seconds..."):
    model, scaler, features, labels, auc, f1, cm, X_test_s, y_test = train_model()

tn, fp, fn, tp = cm.ravel()

# Header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",           "XGBoost + SMOTE")
c2.metric("AUC-ROC",         f"{auc:.3f}")
c3.metric("F1 Score",        f"{f1:.3f}")
c4.metric("Failures Caught", f"{tp} / {tp + fn}")

st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Predict Failure Risk", "🧠 SHAP Explainability"])


# =============================================================================
# TAB 1 — LIVE PREDICTOR
# =============================================================================
with tab1:
    st.header("Enter Vehicle Sensor Readings")
    st.markdown("Adjust the sliders to match your machine's sensor data and click **Score Failure Risk**.")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["L — Light duty", "M — Medium duty", "H — Heavy duty"]
        )
        type_enc = {
            "L — Light duty":  0,
            "M — Medium duty": 1,
            "H — Heavy duty":  2
        }[machine_type]

        air_temp  = st.slider("Air Temperature (K)",      295.0, 305.0, 298.0, 0.1)
        proc_temp = st.slider("Process Temperature (K)",  305.0, 315.0, 310.0, 0.1)

    with col2:
        rpm       = st.slider("Rotational Speed (rpm)",   1168, 2886, 1500)
        torque    = st.slider("Torque (Nm)",               3.8,  76.6,  40.0, 0.1)
        tool_wear = st.slider("Tool Wear (min)",             0,   253,   100)

    if st.button("Score Failure Risk", type="primary"):
        input_df = pd.DataFrame([{
            'Type_encoded':         type_enc,
            'Air_temp_K':           air_temp,
            'Process_temp_K':       proc_temp,
            'Rotational_speed_rpm': rpm,
            'Torque_Nm':            torque,
            'Tool_wear_min':        tool_wear
        }])[features]

        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Failure Probability", f"{prob * 100:.1f}%")
        r2.metric("Risk Level",
                  "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK")

        if prob > 0.7:
            st.error("⚠️ CRITICAL — Do not operate. Schedule immediate maintenance.")
        elif prob > 0.5:
            st.warning("🟡 Schedule maintenance within 24 hours.")
        elif prob > 0.3:
            st.warning("🟡 Elevated risk — monitor this machine closely.")
        else:
            st.success("✅ Machine operating within normal parameters.")

        # Risk gauge bar
        fig_g, ax_g = plt.subplots(figsize=(8, 1))
        ax_g.barh([0], [prob],
                  color='#D85A30' if prob > 0.5 else '#1D9E75',
                  height=0.4)
        ax_g.barh([0], [1 - prob], left=[prob],
                  color='#EEEEEE', height=0.4)
        ax_g.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6)
        ax_g.set_xlim(0, 1)
        ax_g.set_yticks([])
        ax_g.set_xlabel('Failure Probability')
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()

        # Per-prediction SHAP
        st.markdown("---")
        st.subheader("Why this score? — SHAP explanation")
        ex = shap.TreeExplainer(model)
        input_scaled_df = pd.DataFrame(input_scaled, columns=features)
        raw_sv = ex.shap_values(input_scaled_df)
        sv, ev = fix_shap(raw_sv, ex.expected_value)

        display_names = [labels[f] for f in features]
        fig_p, _ = plt.subplots(figsize=(8, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=ev,
                feature_names=display_names
            ), show=False
        )
        st.pyplot(fig_p)
        plt.close()


# =============================================================================
# TAB 2 — SHAP GLOBAL
# =============================================================================
with tab2:
    st.header("SHAP Explainability — Global View")
    st.markdown(
        "Which sensor readings matter most for predicting machine failure "
        "across the full dataset?"
    )

    X_test_df = pd.DataFrame(X_test_s, columns=features)
    sample    = X_test_df.iloc[:100]
    explainer = shap.TreeExplainer(model)
    raw_sv2   = explainer.shap_values(sample)
    shap_vals, base_val = fix_shap(raw_sv2, explainer.expected_value)

    display_names = [labels[f] for f in features]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature importance")
        shap.summary_plot(shap_vals, sample, plot_type="bar",
                          feature_names=display_names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
        st.caption("Higher bar = sensor drives failure prediction more globally.")

    with col2:
        st.subheader("Impact distribution")
        shap.summary_plot(shap_vals, sample,
                          feature_names=display_names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
        st.caption("Red = high sensor value. Right of center = increases failure risk.")

    st.markdown("---")
    st.subheader("Explain a specific vehicle")
    idx = st.slider("Select vehicle index", 0, min(99, len(shap_vals) - 1), 0)
    fig_w, _ = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[idx],
            base_values=base_val,
            feature_names=display_names
        ), show=False
    )
    st.pyplot(fig_w)
    plt.close()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Gayathri Kammidi** · MS Computer Science, Governors State University · May 2026  \n"
    "github.com/gkammidi-prog · linkedin.com/in/gayathri-kambidi-0714j/"
)