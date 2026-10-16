"""GlucoTwin: simulated clinical workflow dashboard (NOT a medical device)."""
from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import predict
from synthetic import FEATURES

st.set_page_config(page_title="GlucoTwin | Synthetic Digital Twin", page_icon="🩺", layout="wide")
st.title("GlucoTwin · Glucose Digital Twin")
st.warning("RESEARCH DEMONSTRATOR ONLY — all patients and wearable readings are synthetic. Predictions are not clinically validated and must not guide medical decisions.")

@st.cache_resource
def load_bundle():
    return joblib.load("glucotwin_models.joblib")

@st.cache_data
def load_data():
    ehr = pd.read_csv("synthetic_ehr.csv")
    series = pd.read_csv("synthetic_wearables.csv", parse_dates=["timestamp"])
    return ehr, series

if not all(Path(x).exists() for x in ["glucotwin_models.joblib", "synthetic_ehr.csv", "synthetic_wearables.csv"]):
    st.error("First run `python train.py` from the repository root to generate synthetic data and train the models.")
    st.stop()

bundle = load_bundle()
ehr, series = load_data()
with st.sidebar:
    st.header("Synthetic patient selector")
    pid = st.selectbox("Patient ID", ehr["patient_id"].tolist())
    st.caption("All patient IDs are fictional.")
    st.header("Scenario controls")
    extra_carbs = st.slider("Additional carbs in last 30 min (g)", 0, 80, 0, 5)
    extra_steps = st.slider("Additional steps in last 30 min", 0, 1500, 0, 50)
    st.caption("What-if estimates alter input features only; they do not simulate a validated physiological response.")

patient = ehr[ehr.patient_id == pid].iloc[0]
history = series[series.patient_id == pid].sort_values("timestamp").copy()
current = history.iloc[-1]
row = pd.DataFrame([{**patient.to_dict(), **current.to_dict()}])
row["carbs_30m"] = row["carbs_30m"] + extra_carbs
row["steps_30m"] = row["steps_30m"] + extra_steps
prob, predicted_glucose = predict(bundle, row)

st.subheader("Patient-specific snapshot")
a,b,c,d = st.columns(4)
a.metric("Current glucose", f"{current.glucose:.0f} mg/dL")
b.metric("Predicted glucose · +2 h", f"{predicted_glucose:.0f} mg/dL")
c.metric("Predicted ≥180 mg/dL risk", f"{prob*100:.1f}%")
d.metric("Synthetic HbA1c", f"{patient.hba1c:.1f}%")

left,right = st.columns([2,1])
with left:
    st.subheader("Wearable glucose trajectory")
    recent = history.tail(48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.timestamp, y=recent.glucose, mode="lines", name="Synthetic CGM", line=dict(width=3)))
    future = pd.to_datetime(current.timestamp) + pd.Timedelta(hours=2)
    fig.add_trace(go.Scatter(x=[current.timestamp, future], y=[current.glucose, predicted_glucose], mode="lines+markers", name="Model forecast (+2 h)", line=dict(dash="dash", width=3)))
    fig.add_hline(y=180, line_dash="dot", annotation_text="Demo spike threshold 180 mg/dL")
    fig.update_layout(xaxis_title="Time", yaxis_title="Glucose (mg/dL)", height=410, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Synthetic EHR")
    st.dataframe(pd.DataFrame({"Field": ["Age", "BMI", "HbA1c", "Years with diabetes", "Medication flag"], "Value": [patient.age, patient.bmi, patient.hba1c, patient.diabetes_years, patient.medication]}), hide_index=True, use_container_width=True)
    st.subheader("Risk signal")
    st.progress(min(max(prob, 0), 1))
    st.caption("Risk probability is a model output on simulated data, not a calibrated clinical risk estimate.")

st.subheader("Model evaluation · held-out synthetic patients")
metrics = json.loads(Path("metrics.json").read_text())
x,y,z = st.columns(3)
x.metric("ROC AUC (synthetic test)", f"{metrics['test_roc_auc']:.3f}")
y.metric("Average precision", f"{metrics['test_average_precision']:.3f}")
z.metric("Glucose MAE", f"{metrics['test_glucose_mae_mg_dl']:.1f} mg/dL")
st.caption(f"Patient-disjoint test: {metrics['n_test_patients']} patients, {metrics['n_test_rows']:,} observations. Synthetic prevalence {metrics['test_spike_prevalence']:.1%}.")
st.info("**Limitations:** No real patient data; no clinical ground truth; synthetic simulator and model may share artificial patterns. A real-world digital twin would require consented clinical data, external validation, sensor integration, calibration, security reviews and regulatory assessment.")
