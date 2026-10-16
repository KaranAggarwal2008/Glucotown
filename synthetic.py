"""Generate fully synthetic, temporally coherent EHR and wearable observations."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["age", "bmi", "hba1c", "diabetes_years", "medication", "glucose", "glucose_delta_30m", "heart_rate", "steps_30m", "sleep_hours", "carbs_30m", "hour_sin", "hour_cos"]
HORIZON_STEPS = 4  # 4 x 30 minutes = 2 hours


def generate(n_patients=90, days=6, seed=42):
    """Return (EHR, time-series). All records are synthetic, NOT clinical data."""
    rng = np.random.default_rng(seed)
    ehr_rows, series_rows = [], []
    start = pd.Timestamp("2026-01-01")
    for p in range(n_patients):
        pid = f"SYN-{p+1:04d}"
        age = int(rng.integers(25, 79))
        bmi = round(float(np.clip(rng.normal(27.5, 4.3), 18, 42)), 1)
        hba1c = round(float(np.clip(rng.normal(7.1, 1.1), 5.1, 11.2)), 1)
        duration = int(rng.integers(0, 18))
        medication = int(rng.random() < 0.56)
        sleep = float(np.clip(rng.normal(7.0, 0.9), 4.0, 9.5))
        ehr_rows.append(dict(patient_id=pid, age=age, bmi=bmi, hba1c=hba1c, diabetes_years=duration, medication=medication))
        n = days * 48
        glucose = float(94 + (hba1c - 5.5) * 24 + rng.normal(0, 12))
        previous = glucose
        carb_effect = 0.0
        for t in range(n):
            ts = start + pd.Timedelta(minutes=30*t)
            hour = ts.hour + ts.minute / 60
            meal = (7 <= hour < 9 or 12 <= hour < 14 or 19 <= hour < 21)
            carbs = float(rng.uniform(25, 85)) if meal and rng.random() < 0.27 else 0.0
            steps = int(rng.poisson(120 if 8 <= hour < 22 else 12))
            heart_rate = float(np.clip(rng.normal(74 + 0.012*steps, 7), 48, 130))
            carb_effect = 0.68*carb_effect + 0.42*carbs
            baseline = 92 + 25*(hba1c-5.5) + 0.3*(bmi-27) - 8*medication
            glucose = float(np.clip(glucose + 0.19*(baseline-glucose) + 0.43*carb_effect - 0.012*steps + rng.normal(0, 7), 65, 360))
            series_rows.append(dict(patient_id=pid, timestamp=ts, glucose=round(glucose, 2), glucose_delta_30m=round(glucose-previous, 2), heart_rate=round(heart_rate, 1), steps_30m=steps, sleep_hours=round(sleep, 2), carbs_30m=round(carbs, 1), hour_sin=float(np.sin(2*np.pi*hour/24)), hour_cos=float(np.cos(2*np.pi*hour/24))))
            previous = glucose
    return pd.DataFrame(ehr_rows), pd.DataFrame(series_rows)


def add_targets(ehr, series, threshold=180):
    """Target: glucose > threshold exactly two hours after observation."""
    frame = series.sort_values(["patient_id", "timestamp"]).copy()
    frame["glucose_2h"] = frame.groupby("patient_id")["glucose"].shift(-HORIZON_STEPS)
    frame = frame.dropna(subset=["glucose_2h"]).merge(ehr, on="patient_id", validate="many_to_one")
    frame["spike_2h"] = (frame["glucose_2h"] >= threshold).astype(int)
    return frame.reset_index(drop=True)


def save_data(out_dir="data", n_patients=90, days=6, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ehr, series = generate(n_patients, days, seed)
    ehr.to_csv(out / "synthetic_ehr.csv", index=False)
    series.to_csv(out / "synthetic_wearables.csv", index=False)
    return ehr, series
