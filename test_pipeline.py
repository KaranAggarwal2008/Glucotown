import numpy as np
from synthetic import generate, add_targets, FEATURES
from model import train


def test_target_has_correct_two_hour_horizon():
    ehr, series = generate(n_patients=3, days=2, seed=9)
    data = add_targets(ehr, series)
    first = data[data.patient_id == "SYN-0001"].iloc[0]
    future = series[series.patient_id == "SYN-0001"].iloc[4]
    assert first.glucose_2h == future.glucose
    assert len(data) == 3 * (2 * 48 - 4)


def test_training_and_artifacts(tmp_path):
    metrics = train(tmp_path / "models", tmp_path / "data", n_patients=35, days=4, seed=42)
    assert (tmp_path / "models" / "glucotwin_models.joblib").exists()
    assert 0 <= metrics["test_roc_auc"] <= 1
    assert metrics["n_test_patients"] > 0
    assert metrics["n_train_patients"] > 0


def test_no_missing_features():
    ehr, series = generate(n_patients=2, days=1, seed=7)
    data = add_targets(ehr, series)
    assert not data[FEATURES].isna().any().any()
