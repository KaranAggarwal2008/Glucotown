# GlucoTwin — Synthetic Patient Digital Twin for Two-Hour Glucose Spike Forecasting

> **Research-only proof of concept.** All patient records and wearable streams are artificially generated. No clinical data, medical validation, or medical-device claims. **Never use predictions for patient care.**

## Challenge submission
- **Event:** Happiest Health Digital Twin Challenge 2026
- **Team name:** GlucoTwin (edit as needed)
- **Team leader:** Karan Aggarwal
- **College / incubator:** **[ENTER CURRENT FULL-TIME INSTITUTION AND VERIFY ELIGIBILITY]**
- **Team size:** 1 (solo)
- **Project title:** GlucoTwin: A Synthetic Patient Digital Twin for Predicting Two-Hour Hyperglycaemia Risk

## Problem statement and healthcare use case
People living with type 2 diabetes may experience high glucose after meals. This proof of concept explores whether a personalised computational representation combining *historical health information* (age, BMI, HbA1c, diagnosis duration, medication flag) and *time-varying wearable information* (glucose, activity, heart rate, sleep, meals) can forecast glucose and flag a potential high-glucose event two hours ahead. It is a **technical demonstration**, not a clinical prediction tool.

## What actually works
1. Generates fully synthetic EHR profiles and 30-minute wearable time series for 90 fictional patients.
2. Fuses static and dynamic features for each observation.
3. Trains a classifier for `glucose >= 180 mg/dL` exactly 2 hours in the future and a regressor for future glucose.
4. Splits training and test observations **by patient**, preventing the same synthetic patient from appearing in both sets.
5. Reports ROC AUC, average precision, Brier score, and glucose MAE on held-out synthetic patients.
6. Runs a Streamlit clinician-style dashboard with historical chart, risk forecast, EHR panel and **illustrative** scenario controls.

## Technical stack
Python 3.11+, NumPy, pandas, scikit-learn HistGradientBoostingClassifier / Regressor, Streamlit, Plotly, joblib, pytest.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Run tests
```bash
pytest -q
```

## Architecture
```text
SYNTHETIC EHR                  SYNTHETIC WEARABLE STREAM
age, BMI, HbA1c,               CGM, steps, heart rate,
diagnosis, medication           sleep, meals, timestamps
        |                                 |
        +-------------+-------------------+
                      |
              patient-ID feature fusion
                      |
          2-hour forward label generation
                      |
         PATIENT-DISJOINT TRAIN / TEST SPLIT
                      |
             +--------+--------+
             |                 |
        Classifier          Regressor
       spike >=180?        glucose +2h
             |                 |
             +--------+--------+
                      |
            Streamlit doctor dashboard
        forecast, risk signal, history,
           synthetic EHR, scenario UI
```
See `GlucoTwin_Presentation.pptx (architecture on slide 4)` for detailed design.

## Data provenance and privacy
All records are generated locally by deterministic rules in `synthetic.py` and are **entirely synthetic**. There are no real patient identifiers, external clinical records, or proprietary data. CSVs can be regenerated with `python train.py`. Generated data is **not representative of the Indian population** and does not establish prevalence or accuracy for India.

## Evaluation methodology
- A forward-looking target is created by shifting glucose **four 30-minute steps** within each patient.
- Patient-disjoint 75/25 train/test partition via `GroupShuffleSplit`.
- Metrics are calculated on the held-out synthetic patients; see `metrics.json` after training.
- **Limitation:** both labels and predictors originate from one synthetic generator; apparent predictive performance may reflect generator rules rather than clinical validity. There is no prospective validation, independent dataset or calibrated clinical threshold.

## Required competition materials — action items
- [x] Working source code, tests, documentation, MIT license
- [x] Architecture description (also generate PDF/PPT diagram for submission)
- [ ] **At least 20-minute demo/explanation video**: add an unlisted YouTube URL or linked video file here: **[VIDEO LINK]**
- [x] **Presentation PDF/PPT**: add file in the repository root and link it here: `GlucoTwin_Presentation.pptx`
- [x] **Architecture PDF/PPT**: add file in the repository root and link it here: `GlucoTwin_Presentation.pptx` (slide 4)
- [ ] Replace college/institution placeholder; verify full-time student eligibility
- [ ] Push this repository to **public GitHub**, verify links in an incognito browser, then submit the URL on Unstop

## Suggested 20-minute video outline
0:00–2:00 problem and use case; 2:00–5:00 architecture; 5:00–8:00 synthetic data and privacy; 8:00–11:00 features and labels; 11:00–14:00 patient-disjoint training and metrics; 14:00–18:00 live dashboard and what-if demonstration; 18:00–20:00 limitations, clinical validation roadmap, conclusion.

## Responsible use and next steps
Future work would require consented, ethically approved patient data, independent external evaluation, sensor integrations, subgroup performance analysis, calibration, clinician co-design, security and privacy review, and relevant Indian regulatory assessment. This prototype does not prescribe interventions.

## License
MIT — see `LICENSE`.

## Flat repository layout
All files are in the repository root (no nested directories). Upload the extracted files directly to a public GitHub repository. `GlucoTwin_Presentation.pptx` contains the project deck and architecture diagram. Replace the institution placeholder and add the required 20-minute video URL before submission.
