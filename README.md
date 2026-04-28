# 🚗 Vehicle Predictive Maintenance Risk Scorer

> XGBoost + SMOTE + SHAP — real-time machine failure prediction — deployed live

**[🚀 Live Demo](https://automotive-maintenance-predictor-sis7xa3bqtdwc6fs8twhqz.streamlit.app/)** &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/gayathri-kambidi-0714j/) &nbsp;·&nbsp; [GitHub](https://github.com/gkammidi-prog)

---

## The Problem

Unplanned machine failures cost manufacturers billions annually in downtime, repairs, and safety incidents. Predicting failure before it happens — using real sensor data — enables maintenance teams to intervene at the right time, not too early (wasting resources) and not too late (causing breakdowns). This system scores failure risk in real time from six sensor readings and explains exactly which sensor is driving the risk.

---

## Results

| Model | AUC-ROC | F1 Score | Failures Caught | Missed | False Alarms |
|-------|---------|----------|-----------------|--------|--------------|
| **XGBoost** | **0.972** | **0.704** | **56 / 68** | 12 | **35** |
| Gradient Boosting | 0.971 | 0.468 | 62 / 68 | 6 | 135 |
| Random Forest | 0.970 | 0.651 | 54 / 68 | 14 | 44 |
| Logistic Regression | 0.907 | 0.244 | 57 / 68 | 11 | 343 |
| KNN | 0.897 | 0.433 | 52 / 68 | 16 | 120 |
| Decision Tree | 0.828 | 0.516 | 47 / 68 | 21 | 67 |

> **XGBoost selected** — highest AUC (0.972), best F1 (0.704), and only 35 false alarms vs Gradient Boosting's 135. In fleet management, unnecessary maintenance trips cost money. XGBoost delivers the best balance between catching failures and avoiding false alarms.

---

## What This System Does

- Predicts **machine failure risk** from 6 real sensor readings in real time
- Trained on **10,000 industrial machine records** (AI4I 2020 dataset, 3.39% failure rate)
- Handles **extreme class imbalance** using SMOTE — 96.61% healthy vs 3.39% failure
- Explains every prediction using **SHAP waterfall charts** — shows which sensor is driving the risk
- Deployed as a **clean 2-tab Streamlit app** — prediction + global SHAP analysis

---

## Live App

```
Tab 1 — Predict Failure Risk
  Input: Machine type, air temp, process temp,
         rotational speed, torque, tool wear
  Output: Failure probability % + risk level +
          risk gauge bar + SHAP waterfall

Tab 2 — SHAP Explainability
  Global feature importance across full dataset
  SHAP distribution — which sensors matter most
  Per-vehicle waterfall explanation
```

👉 **[Try it live](https://automotive-maintenance-predictor-sis7xa3bqtdwc6fs8twhqz.streamlit.app/)**

---

## Engineering Decisions

**XGBoost over Gradient Boosting despite lower recall**
Gradient Boosting catches more failures (62 vs 56) but generates 135 false alarms vs XGBoost's 35. In a fleet of 10,000 vehicles, 100 unnecessary maintenance trips per cycle costs millions. XGBoost's F1 of 0.704 vs 0.468 reflects the right business tradeoff — catch most failures without flooding maintenance crews with false alarms.

**SMOTE over undersampling**
Machine failure is only 3.39% of the dataset. Undersampling discards 96%+ of healthy machine signal. SMOTE synthesises minority-class failure examples, giving the model enough failure cases to learn real sensor patterns without losing information.

**Recall over accuracy**
A naive model predicting "healthy" every time scores 96.61% accuracy and catches zero failures. The model is explicitly optimised to minimise missed failures — the metric that matters when a missed failure means a broken machine on the factory floor.

**SHAP for per-sensor attribution**
Built-in feature importance shows global averages. SHAP shows per-prediction attribution — exactly which sensor pushed this specific vehicle's risk score up or down. Maintenance teams can act on this directly: "Tool wear is driving 60% of this machine's risk — replace the tool."

---

## Failure Type Breakdown

| Failure Type | Count | % of Dataset |
|---|---|---|
| Heat Dissipation (HDF) | 115 | 1.15% — most common |
| Overstrain (OSF) | 98 | 0.98% |
| Power Failure (PWF) | 95 | 0.95% |
| Tool Wear (TWF) | 46 | 0.46% |
| Random Failure (RNF) | 19 | 0.19% — hardest to predict |

> Heat Dissipation is the top failure driver — temperature management is the highest-priority maintenance focus.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | XGBoost · scikit-learn · Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · KNN |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP — waterfall + summary plots |
| Data | Pandas · NumPy · StandardScaler |
| Deployment | Streamlit Cloud |

---

## Run Locally

```bash
git clone https://github.com/gkammidi-prog/automotive-maintenance-predictor
cd automotive-maintenance-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Place `ai4i2020.csv` from Kaggle into the `data/` folder for real data.
Without it, the app generates realistic synthetic sensor data automatically.

---

## Dataset

| Dataset | Size | Source |
|---------|------|--------|
| AI4I 2020 Predictive Maintenance | 10,000 records · 14 features | Kaggle (stephanmatzka) |

Features: machine type, air temperature, process temperature, rotational speed, torque, tool wear.

---

## Portfolio

| Project | Domain | Highlight |
|---------|--------|-----------|
| **Automotive Maintenance** *(this repo)* | Manufacturing | XGBoost AUC 0.972 · SHAP per-sensor |
| Mobile Sentiment Analyzer | NLP | 5-model NLP benchmark · Linear beats XGBoost |
| E-Commerce Cart Abandonment | Retail | 6-model benchmark · AUC 0.999 |
| Credit Risk & Fraud Detection | Banking | AUC 0.869 · Fraud Recall 75% |
| Medicare HCC Risk Score | Healthcare | Recall 90.5% · 71,518 encounters |

---

## Author

**Gayathri Kammidi**
MS Computer Science · Governors State University · May 2026
4+ years in Data Engineering & ML — GCP · BigQuery · Airflow · Python · XGBoost

[LinkedIn](https://www.linkedin.com/in/gayathri-kambidi-0714j/) · [GitHub](https://github.com/gkammidi-prog)
