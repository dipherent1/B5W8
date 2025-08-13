# Adey Innovations — Dual-Stream Fraud Detection (E‑commerce + Credit Cards)

A production‑style, **two‑week** project scaffold to build, evaluate, and explain fraud detection models for:

1) **E‑commerce transactions** (`Fraud_Data.csv` + `IpAddress_to_Country.csv`)
2) **Bank credit‑card transactions** (`creditcard.csv`)

The pipeline covers data cleaning, geolocation enrichment, feature engineering, **class‑imbalance handling** (SMOTE/undersampling), baseline + ensemble models, evaluation with **AUC‑PR / F1 / Confusion Matrix**, and **SHAP-based explainability** (global + local).

> 🔧 **You bring the data**: drop the three CSVs into `data/` and run the scripts below.

---

## 🗂 Dataset Layout (place files in `data/`)
- `Fraud_Data.csv` — e‑commerce transactions with `class` as target (0=legit, 1=fraud)
- `IpAddress_to_Country.csv` — IP integer ranges to country mapping
- `creditcard.csv` — PCA‑anonymized credit‑card transactions with `Class` target

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) E‑commerce model
```bash
python -m src.train_ecommerce --data_dir data --out_dir reports/ecommerce
```

### 3) Credit‑card model
```bash
python -m src.train_creditcard --data_dir data --out_dir reports/creditcard
```

Outputs include metrics JSON, confusion matrix, PR curve, feature importance CSVs, and SHAP plots.

---

## 📊 Evaluation (Imbalanced‑aware)
We report:
- **AUC‑PR** (primary) — robust under heavy class imbalance
- **F1‑score** (thresholded)
- **Confusion matrix** (TP/FP/FN/TN)
- **Precision‑Recall curve** and threshold sweep utility
  
Tune thresholds by business cost ratios using `--fp_cost` and `--fn_cost` (see scripts).

---

## 🧠 Model Lineup
- **Baseline**: Logistic Regression (interpretable reference)
- **Ensemble**: XGBoost (strong non‑linear model, handles heterogenous features)
  
We select the “best” model per dataset by **AUC‑PR**; ties are broken by cost‑sensitive utility.

---

## 🔍 Explainability
- **Global**: SHAP summary bar/bee swarm plots highlight top features driving fraud.
- **Local**: SHAP force plot for top‑risk cases (saved as HTML).

---

## 🛠 Feature Highlights (E‑commerce)
- Time: `hour_of_day`, `day_of_week`, `time_since_signup_hours`
- Velocity: rolling counts per `user_id`, `device_id`, `ip_address` over **1h** and **24h**
- Geo: map IPv4 to country using as‑of join on lower bounds with upper bound filter

(Credit‑card dataset ships engineered PCA components `V1..V28`; we standardize `Amount`.)

---

## 🧪 Repro Tips
- Keep **resampling on the training split only**.
- Use **stratified** train/test split with temporal holdout if needed.
- Log seed for determinism (`--seed`).

---

## 📁 Project Tree
```
adey_fraud_project/
├─ data/                          # Put CSVs here (not included)
├─ reports/
├─ notebooks/
├─ scripts/
├─ src/
│  ├─ data_utils.py
│  ├─ model_utils.py
│  ├─ shap_utils.py
│  ├─ train_ecommerce.py
│  └─ train_creditcard.py
├─ requirements.txt
├─ Makefile
└─ README.md
```

---

## 📞 Team & Milestones
- Tutors: Mahlet, Rediet, Kerod, Rehmet
- Key Dates (UTC): 
  - Discussion: Wed **16 July 2025** 09:30
  - Interim‑1: Sun **20 July 2025** 20:00
  - Interim‑2: Sun **27 July 2025** 20:00
  - Final: Tue **29 July 2025** 20:00

(Adjust if your local calendar differs.)

---

## 🔐 License
MIT — see `LICENSE`.
