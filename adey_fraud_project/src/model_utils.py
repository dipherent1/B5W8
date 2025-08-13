import os, json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score,
    confusion_matrix, classification_report, auc
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Dataclasses & Helpers
# -----------------------------
@dataclass
class TrainConfig:
    test_size: float = 0.2
    seed: int = 42
    smote: bool = True
    undersample: bool = False
    fp_cost: float = 1.0
    fn_cost: float = 5.0


def train_test_split_stratified(X, y, cfg: TrainConfig):
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)


def evaluate_scores(y_true, y_proba, threshold: Optional[float] = None) -> Dict[str, Any]:
    # PR metrics
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(recall, precision)
    ap = average_precision_score(y_true, y_proba)

    if threshold is None:
        # Choose threshold that maximizes F1
        f1s = []
        for t in np.linspace(0.01, 0.99, 99):
            y_pred = (y_proba >= t).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        best_idx = int(np.argmax(f1s))
        threshold = np.linspace(0.01, 0.99, 99)[best_idx]

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return {
        "auc_pr": float(auc_pr),
        "average_precision": float(ap),
        "threshold": float(threshold),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def plot_pr_curve(pr_curve: Dict[str, List[float]], out_path: str, title: str):
    plt.figure()
    plt.plot(pr_curve["recall"], pr_curve["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Pipeline Builders
# -----------------------------
def build_pipelines_ecommerce(cat_cols: List[str], num_cols: List[str], cfg: TrainConfig):
    # Preprocessors
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    pre = ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", scaler, num_cols),
    ])

    # Models
    logreg = LogisticRegression(max_iter=200, class_weight=None)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=cfg.seed, eval_metric="logloss", n_jobs=-1, tree_method="hist"
    )

    steps_base = [("pre", pre)]
    if cfg.smote:
        steps_base.append(("smote", SMOTE(random_state=cfg.seed)))
    if cfg.undersample:
        steps_base.append(("under", RandomUnderSampler(random_state=cfg.seed)))

    pipe_logreg = ImbPipeline(steps=steps_base + [("clf", logreg)])
    pipe_xgb = ImbPipeline(steps=steps_base + [("clf", xgb)])
    return {"logreg": pipe_logreg, "xgb": pipe_xgb}


def build_pipelines_creditcard(num_cols: List[str], cfg: TrainConfig):
    scaler_amount = RobustScaler()
    pre = ColumnTransformer([
        ("nums", "passthrough", [c for c in num_cols if c != "Amount"]),
        ("amt", scaler_amount, ["Amount"]),
    ])

    logreg = LogisticRegression(max_iter=200, class_weight=None)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=cfg.seed, eval_metric="logloss", n_jobs=-1, tree_method="hist"
    )

    steps_base = [("pre", pre)]
    if cfg.smote:
        steps_base.append(("smote", SMOTE(random_state=cfg.seed)))
    if cfg.undersample:
        steps_base.append(("under", RandomUnderSampler(random_state=cfg.seed)))

    pipe_logreg = ImbPipeline(steps=steps_base + [("clf", logreg)])
    pipe_xgb = ImbPipeline(steps=steps_base + [("clf", xgb)])
    return {"logreg": pipe_logreg, "xgb": pipe_xgb}


# -----------------------------
# Train/Eval Runner
# -----------------------------
def run_training(
    X: pd.DataFrame, y: pd.Series, pipelines: Dict[str, Any],
    out_dir: str, cfg: TrainConfig, model_tag: str
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, cfg)

    results = {}
    best_aucpr = -1.0
    best_name = None

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_scores(y_test.values, y_proba)
        results[name] = metrics

        # Save artifacts
        model_path = os.path.join(out_dir, f"{model_tag}_{name}.joblib")
        dump(pipe, model_path)

        plot_pr_curve(metrics["pr_curve"], os.path.join(out_dir, f"{model_tag}_{name}_pr_curve.png"),
                      title=f"{model_tag.upper()} — {name} (AUC‑PR={metrics['auc_pr']:.3f})")

        # Track best
        if metrics["auc_pr"] > best_aucpr:
            best_aucpr = metrics["auc_pr"]
            best_name = name

    # Persist summary
    save_json(results, os.path.join(out_dir, f"{model_tag}_metrics.json"))
    summary = {
        "best_model": best_name,
        "best_auc_pr": best_aucpr,
        "all_models": list(pipelines.keys())
    }
    save_json(summary, os.path.join(out_dir, f"{model_tag}_summary.json"))
    return {"summary": summary, "results": results, "splits": {"X_train_shape": X_train.shape, "X_test_shape": X_test.shape}}
