import os, json
import numpy as np
import pandas as pd
from typing import Any, Dict
import shap
from joblib import load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def explain_with_shap(model_path: str, X_sample: pd.DataFrame, out_dir: str, max_display: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    pipe = load(model_path)
    model = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    X_trans = pre.transform(X_sample)

    # Choose explainer
    if hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_trans, feature_dependence="independent")

    shap_values = explainer.shap_values(X_trans)

    # Global summary
    plt.figure()
    shap.summary_plot(shap_values, X_trans, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"))
    plt.close()

    # Bar
    plt.figure()
    shap.summary_plot(shap_values, X_trans, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_bar.png"))
    plt.close()

    # Local force plot for top-risk examples
    proba = pipe.predict_proba(X_sample)[:, 1]
    top_idx = np.argsort(-proba)[:3]
    for i, idx in enumerate(top_idx):
        fp = shap.force_plot(
            explainer.expected_value, shap_values[idx, :], X_trans[idx, :],
            matplotlib=True, show=False
        )
        plt.savefig(os.path.join(out_dir, f"shap_force_{i+1}.png"))
        plt.close()

    # Also save feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = pd.DataFrame({"feature": [f"f_{i}" for i in range(X_trans.shape[1])], "importance": importances})
        fi.sort_values("importance", ascending=False).to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
