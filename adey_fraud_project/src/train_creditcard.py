import argparse, os, json
import pandas as pd
from typing import List
from .data_utils import load_csv_safe, preprocess_creditcard
from .model_utils import TrainConfig, build_pipelines_creditcard, run_training, save_json
from .shap_utils import explain_with_shap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="reports/creditcard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp_cost", type=float, default=1.0)
    ap.add_argument("--fn_cost", type=float, default=5.0)
    ap.add_argument("--no_smote", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cc_path = os.path.join(args.data_dir, "creditcard.csv")
    df = load_csv_safe(cc_path)
    df = preprocess_creditcard(df)

    target = "Class"
    num_cols = [c for c in df.columns if c != target]

    X = df[num_cols]
    y = df[target]

    cfg = TrainConfig(seed=args.seed, smote=(not args.no_smote), fp_cost=args.fp_cost, fn_cost=args.fn_cost)
    pipes = build_pipelines_creditcard(num_cols=num_cols, cfg=cfg)
    out = run_training(X, y, pipes, args.out_dir, cfg, model_tag="creditcard")

    # SHAP on best model for a small sample
    best = out["summary"]["best_model"]
    model_path = os.path.join(args.out_dir, f"creditcard_{best}.joblib")
    sample = X.sample(min(200, len(X)), random_state=args.seed)
    explain_with_shap(model_path, sample, os.path.join(args.out_dir, "shap"))

    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()
