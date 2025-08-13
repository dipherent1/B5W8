import argparse, os, json
import pandas as pd
from typing import List
from .data_utils import load_csv_safe, preprocess_ecommerce
from .model_utils import TrainConfig, build_pipelines_ecommerce, run_training, save_json
from .shap_utils import explain_with_shap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="reports/ecommerce")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp_cost", type=float, default=1.0)
    ap.add_argument("--fn_cost", type=float, default=5.0)
    ap.add_argument("--no_smote", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fraud_path = os.path.join(args.data_dir, "Fraud_Data.csv")
    ip_path = os.path.join(args.data_dir, "IpAddress_to_Country.csv")

    df_fraud = load_csv_safe(fraud_path)
    df_ip = None
    if os.path.exists(ip_path):
        df_ip = load_csv_safe(ip_path)

    df = preprocess_ecommerce(df_fraud, df_ip)

    target = "class"
    cat_cols = ["source", "browser", "sex", "country"]
    cat_cols = [c for c in cat_cols if c in df.columns]  # country may be missing if no IP map
    num_cols = [
        "purchase_value", "age", "hour_of_day", "day_of_week", "time_since_signup_hours",
        "count_user_id_1h", "count_user_id_24h",
        "count_device_id_1h", "count_device_id_24h",
        "count_ip_address_1h", "count_ip_address_24h",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    X = df[cat_cols + num_cols]
    y = df[target]

    cfg = TrainConfig(seed=args.seed, smote=(not args.no_smote), fp_cost=args.fp_cost, fn_cost=args.fn_cost)
    pipes = build_pipelines_ecommerce(cat_cols=cat_cols, num_cols=num_cols, cfg=cfg)
    out = run_training(X, y, pipes, args.out_dir, cfg, model_tag="ecommerce")

    # SHAP on best model for a small sample
    best = out["summary"]["best_model"]
    model_path = os.path.join(args.out_dir, f"ecommerce_{best}.joblib")
    sample = X.sample(min(200, len(X)), random_state=args.seed)
    explain_with_shap(model_path, sample, os.path.join(args.out_dir, "shap"))

    save_json(out["summary"], os.path.join(args.out_dir, "summary_top.json"))
    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()
