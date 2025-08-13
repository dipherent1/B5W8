import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict

# -----------------------------
# General Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def ipv4_to_int(ip_str: str) -> Optional[int]:
    try:
        parts = [int(p) for p in ip_str.split(".")]
        if len(parts) != 4 or any(p < 0 or p > 255 for p in parts):
            return None
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return None


def map_ip_to_country(transactions: pd.DataFrame, ip_ranges: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dotted IPv4 in `transactions['ip_address']` to integer, then join to
    ip_ranges (with ['lower_bound_ip_address','upper_bound_ip_address','country']).
    We use an asof join on lower bound and then filter by upper bound.
    """
    df = transactions.copy()
    df["ip_int"] = df["ip_address"].astype(str).map(ipv4_to_int).astype("Int64")

    ranges = ip_ranges.copy()
    ranges["lower_int"] = ranges["lower_bound_ip_address"]
    ranges["upper_int"] = ranges["upper_bound_ip_address"]
    ranges = ranges.sort_values("lower_int")

    # asof join expects keys sorted and merges on nearest lower key
    df = df.sort_values("ip_int")
    merged = pd.merge_asof(
        df, ranges[["lower_int", "upper_int", "country"]],
        left_on="ip_int", right_on="lower_int", direction="backward"
    )
    # Keep only rows where ip_int <= upper_int
    matched = merged[(merged["ip_int"].notna()) & (merged["upper_int"].notna()) & (merged["ip_int"] <= merged["upper_int"])]
    unmatched = merged[~merged.index.isin(matched.index)]
    matched["country"] = matched["country"].astype("category")
    # For unmatched, set country as NA
    merged.loc[matched.index, "country"] = matched["country"]
    merged.loc[unmatched.index, "country"] = pd.NA
    merged = merged.drop(columns=["lower_int", "upper_int"])
    return merged.sort_index()


# -----------------------------
# E-commerce Feature Engineering
# -----------------------------
def preprocess_ecommerce(df: pd.DataFrame, ip_ranges: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Clean types, timestamps, add time/velocity features, optional IP->country mapping."""
    df = df.copy()

    # Basic cleaning
    df = df.drop_duplicates()
    # Types
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["purchase_value"] = pd.to_numeric(df["purchase_value"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)

    # Drop rows with essential nulls
    df = df.dropna(subset=["signup_time", "purchase_time", "purchase_value", "age"])

    # Time-based features
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0=Mon
    df["time_since_signup_hours"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600.0

    # Velocity features: counts over 1h and 24h by user/device/ip
    df = df.sort_values("purchase_time")
    for key in ["user_id", "device_id", "ip_address"]:
        df[f"count_{key}_1h"] = (
            df.set_index("purchase_time")
              .groupby(key)["class"]
              .rolling("1H").count()
              .reset_index(level=0, drop=True) - 1
        ).fillna(0).astype(int)
        df[f"count_{key}_24h"] = (
            df.set_index("purchase_time")
              .groupby(key)["class"]
              .rolling("24H").count()
              .reset_index(level=0, drop=True) - 1
        ).fillna(0).astype(int)

    # Optional geo mapping
    if ip_ranges is not None:
        df = map_ip_to_country(df, ip_ranges)

    return df


# -----------------------------
# Creditcard Preprocessing
# -----------------------------
def preprocess_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    # Ensure numeric
    numeric_cols = [c for c in df.columns if c not in ["Class"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols + ["Class"])
    df["Class"] = df["Class"].astype(int)
    return df
