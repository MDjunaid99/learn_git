"""Part A: audit, clean and engineer the synthetic transaction export."""
import pandas as pd

RAW_FILE = "transactions_raw.csv"
NUMERIC = [
    "amount_inr", "hour_of_day", "txn_count_last_24h",
    "avg_amount_30d_inr", "distance_from_home_km", "is_new_device",
    "account_age_days", "chargeback_filed", "is_fraud",
]


def clean_transactions(raw_df):
    df = raw_df.copy()
    raw_count = len(df)
    df = df.drop_duplicates()
    for col in ["merchant_category", "city", "payment_mode"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip().str.lower().str.title()
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["amount_inr"])
    df = df.loc[df["amount_inr"].between(0.01, 1_000_000)]
    df = df.dropna(subset=["is_fraud", "avg_amount_30d_inr"])
    df["distance_from_home_km"] = df["distance_from_home_km"].fillna(
        df["distance_from_home_km"].median()
    )
    df["txn_count_last_24h"] = df["txn_count_last_24h"].fillna(0)
    df["hour_of_day"] = df["hour_of_day"].fillna(df["hour_of_day"].median())
    df["is_new_device"] = df["is_new_device"].fillna(0)
    df["account_age_days"] = df["account_age_days"].fillna(df["account_age_days"].median())
    df["chargeback_filed"] = df["chargeback_filed"].fillna(0)
    df.attrs["raw_rows"] = raw_count
    return df.reset_index(drop=True)


def add_features(df):
    out = df.copy()
    out["amount_to_average_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"].clip(lower=1)
    out["is_night"] = out["hour_of_day"].between(0, 5).astype(int)
    out["is_high_velocity"] = (out["txn_count_last_24h"] > 6).astype(int)
    out["is_new_account"] = (out["account_age_days"] < 30).astype(int)
    out["is_long_distance"] = (out["distance_from_home_km"] > 100).astype(int)
    return out


def main():
    raw = pd.read_csv(RAW_FILE)
    print("=== DATA AUDIT ===")
    print(f"Raw rows: {len(raw)}")
    print(f"Exact duplicate rows: {int(raw.duplicated().sum())}")
    print(f"Missing amount rows: {int(raw['amount_inr'].isna().sum())}")
    print(f"Missing distance rows: {int(raw['distance_from_home_km'].isna().sum())}")
    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)
    print("=== CLEANING RESULTS ===")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Rows removed: {len(raw) - len(cleaned)}")
    print(f"Missing values remaining: {int(featured.isna().sum().sum())}")
    print(f"Fraud rows: {int(featured['is_fraud'].sum())}")
    featured.to_csv("transactions_features.csv", index=False)
    print("Saved transactions_features.csv")


if __name__ == "__main__":
    main()
