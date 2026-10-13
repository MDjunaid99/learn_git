"""Part B: reproducible fraud model, imbalance handling and threshold selection."""
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from prepare import clean_transactions, add_features

SEED = 2605
TARGET = "is_fraud"
CATEGORICAL = ["merchant_category", "city", "payment_mode"]
NUMERIC = [
    "amount_inr", "hour_of_day", "txn_count_last_24h",
    "avg_amount_30d_inr", "distance_from_home_km", "is_new_device",
    "account_age_days", "amount_to_average_ratio", "is_night",
    "is_high_velocity", "is_new_account", "is_long_distance",
]
FEATURES = CATEGORICAL + NUMERIC


def build_final_model():
    raw = pd.read_csv("transactions_raw.csv")
    data = add_features(clean_transactions(raw))
    # Chargeback is deliberately excluded: it can occur after a payment and leak outcome.
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=SEED, stratify=y_rest
    )
    print("=== SPLIT COUNTS ===")
    for name, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"{name}: rows={len(yy)}, fraud={int(yy.sum())}, non_fraud={int((yy == 0).sum())}")

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = Pipeline([
        ("prep", prep),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=SEED
        )),
    ])
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    # Pick threshold with best F1 on validation; deterministic tie-break favours higher threshold.
    thresholds = np.arange(0.10, 0.91, 0.01)
    scored = [(f1_score(y_val, val_prob >= t, zero_division=0), t) for t in thresholds]
    best_f1, threshold = max(scored, key=lambda x: (x[0], x[1]))
    test_prob = model.predict_proba(X_test)[:, 1]
    pred = (test_prob >= threshold).astype(int)
    print("=== VALIDATION THRESHOLD ===")
    print(f"Chosen threshold: {threshold:.2f}")
    print(f"Validation F1 at threshold: {best_f1:.4f}")
    print("=== TEST METRICS ===")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"Average precision: {average_precision_score(y_test, test_prob):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, test_prob):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]).tolist())
    with open("model_bundle.pkl", "wb") as f:
        pickle.dump({"model": model, "threshold": float(threshold),
                     "features": FEATURES}, f)
    print("Saved model_bundle.pkl")
    return model, float(threshold)


if __name__ == "__main__":
    build_final_model()
