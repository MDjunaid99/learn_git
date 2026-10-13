"""Smoke test: expects the API server to be running at localhost:8000."""
import json
import requests

BASE = "http://127.0.0.1:8000"
GOOD = {
    "merchant_category": "Electronics", "city": "Hyderabad", "payment_mode": "Card",
    "amount_inr": 82000, "hour_of_day": 3, "txn_count_last_24h": 8,
    "avg_amount_30d_inr": 12000, "distance_from_home_km": 140,
    "is_new_device": 1, "account_age_days": 10,
    "question": "card payment above INR 50000 from a new device"
}
cases = [
    ("health", "GET", "/health", None),
    ("metrics", "GET", "/metrics", None),
    ("flagged-valid", "POST", "/triage", {**GOOD, "txn_id": "T900001", "fraud_probability": 0.96}),
    ("low-risk", "POST", "/triage", {**GOOD, "txn_id": "T900011", "fraud_probability": 0.05}),
    ("bad-amount-zero", "POST", "/triage", {**GOOD, "txn_id": "T900012", "amount_inr": 0}),
    ("bad-amount-negative", "POST", "/triage", {**GOOD, "txn_id": "T900013", "amount_inr": -1}),
    ("bad-hour", "POST", "/triage", {**GOOD, "txn_id": "T900014", "hour_of_day": 27}),
    ("bad-payment-mode", "POST", "/triage", {**GOOD, "txn_id": "T900015", "payment_mode": "Crypto"}),
    ("missing-question", "POST", "/triage", {k: v for k, v in {**GOOD, "txn_id": "T900016"}.items() if k != "question"}),
    ("bad-probability", "POST", "/triage", {**GOOD, "txn_id": "T900017", "fraud_probability": 1.7}),
    ("llm-outage", "POST", "/triage", {**GOOD, "txn_id": "T900099", "fraud_probability": 0.99}),
]


def main():
    rows = []
    for name, method, path, payload in cases:
        try:
            if method == "GET":
                response = requests.get(BASE + path, timeout=5)
            else:
                response = requests.post(BASE + path, json=payload, timeout=10)
            try:
                body = response.json()
            except ValueError:
                body = {"body": response.text[:120]}
            rows.append((name, response.status_code, body.get("decision", body.get("status", body.get("detail", "ok")))))
        except requests.RequestException as exc:
            rows.append((name, "ERROR", str(exc)))
    print("test | HTTP | result")
    for row in rows:
        print(" | ".join(map(str, row)))
    print(f"Requests sent: {len(rows)}")
    with open("smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump([{"test": r[0], "http": r[1], "result": r[2]} for r in rows], f, indent=2)


if __name__ == "__main__":
    main()
