"""Part C: policy-grounded structured case notes with retries and validation."""
import json
from mock_llm import generate
from retrieval import retrieve

ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_TEAMS = {"card_risk", "upi_risk", "customer_care", "merchant_ops"}
ALLOWED_ACTIONS = {"hold_for_review", "block_and_notify"}


def build_prompt(txn_id, fraud_probability, amount_inr, policy):
    return (
        "Write a concise case note using only the supplied policy. Return JSON only.\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {float(fraud_probability):.4f}\n"
        f"AMOUNT_INR: {float(amount_inr):.2f}\n"
        f"POLICY_ID: {policy['policy_id']}\n"
        f"POLICY_TITLE: {policy['title']}\n"
        f"POLICY_TEXT: {policy['text']}\n"
        "Required JSON fields: summary, priority, recommended_team, "
        "recommended_action, policy_id."
    )


def validate(note, expected_policy_id):
    if not isinstance(note, dict):
        return False, "not_object"
    required = {"summary", "priority", "recommended_team", "recommended_action", "policy_id"}
    if not required.issubset(note):
        return False, "missing_field"
    if not isinstance(note["summary"], str) or not note["summary"].strip():
        return False, "invalid_summary"
    if note["priority"] not in ALLOWED_PRIORITIES:
        return False, "invalid_priority"
    if note["recommended_team"] not in ALLOWED_TEAMS:
        return False, "invalid_team"
    if note["recommended_action"] not in ALLOWED_ACTIONS:
        return False, "invalid_action"
    if note["policy_id"] != expected_policy_id:
        return False, "policy_mismatch"
    return True, "ok"


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=5):
    found = retrieve(question)
    policy = found["policy"]
    if policy is None:
        return {"txn_id": txn_id, "status": "no_policy", "policy_id": None,
                "attempts": 0, "case_note": None, "error": "no_policy_above_threshold"}
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy)
    last_error = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = generate(prompt)
            try:
                note = json.loads(raw)
            except json.JSONDecodeError:
                last_error = "invalid_json"
                continue
            ok, reason = validate(note, policy["policy_id"])
            if ok:
                return {"txn_id": txn_id, "status": "ok", "policy_id": policy["policy_id"],
                        "attempts": attempt, "case_note": note, "error": None}
            last_error = reason
        except ConnectionError as exc:
            return {"txn_id": txn_id, "status": "error", "policy_id": policy["policy_id"],
                    "attempts": attempt, "case_note": None, "error": str(exc)}
    return {"txn_id": txn_id, "status": "invalid_after_retries",
            "policy_id": policy["policy_id"], "attempts": max_attempts,
            "case_note": None, "error": last_error}


def main():
    with open("case_scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)
    print("txn_id | policy | status | attempts | error")
    for case in scenarios:
        result = run_case(case["txn_id"], case["fraud_probability"], case["amount_inr"], case["question"])
        print(f'{result["txn_id"]} | {result["policy_id"]} | {result["status"]} | '
              f'{result["attempts"]} | {result["error"]}')


if __name__ == "__main__":
    main()
