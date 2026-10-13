# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
This is a fictional, synthetic-data fraud triage capstone.
It cleans a raw payment export, trains a class-imbalance-aware model, retrieves fraud policies, validates structured case notes, and exposes a FastAPI service.
All language-model calls use the supplied offline `mock_llm.py`; no API key or paid service is required.
This is a learning demo, not a real payment decision system.

## Reproduce
Run every command from the repository root using Python 3.11 or newer.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
```

In terminal 1:
```bash
uvicorn app:app --reload
```

In terminal 2:
```bash
python smoke_test.py
```

Open `http://127.0.0.1:8000/docs` to inspect the API. The service writes request logs to `service.log`.

## Part A results
The raw export has 6,090 rows and 90 exact duplicate rows. There are 79 missing amount values and 196 missing distance values. Cleaning leaves 5,919 rows, removes 171 rows, leaves zero missing values, and retains 366 fraud rows. `prepare.py` writes `transactions_features.csv`.

## Part B results
The stratified split contains 3,551 training rows with 220 frauds, 1,184 validation rows with 73 frauds, and 1,184 test rows with 73 frauds. In the development run, the selected validation F1 threshold is 0.87 and validation F1 is 0.5865. Test precision is 0.5507, recall is 0.5205, F1 is 0.5352, average precision is 0.5207, and ROC AUC is 0.9115. The confusion matrix `[[TN, FP], [FN, TP]]` is `[[1080, 31], [35, 38]]`. `train.py` saves `model_bundle.pkl` for the service.

## Part C results
TF-IDF policy retrieval matches all 10 supplied expected results, including rejecting both unrelated queries (10/10). The case-note pipeline validates required JSON fields, allowed priority/team/action values, and the retrieved policy ID. All 10 supplied scenarios produce valid notes after 1–4 attempts.

## Part D results
The service exposes `/health`, `/metrics`, and `/triage`, validates request and response schemas, and logs each request to `service.log`. The smoke test sends 11 requests: valid requests return 200, six malformed-input cases return 422, and the simulated language-model outage fails closed by retaining `hold_for_review`. The committed `service.log` is from one complete smoke-test run.

## Limitations
- The dataset is synthetic and does not represent real merchant or customer behaviour.
- Logistic regression is a baseline and may miss nonlinear fraud patterns.
- The threshold is selected on a small synthetic validation set and needs business-cost review before production.
- TF-IDF retrieval is lexical and can miss semantically related policies.
- The offline mock language model is not a real language model and its behaviour is scripted.
- The service has no authentication, database, distributed rate limiting, or production deployment controls.
- A model probability is not proof of fraud; human review is required for held payments.
